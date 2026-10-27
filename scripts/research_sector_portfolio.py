from pathlib import Path

import yaml

from moneymore.data.store import ParquetStore
from moneymore.portfolio_constructor import (
    adjusted_close_panel,
    global_topk_portfolio,
    trailing_return_correlation,
)
from moneymore.research.sector_model import (
    build_global_factor_snapshot,
)
from moneymore.strategy_universe import active_strategy_universe
from moneymore.research.score_evidence import archive_score_evidence

ROOT = Path(__file__).resolve().parents[1]
STORE = ParquetStore(ROOT / "data")
CONFIG = yaml.safe_load(
    (ROOT / "configs" / "sector_models.yaml").read_text(encoding="utf-8")
)
active_universe = active_strategy_universe(STORE)
symbol_sectors = dict(
    zip(
        active_universe["symbol"].astype(str),
        active_universe["industry"].fillna("未分类").astype(str),
    )
)
global_config = CONFIG["global_selection"]
audit_capture = {}
global_scores = build_global_factor_snapshot(
    STORE,
    list(symbol_sectors),
    global_config["factors"],
    audit_capture=audit_capture,
)
global_correlation = trailing_return_correlation(
    adjusted_close_panel(STORE, list(symbol_sectors)),
    global_scores["date"].max(),
    int(global_config["correlation_lookback"]),
)
try:
    previous_global = STORE.read("global_factor_recommendation")
    previous_holdings = set(
        previous_global.loc[previous_global["selected"].astype(bool), "symbol"].astype(str)
    )
except FileNotFoundError:
    previous_holdings = set()
holdings, portfolio_weights, ranked = global_topk_portfolio(
    global_scores[["date", "symbol", "score"]],
    previous_holdings,
    symbol_column="symbol",
    top_k=int(global_config["top_k"]),
    exit_rank=int(global_config["exit_rank"]),
    max_replacements=int(global_config["max_replacements"]),
    minimum_weight=float(global_config["minimum_weight"]),
    maximum_weight=float(global_config["maximum_weight"]),
    correlation=global_correlation,
    correlation_penalty=float(global_config["correlation_penalty"]),
    cluster_correlation_threshold=float(global_config["cluster_correlation_threshold"]),
    maximum_cluster_members=int(global_config["maximum_cluster_members"]),
)
ranked["selected"] = ranked["symbol"].astype(str).isin(holdings)
ranked["target_weight"] = ranked["symbol"].astype(str).map(portfolio_weights).fillna(0.0)
ranked["rank"] = ranked["global_rank"]
ranked["sector"] = ranked["symbol"].map(symbol_sectors).fillna("unmapped")
ranked["model_id"] = "global_multifactor_v1"

# Persist evidence before publishing recommendations; archive failure must not silently publish.
archive_score_evidence(ROOT, audit_capture, ranked, global_config)

STORE.merge_curated(
    "global_factor_recommendation",
    [ranked],
    ["symbol"],
)
print(ranked.loc[ranked["selected"]].to_string(index=False))
