"""Current-pool style ablations; no sector quotas, account writes or return selection."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.portfolio_constructor import global_topk_portfolio, trailing_return_correlation
from moneymore.research.baseline_v2_daily_scores import score_pair
from moneymore.research.baseline_v2_model import build_v2_snapshot, refresh_official_basic
from moneymore.research.baseline_v2_release import digest, read_json
from moneymore.research.sector_model import build_global_factor_snapshot
from moneymore.research.security_identity import SecurityIdentity
from moneymore.research.style_v2_experiment import OverlayStore
from moneymore.strategy_universe import active_strategy_universe


def exposure(frame, weights=None):
    weights = weights or {s: 1 / len(frame) for s in frame.symbol}
    core = frame.industry.astype(str).isin(["电子", "计算机", "通信"])
    growth = frame.symbol.str.startswith(("30", "68"))
    return {
        "stocks": len(frame),
        "growth_board_weight": float(sum(weights.get(s, 0) for s in frame.loc[growth, "symbol"])),
        "core_information_tech_weight": float(
            sum(weights.get(s, 0) for s in frame.loc[core, "symbol"])
        ),
        "industries": {
            str(k): float(sum(weights.get(s, 0) for s in f.symbol))
            for k, f in frame.groupby("industry", dropna=False)
        },
    }


def main():
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    as_of = str(store.read("daily", columns=["trade_date"]).trade_date.max())
    prior = max(d for d in trading_dates(store, as_of) if d < as_of)
    refresh_official_basic(root, store, prior)
    policy = yaml.safe_load((root / "configs/baseline_v2.yaml").read_text(encoding="utf-8"))
    refit = read_json(root / "state/baseline-v2-release/corrected-refit.json")
    overlay_report = read_json(root / "state/baseline-v2-release/adjustment-overlay.json")
    overlay_path = Path(overlay_report["output_directory"]) / "native-with-derived-factors.parquet"
    changes = yaml.safe_load((root / "configs/baseline_v2_market_evidence.yaml").read_text(encoding="utf-8"))["symbol_changes"]
    derived_store = OverlayStore(store, overlay_path, refit["overlay_sha256"], SecurityIdentity(changes))
    _, variants, capture, exclusions = build_v2_snapshot(root, derived_store, as_of, policy)
    universe = active_strategy_universe(store).set_index("symbol")
    old = build_global_factor_snapshot(store, sorted(universe.index), policy["factors"])
    old["industry"] = old.symbol.map(universe.industry)
    old["name"] = old.symbol.map(universe.name)
    weights_path = Path(refit["directory"]) / "state/baseline-v2-validation/weight-validation.json"
    weights = read_json(weights_path)
    eligible = capture["raw"].loc[~capture["raw"].symbol.isin(exclusions)]
    learned = score_pair(eligible, policy, weights)
    learned["score"] = learned.score_learned
    frames = {
        "legacy_current_data_path": old,
        "vendor_repaired_legacy_formula": capture["repaired_legacy"],
        **variants,
        "learned_industry_size_comparable": learned,
    }
    directory = (
        root / "state/baseline-v2-style-release" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True, exist_ok=False)
    capture["raw"].to_parquet(directory / "raw-factors.parquet", index=False)
    capture["financial_provenance"].to_parquet(directory / "financial-input.parquet", index=False)
    capture["prices"].to_parquet(directory / "prices.parquet")
    raw = capture["raw"].copy()
    absent = sorted(set(universe.index) - set(raw.symbol))
    if absent:
        missing = universe.loc[absent].reset_index()[["symbol", "name", "industry", "board"]]
        raw = pd.concat([raw, missing], ignore_index=True)
    raw["excluded"] = raw.symbol.isin(exclusions)
    raw["exclusion_reasons"] = raw.symbol.map(lambda s: ";".join(exclusions.get(s, [])))
    raw.to_parquet(directory / "eligibility.parquet", index=False)
    prices = capture["prices"].copy()
    prices.index = pd.to_datetime(prices.index)
    p = policy["portfolio"]
    correlation = trailing_return_correlation(
        prices, as_of, lookback=int(p["correlation_lookback"])
    )
    results = {}
    for name, frame in frames.items():
        frame = frame.copy()
        if "industry" not in frame:
            frame["industry"] = frame.symbol.map(universe.industry)
        selected, targets, ranking = global_topk_portfolio(
            frame,
            set(),
            symbol_column="symbol",
            top_k=int(p["top_k"]),
            exit_rank=int(p["exit_rank"]),
            max_replacements=int(p["max_replacements"]),
            gross_exposure=float(p["gross_exposure"]),
            minimum_weight=float(p["minimum_weight"]),
            maximum_weight=float(p["maximum_weight"]),
            correlation=correlation,
            cluster_correlation_threshold=float(p["cluster_correlation_threshold"]),
            maximum_cluster_members=int(p["maximum_cluster_members"]),
            weighting_method=p["weighting_method"],
        )
        ranking["selected"] = ranking.symbol.isin(selected)
        ranking["target_weight"] = ranking.symbol.map(targets).fillna(0)
        ranking["name"] = ranking.symbol.map(universe.name)
        ranking.to_parquet(directory / (name + ".parquet"), index=False)
        results[name] = {
            "eligible": len(frame),
            **exposure(ranking.loc[ranking.selected], targets),
            "targets": targets,
        }
    report = {
        "status": "CURRENT_POOL_STYLE_AUDIT_NOT_PROMOTION",
        "as_of": as_of,
        "directory": str(directory),
        "source_weights": str(weights_path),
        "source_weights_sha256": digest(weights_path),
        "overlay_path": str(overlay_path),
        "overlay_sha256": refit["overlay_sha256"],
        "identity_policy": changes,
        "universe_size": len(universe),
        "eligible": len(eligible),
        "excluded": len(exclusions),
        "all_pool": exposure(raw),
        "excluded_pool": exposure(raw.loc[raw.excluded]) if raw.excluded.any() else {},
        "variants": results,
        "core_tech_definition": ["电子", "计算机", "通信"],
        "board_is_not_industry": True,
        "sector_quota": False,
        "production_changed": False,
        "real_orders_enabled": False,
        "known_weight_research_failures": weights["gate_failures"],
    }
    write_diagnostic(directory / "report.json", report)
    write_diagnostic(root / "state/baseline-v2-style-release/latest.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
