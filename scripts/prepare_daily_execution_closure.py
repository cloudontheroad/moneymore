"""Conservative execution-data closure, NOT future target holdings or a new strategy."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.portfolio_constructor import global_topk_portfolio, trailing_return_correlation
from moneymore.research.baseline_v2_daily_replay import frozen_prices

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    report = json.loads((base / "daily-scores.json").read_text(encoding="utf-8"))
    source = Path(report["output_directory"])
    panel_path = source / "daily-scores.parquet"
    if hashlib.sha256(panel_path.read_bytes()).hexdigest() != report["panel_sha256"]:
        raise ValueError("FROZEN_SCORE_HASH_MISMATCH")
    frozen = json.loads((source / "frozen-inputs.json").read_text(encoding="utf-8"))
    pool_path = (
        Path(frozen["identity_audit"]["output_directory"]) / "identity-top1000-preview.parquet"
    )
    if hashlib.sha256(pool_path.read_bytes()).hexdigest() != frozen["pool_sha256"]:
        raise ValueError("FROZEN_POOL_HASH_MISMATCH")
    panel, pool = pd.read_parquet(panel_path), pd.read_parquet(pool_path)
    prices = frozen_prices(
        root, ParquetStore(root / "data"), frozen, sorted(pool.security_id.unique())
    )
    p = frozen["policy"]["portfolio"]
    if int(p["top_k"]) > int(p["exit_rank"]):
        raise ValueError("CLOSURE_REQUIRES_TOP_K_NO_GREATER_THAN_EXIT_RANK")
    directory = base / "execution-closures" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    rows = []
    for count, (date, frame) in enumerate(panel.groupby("date"), 1):
        correlation = trailing_return_correlation(
            prices.reindex(columns=frame.symbol), date, lookback=int(p["correlation_lookback"])
        )
        for variant in ("fixed", "learned"):
            scores = frame.rename(columns={"score_" + variant: "score"})
            _, _, ranking = global_topk_portfolio(
                scores,
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
            for symbol in ranking.head(int(p["top_k"])).symbol:
                rows.append({"date": date, "variant": variant, "symbol": symbol})
        if count % 50 == 0:
            print(json.dumps({"status": "CLOSURE_SCAN", "days": count}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(directory / "candidate-support.parquet", index=False)
    result = {
        "status": "CONSERVATIVE_EXECUTION_SUPPORT_NOT_TARGETS",
        "output_directory": str(directory),
        "score_panel_sha256": report["panel_sha256"],
        "start": "20240131",
        "end": "20251231",
        "required_symbols": sorted(frame.symbol.unique()),
        "days": panel.date.nunique(),
        "support_sha256": hashlib.sha256(
            (directory / "candidate-support.parquet").read_bytes()
        ).hexdigest(),
        "proof": "Empty initial holdings. With k retained names and top_k >= k, deterministic diversified first top_k contains at least top_k-k nonheld names, covering every new addition; if k>top_k no addition. Retained holdings are previous additions. Corporate actions here issue only same-identity shares.",
        "limitations": [
            "OFFLINE_DATA_ACQUISITION_ONLY_NOT_ADVANCE_TRADING_DECISIONS",
            "NO_ACTUAL_HOLDING_OR_RETURN_CLAIM",
        ],
        "real_orders_enabled": False,
        "production_changed": False,
    }
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(base / "execution-closure.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "required_symbols"}
            | {"required_symbol_count": len(result["required_symbols"])}
        )
    )
