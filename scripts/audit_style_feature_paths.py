"""Bounded S1 feature parity and reused historical-universe evidence audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.qlib_challenger import build_challenger_dataset, challenger_universe
from moneymore.data.research import load_total_return_stock_bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="20261030")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new evidence directory")
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    universe = challenger_universe(root, store)
    names = [s for s in ["600036.SH", "601398.SH", "600519.SH", "300502.SZ", "688668.SH", "688048.SH", "300757.SZ", "688627.SH"] if s in universe]
    result = {"as_of": args.as_of, "orders_enabled": False,
        "scope": "bounded deterministic 8-stock sample, not full-universe attestation", "features": []}
    for symbol in names:
        try:
            one = {symbol: universe[symbol]}
            offline = build_challenger_dataset(store, one, sequence_length=60, require_label=False, feature_count=12)
            live = build_challenger_dataset(store, one, sequence_length=60, require_label=False, feature_count=12, live_as_of=args.as_of)
            dates = offline.index.get_level_values("datetime")
            same = offline.loc[dates == pd.Timestamp(args.as_of), "feature"]
            latest = live["feature"]
            aligned = same.reindex(index=latest.index, columns=latest.columns)
            error = (aligned - latest).abs()
            lag0 = offline.loc[dates <= pd.Timestamp(args.as_of), "feature"].filter(regex="lag00$")
            scale = {col: {"std": float(lag0[col].std()), "p01": float(lag0[col].quantile(.01)),
                "p99": float(lag0[col].quantile(.99))} for col in lag0}
            bars = load_total_return_stock_bars(store, symbol)
            bars = bars.loc[bars.date.astype(str) <= args.as_of]
            raw = bars.raw_close.astype(float)
            adjusted = bars.close.astype(float)
            events = ((raw.pct_change(fill_method=None).abs() > .20) & (adjusted.pct_change(fill_method=None).abs() < .05))
            examples = bars.loc[events, ["date", "raw_close", "close"]].tail(10).to_dict("records")
            result["features"].append({"symbol": symbol, "offline_rows": len(same), "live_rows": len(latest),
                "prediction_date": str(latest.index.get_level_values("datetime").max().date()),
                "parity_missing_values": int(error.isna().sum().sum()),
                "parity_differences_over_1e_6": int((error > 1e-6).sum().sum()),
                "max_parity_difference": float(error.max().max()), "lag0_scale": scale,
                "raw_vs_adjusted_large_jump_examples": examples,
                "return20_raw_adjusted_difference_at_end": float(raw.pct_change(20, fill_method=None).iloc[-1] - adjusted.pct_change(20, fill_method=None).iloc[-1])})
        except Exception as exc:
            result["features"].append({"symbol": symbol, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
    try:
        report_path = root / "state/baseline-v2-validation/daily-inputs/daily-scores.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frozen_path = Path(report["output_directory"]) / "frozen-inputs.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        pool = Path(frozen["identity_audit"]["output_directory"]) / "identity-top1000-preview.parquet"
        actual_hash = hashlib.sha256(pool.read_bytes()).hexdigest()
        if actual_hash != frozen["pool_sha256"]:
            raise ValueError("historical pool hash mismatch")
        df = pd.read_parquet(pool)
        date_col = next(c for c in ("date", "signal_date", "trade_date") if c in df)
        symbol_col = next(c for c in ("symbol", "ts_code") if c in df)
        counts = df.groupby(date_col)[symbol_col].nunique()
        result["historical_pool"] = {"status": "EXISTING_HASH_BOUND_EVIDENCE_FOUND", "path": str(pool),
            "sha256": actual_hash, "frozen_manifest": str(frozen_path), "rows": len(df),
            "columns": list(df.columns), "dates": len(counts), "start": str(df[date_col].min()),
            "end": str(df[date_col].max()), "min_members": int(counts.min()), "max_members": int(counts.max()),
            "not_yet_proven": ["daily float-cap benchmark weights", "GRU 2015-to-2026 entire training membership coverage",
                               "all upstream membership/source assumptions"]}
    except Exception as exc:
        result["historical_pool"] = {"status": "NOT_VERIFIED", "error": f"{type(exc).__name__}: {exc}"}
    result["status"] = "AUDIT_COMPLETED_NOT_MODEL_APPROVAL"
    result["feature_normalization"] = "no fitted normalization in current QlibPanelDataset; scaling must be evaluated, not assumed"
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps({"status": result["status"], "sample_stocks": len(names),
        "stock_failures": sum(r.get("status") == "FAILED" for r in result["features"]),
        "parity_differences": sum(r.get("parity_differences_over_1e_6", 0) for r in result["features"]),
        "historical_pool": result["historical_pool"]["status"]}))


if __name__ == "__main__":
    main()
