"""Reuse hash-bound PIT membership for a limited total-cap benchmark, not alpha."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("benchmark evidence is immutable")
    root = Path(__file__).resolve().parents[1]
    audit_path = root / "state/style-balance/s1-feature-paths-20261102-a/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    pool = audit["historical_pool"]
    path = Path(pool["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != pool["sha256"]:
        raise ValueError("pool evidence changed")
    frame = pd.read_parquet(path)
    signal = frame.signal_date.astype(str)
    source = frame.trade_date.astype(str)
    if (source >= signal).any() or frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("source is not strictly prior or membership duplicated")
    if not frame.groupby("signal_date").ts_code.nunique().eq(1000).all():
        raise ValueError("unexpected pool size")
    cap = pd.to_numeric(frame.total_mv, errors="coerce")
    if cap.isna().any() or (cap <= 0).any():
        raise ValueError("market cap missing or nonpositive")
    weights = frame[["signal_date", "trade_date", "ts_code", "security_id"]].copy()
    weights["weight"] = cap / cap.groupby(signal).transform("sum")
    if not weights.groupby("signal_date").weight.sum().between(1-1e-10, 1+1e-10).all():
        raise ValueError("weight budget mismatch")
    args.output.mkdir(parents=True)
    target = args.output / "weights.parquet"
    weights.to_parquet(target, index=False)
    manifest = {"status": "LIMITED_RESEARCH_REFERENCE_FROZEN_NOT_PUBLICATION_BENCHMARK",
        "method": "previous_session_total_market_cap_weights_within_historical_top1000",
        "not_float_cap": True, "source_pool_sha256": pool["sha256"],
        "weights_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "start": str(signal.min()), "end": str(signal.max()), "sessions": int(signal.nunique()),
        "rows": len(weights), "source_strictly_prior": True, "orders_enabled": False,
        "scope": "2024-2025 existing known exploration only; no claim for 2015-2026 training pool",
        "remaining": ["free-float cap inputs", "upstream pool ranking reconstruction", "2026/current extension", "industry and named style matrix"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
