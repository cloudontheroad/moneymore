"""Finite, serial metadata queue for exact financial dependencies (not fact approval)."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from collect_public_disclosure_index import collect

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-release"
    source = json.loads((base / "financial-vintage-requirements.json").read_text(encoding="utf-8"))
    paths = [Path(p) for p in source["artifact_hashes"] if p.endswith("-requirements.parquet")]
    if len(paths) != 1 or digest(paths[0]) != source["artifact_hashes"][str(paths[0])]:
        raise ValueError("FINANCIAL_DEPENDENCY_HASH_CHANGED")
    frame = pd.read_parquet(paths[0])
    queue = (
        frame.groupby(["symbol", "ann_date"], as_index=False)
        .agg(
            affected_days=("affected_scored_stock_days", "sum"),
        )
        .sort_values(["affected_days", "symbol", "ann_date"], ascending=[False, True, True])
    )
    results = []
    cached_complete = 0
    for row in queue.itertuples():
        date = pd.Timestamp(row.ann_date)
        start, end = (
            (date - pd.Timedelta(days=1)).strftime("%Y%m%d"),
            (date + pd.Timedelta(days=1)).strftime("%Y%m%d"),
        )
        existing = (
            base
            / "financial-documents/public-index"
            / f"{row.symbol}-{start}-{end}"
            / "result.json"
        )
        if existing.exists():
            previous = json.loads(existing.read_text(encoding="utf-8"))
            if previous["status"] == "INDEX_COMPLETE_FACTS_NOT_REVIEWED":
                # Revalidate cache on use; finished queue metadata alone is not evidence.
                cached_complete += 1
                continue
            if previous["status"] == "STOPPED_NO_RETRY":
                raise ValueError("PREVIOUS_FAILURE_REQUIRES_REVIEW_NOT_AUTOMATIC_RETRY")
        if results:
            time.sleep(2)
        result = collect(root, row.symbol, start, end)
        results.append(
            {
                "symbol": row.symbol,
                "ann_date": row.ann_date,
                "status": result["status"],
                "requests": result["requests"],
                "rows": len(result.get("rows", [])),
            }
        )
        report = {
            "status": "BOUNDED_METADATA_ONLY",
            "dependency_sha256": digest(paths[0]),
            "queue_windows": len(queue),
            "cached_complete_windows_skipped": cached_complete,
            "maximum_jobs": args.jobs,
            "maximum_http_requests": args.jobs * 4,
            "results": results,
            "financial_gate_passed": False,
            "real_orders_enabled": False,
        }
        write_diagnostic(base / "financial-index-progress.json", report)
        print(json.dumps(results[-1]), flush=True)
        if result["status"] != "INDEX_COMPLETE_FACTS_NOT_REVIEWED" or len(results) >= args.jobs:
            break
