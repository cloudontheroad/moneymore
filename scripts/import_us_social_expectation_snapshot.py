"""Store immutable, point-in-time expectation and social-sentiment snapshots.

The input JSONL is produced by a market-wide WebSearch discovery pass. Search
result counts are not accepted as mention counts; platform-derived counts must
declare their measurement window and source URLs.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/us_overnight_event_warehouse/social_expectations"
UNIVERSE = max((ROOT / "data/us_overnight_paper_only/universe").glob("top1000-*.parquet"))
NUMERIC = [
    "mention_count", "mention_acceleration_24h_vs_7d", "unique_author_count",
    "bullish_share", "bearish_share", "net_sentiment", "sentiment_change_24h",
    "disagreement", "source_diversity", "days_to_earnings",
    "eps_consensus", "eps_revision_7d", "eps_revision_30d",
    "revenue_consensus", "revenue_revision_7d", "revenue_revision_30d",
    "implied_move", "iv_rank",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    allowed = set(pd.read_parquet(UNIVERSE)["code"].astype(str))
    rows = []
    for line_number, line in enumerate(args.jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {"symbol", "snapshot_at_utc", "window_start_utc", "source_urls"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing {sorted(missing)}")
        symbol = "US." + str(item["symbol"]).upper().removeprefix("US.")
        if symbol not in allowed:
            continue
        if not item["source_urls"]:
            raise ValueError(f"line {line_number}: source_urls cannot be empty")
        if item.get("measurement_method") == "web_search_result_count":
            raise ValueError(f"line {line_number}: search result counts are not platform mentions")
        row = {**item, "symbol": symbol}
        row["snapshot_at_utc"] = pd.to_datetime(row["snapshot_at_utc"], utc=True)
        row["window_start_utc"] = pd.to_datetime(row["window_start_utc"], utc=True)
        for column in NUMERIC:
            row[column] = pd.to_numeric(row.get(column), errors="coerce")
        rows.append(row)
    if not rows:
        print(json.dumps({"imported": 0}))
        return 0
    frame = pd.DataFrame(rows).sort_values(["snapshot_at_utc", "symbol"])
    OUT.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(UTC).isoformat()
    stamp = captured.replace(":", "").replace("-", "")
    snapshot = OUT / f"{stamp}.parquet"
    frame.to_parquet(snapshot, index=False)
    canonical = OUT / "canonical.parquet"
    if canonical.exists():
        frame = pd.concat([pd.read_parquet(canonical), frame], ignore_index=True, sort=False)
    frame = frame.drop_duplicates(["snapshot_at_utc", "symbol"], keep="last")
    frame.to_parquet(canonical, index=False)
    print(json.dumps({"imported": len(rows), "canonical_rows": len(frame), "snapshot": str(snapshot)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
