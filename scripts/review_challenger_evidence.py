"""Rebuild derived forward/style evidence without inference, orders or NAV writes."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from moneymore.challenger_diagnostics import run_challenger_review
from moneymore.data.store import ParquetStore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--as-of", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
history_paths = [
    root / "data/processed" / (name + ".parquet")
    for name in (
        "multi_sector_account_daily",
        "qlib_challenger_account_daily",
        "qlib_candidate_account_daily",
    )
]
before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in history_paths}
with sqlite3.connect(
    f"file:{(root / 'state/paper_orders.sqlite3').as_posix()}?mode=ro", uri=True
) as connection:
    fills_before = connection.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
result = run_challenger_review(root, ParquetStore(root / "data"), args.as_of)
after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in history_paths}
with sqlite3.connect(
    f"file:{(root / 'state/paper_orders.sqlite3').as_posix()}?mode=ro", uri=True
) as connection:
    fills_after = connection.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
if before != after or fills_before != fills_after:
    raise RuntimeError(
        "Concurrent ledger mutation detected; rerun diagnostics when services are idle"
    )
print(
    json.dumps(
        {
            "as_of": args.as_of,
            "recommendations": result["recommendations"],
            "history_hashes_unchanged": before == after,
            "fills_before": fills_before,
            "fills_after": fills_after,
        },
        ensure_ascii=False,
        indent=2,
    )
)
