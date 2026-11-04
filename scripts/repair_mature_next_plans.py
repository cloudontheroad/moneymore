"""Refresh the two original A/B close plans, without submitting orders."""
import json
from pathlib import Path
import sqlite3
import sys

from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.qlib_candidate_observer import candidate_catalog, run_candidate_daily

root = Path(__file__).resolve().parents[1]
day = sys.argv[1]
db = root / "state/paper_orders.sqlite3"
def counts():
    with sqlite3.connect(db) as c:
        return tuple(c.execute("SELECT COUNT(*) FROM " + t).fetchone()[0] for t in ("orders", "fills"))
before = counts()
rows = []
for candidate in candidate_catalog(root):
    if candidate["candidate_tag"] not in {"20261103A", "20261103B"}:
        continue
    result = run_candidate_daily(candidate=candidate, root=root,
        store=ParquetStore(root / "data"), broker=PaperBroker(db),
        config=BacktestConfig.from_yaml(root / "configs/default.yaml"), trade_date=day)
    rows.append({"tag": candidate["candidate_tag"], "status": result.status, "report": result.report_path})
assert counts() == before, "Plan-only repair unexpectedly changed orders/fills"
print(json.dumps({"results": rows, "orders_and_fills_unchanged": True}), flush=True)
