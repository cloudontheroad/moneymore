"""Isolated next-open fixture. Never reads future/live quotes or sends real orders."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd
from threadpoolctl import threadpool_limits
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.portfolio_candidate import execute_portfolio_candidate_open
from moneymore.qlib_candidate_observer import candidate_catalog
from release_cost_ab import IsolatedHistory

root = Path(__file__).resolve().parents[1]
folder = Path(tempfile.mkdtemp(prefix="next-open-20261105-", dir=root / "state"))
database = folder / "state/paper_orders.sqlite3"
database.parent.mkdir()
with sqlite3.connect(f"file:{(root/'state/paper_orders.sqlite3').as_posix()}?mode=ro", uri=True) as src, sqlite3.connect(database) as dst:
    src.backup(dst)
clock = datetime(2026, 11, 5, 9, 31, tzinfo=ZoneInfo("Asia/Shanghai"))
store = ParquetStore(root / "data")
marks = store.read("daily", filters=[("trade_date", "==", "20261104")]).set_index("ts_code").close.to_dict()
class Fixture:
    name = "ISOLATED_NOV4_CLOSE_FIXTURE_NOT_REAL_NOV5_QUOTES"
    def intraday_quotes(self, symbols):
        return pd.DataFrame([dict(symbol=s,last=marks[s],timestamp=clock.timestamp()*1000,
            up_limit=marks[s]*2,down_limit=marks[s]*.5,execution_ready=True) for s in symbols])
rows = []
with threadpool_limits(limits=4):
    for candidate in candidate_catalog(root):
        tag = candidate["candidate_tag"]
        if tag not in {"20261103A","20261103B","20261104CA","20261104CB"}:
            continue
        target = folder / "state/qlib-portfolio-plans" / tag
        target.mkdir(parents=True)
        shutil.copy2(root / "state/qlib-portfolio-plans" / tag / "20261104.json", target / "20261104.json")
        args = dict(root=folder,store=IsolatedHistory(store,ParquetStore(folder/"data")),broker=PaperBroker(database),
            config=BacktestConfig.from_yaml(root/"configs/default.yaml"),candidate=candidate,provider=Fixture(),trade_date="20261105",now=clock)
        result = execute_portfolio_candidate_open(**args)
        count = len(args["broker"].fills(candidate["account_id"]))
        duplicate = execute_portfolio_candidate_open(**args)
        assert count == len(args["broker"].fills(candidate["account_id"])), "Repeated fixture duplicated fills"
        rows.append(dict(tag=tag,status=result["status"],fills=count,reconciliation=result.get("reconciliation"),
            reason=result.get("plan",{}).get("reason"),repeat_status=duplicate["status"]))
        print(json.dumps(rows[-1]),flush=True)
(folder/"results.json").write_text(json.dumps(dict(fixture_only=True,real_orders_sent=0,rows=rows),indent=2))
print("EVIDENCE",folder,flush=True)
assert len(rows)==4 and all(r["status"]=="COMPLETED" and r["reconciliation"]["matched"] for r in rows)
