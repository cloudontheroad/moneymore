"""Latest completed close plans only; no invented historical orders or fills."""
import json
import sqlite3
import traceback
from pathlib import Path
from dotenv import load_dotenv
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.qlib_candidate_observer import candidate_catalog, run_candidate_daily
from moneymore.challenger_diagnostics import write_diagnostic

if __name__ == '__main__':
    import sys
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    day = sys.argv[1]
    tags=set(sys.argv[2:]) or {'20261027','20261029','20261103A','20261103B','20261104CA','20261104CB'}
    broker = PaperBroker(root / 'state/paper_orders.sqlite3')
    rows = []
    for candidate in candidate_catalog(root):
        if candidate['candidate_tag'] not in tags:
            continue
        try:
            if candidate.get('validation_hold'):
                raise ValueError('RESEARCH_GATE_BLOCKED')
            result = run_candidate_daily(candidate=candidate,root=root,store=ParquetStore(root/'data'),broker=broker,
                config=BacktestConfig.from_yaml(root/'configs/default.yaml'),trade_date=day)
            row = dict(tag=candidate['candidate_tag'],status=result.status,report=result.report_path)
        except Exception as exc:
            traceback.print_exc()
            row = dict(tag=candidate['candidate_tag'],status='FAILED',error=str(exc))
        rows.append(row)
        print(json.dumps(row),flush=True)
        write_diagnostic(root/'state/restart-recovery-20261111'/('plans.json' if len(tags)>1 else 'plan-'+next(iter(tags))+'.json'),dict(signal_date=day,results=rows,real_orders_enabled=False))
