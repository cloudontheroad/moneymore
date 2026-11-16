"""Clear an already-passed data-failure latch, then generate prior-close intents.

No risk threshold change and no historical execution. Requires no active recovery.
"""
import json
from pathlib import Path
from urllib.request import urlopen
from dotenv import load_dotenv
from moneymore.execution.paper import PaperBroker
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.multi_sector_daily import run_multi_sector_daily, MULTI_SECTOR_ACCOUNT
from moneymore.challenger_diagnostics import write_diagnostic

if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1];load_dotenv(root/'.env')
    with urlopen('http://127.0.0.1:8788/api/tasks',timeout=20) as response: tasks=json.load(response)
    day='20261110'
    broker=PaperBroker(root/'state/paper_orders.sqlite3')
    checked=run_multi_sector_daily(store=ParquetStore(root/'data'),broker=broker,
        config=BacktestConfig.from_yaml(root/'configs/default.yaml'),trade_date=day,
        signal_dir=root/'state/multi-sector-signals',report_dir=root/'state/multi-sector-shadow',planning_only=True)
    state=broker.risk_state(MULTI_SECTOR_ACCOUNT)
    if (checked.data_health.get('status')!='HEALTHY' or not broker.reconcile(MULTI_SECTOR_ACCOUNT).matched
            or state['proposed_state']!='NORMAL' or state['reason_code']!='RISK_CHECKS_PASS'):
        raise RuntimeError('LATEST_RISK_AND_DATA_MUST_PASS')
    before=state
    state=broker.approve_risk_recovery(MULTI_SECTOR_ACCOUNT,operator='USER_REQUESTED_RESTART_RECOVERY_20261111')
    result=run_multi_sector_daily(store=ParquetStore(root/'data'),broker=broker,
        config=BacktestConfig.from_yaml(root/'configs/default.yaml'),trade_date=day,
        signal_dir=root/'state/multi-sector-signals',report_dir=root/'state/multi-sector-shadow',planning_only=True)
    write_diagnostic(root/'state/restart-recovery-20261111/baseline.json',dict(before=before,after=state,
        status=result.status,signal_date=day,orders=len(result.orders),executions=len(result.executions),real_orders_enabled=False))
    print(json.dumps(dict(status=result.status,orders=len(result.orders),executions=len(result.executions))),flush=True)
