"""Run isolated existing paper intraday branches after current system gate check."""
import json
from pathlib import Path
from urllib.request import urlopen
from dotenv import load_dotenv
from moneymore.intraday_execution import run_baseline_intraday_once, INTRADAY_ACCOUNT, EXPOSURE_INTRADAY_ACCOUNT
from moneymore.baseline_exposure_daily import BASELINE_EXPOSURE_ACCOUNT
from moneymore.execution.isolation import isolated_execution_call
from moneymore.challenger_diagnostics import write_diagnostic

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    with urlopen('http://127.0.0.1:8788/api/system-health',timeout=20) as response:
        health = json.load(response)
    if health['trading_gate'] != 'OPEN':
        raise RuntimeError('SYSTEM_TRADING_GATE_NOT_OPEN')
    results = [isolated_execution_call(INTRADAY_ACCOUNT,lambda:run_baseline_intraday_once(root=root)),
        isolated_execution_call(EXPOSURE_INTRADAY_ACCOUNT,lambda:run_baseline_intraday_once(root=root,
            source_account=BASELINE_EXPOSURE_ACCOUNT,account_id=EXPOSURE_INTRADAY_ACCOUNT,
            snapshot_table='baseline_pysystemtrade_intraday_snapshots',tick_table='baseline_pysystemtrade_intraday_account_ticks',
            daily_table='baseline_pysystemtrade_intraday_account_daily',audit_subdirectory='baseline-pysystemtrade-intraday-execution'))]
    write_diagnostic(root/'state/restart-recovery-20261111/intraday.json',dict(results=results,real_orders_enabled=False))
    print(json.dumps([dict(account=r.get('account_id'),status=r.get('status'),error=r.get('error'),executions=len(r.get('executions',[]))) for r in results]),flush=True)
