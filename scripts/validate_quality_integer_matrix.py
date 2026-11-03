"""All fixed heldout cases, isolated ledgers and independent read-only audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import pandas as pd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allocations',type=Path);parser.add_argument('--calibration',type=Path);parser.add_argument('--inputs',type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_MATRIX')
    root=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True)
    signals=['20260922','20260923','20260924','20260928','20260929']
    probes=['archived_main_gru','archived_candidate_20260812']
    protocol={'scope':'fixed_five_day_two_signal_integer_matrix_independent_cash_entries',
        'signals':signals,'probes':probes,'variants':['quality_unbuffered','quality_buffered'],
        'capital_each_case':10_000_000,'orders_enabled':False,'continuous_holdings':False,'input_version':str(args.inputs) if args.inputs else None,
        'solver_policy':'same_10_second_seed_20_second_target_1percent_gap_no_retries_or_hindsight'}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for signal in signals:
        for probe in probes:
            folder=args.output/f'{signal}-{probe}'
            command=[sys.executable,str(root/'scripts/validate_quality_integer.py'),'--output',str(folder),
                '--signal',signal,'--probe',probe,'--capital','10000000']
            for name in ['allocations','calibration','inputs']:
                if getattr(args,name) is not None:command.extend(['--'+name,str(getattr(args,name))])
            completed=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
            (args.output/f'{signal}-{probe}.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            if completed.returncode:raise RuntimeError('CHILD_FAILED:'+str(folder))
            report=json.loads((folder/'results.json').read_text(encoding='utf-8'))
            for case in report['cases']:
                database=folder/case['variant']/'paper.sqlite3'
                with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as con:
                    fills=pd.read_sql_query('SELECT * FROM fills',con)
                    quantities=dict(con.execute("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0").fetchall())
                    cash=float(con.execute("SELECT cash FROM accounts WHERE account_id='isolated'").fetchone()[0])
                rebuilt=10_000_000+sum((1 if r.side=='SELL' else -1)*r.price*r.quantity-r.fee for r in fills.itertuples())
                if abs(rebuilt-cash)>.01 or len(fills)!=case['fills']:raise ValueError('INDEPENDENT_LEDGER_MISMATCH')
                success=case['plan_status']=='PASS_REFERENCE_QUOTES_ONLY'
                if success:
                    expected={s:q for s,q in case['plan']['target_quantities'].items() if q>0}
                    if quantities!=expected:raise ValueError('INDEPENDENT_QUANTITY_MISMATCH')
                    if any(q%100 or (s.startswith(('688','689')) and q<200) for s,q in quantities.items()):raise ValueError('LOT_RULE_FAILED')
                elif len(fills) or quantities:raise ValueError('BLOCKED_CASE_HAS_TRADES')
                rows.append({**{k:v for k,v in case.items() if k!='plan'},'signal':signal,'probe':probe,
                    'ledger_cash_error':abs(rebuilt-cash),'case_report_sha256':hashlib.sha256((folder/'results.json').read_bytes()).hexdigest()})
            (args.output/'progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            print(json.dumps({'signal':signal,'probe':probe,'completed_cases':len(rows)}),flush=True)
    summary={}
    for variant in protocol['variants']:
        x=[r for r in rows if r['variant']==variant];ok=[r for r in x if r['plan_status']=='PASS_REFERENCE_QUOTES_ONLY']
        summary[variant]={'cases':len(x),'executed':len(ok),'blocked':len(x)-len(ok),'fills':sum(r['fills'] for r in ok),
            'close_all_passed':sum(r['close_economic_breaches']==0 and r['close_quality_breaches']==0 for r in ok) if ok else None,
            'economic_breach_cases':sum(r['close_economic_breaches']>0 for r in ok) if ok else None,
            'quality_breach_cases':sum(r['close_quality_breaches']>0 for r in ok) if ok else None}
    (args.output/'results.json').write_text(json.dumps({'summary':summary,'cases':rows,'adopted':False,'orders_enabled':False},indent=2,allow_nan=False))
    print(json.dumps(summary))


if __name__=='__main__':main()
