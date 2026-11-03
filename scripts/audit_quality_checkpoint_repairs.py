"""Independent read-only cash, stock, T+1 and postfill risk reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from moneymore.research.quality_contract import quality_upper_only_bounds
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.frozen_risk_scale import CORE


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_AUDIT')
    root=Path(__file__).resolve().parents[1];protocol=json.loads((args.study/'protocol.json').read_text());report=json.loads((args.study/'results.json').read_text())
    version=Path(protocol['input_version']);calibration=Path(protocol['calibration'])
    envelope=json.loads((calibration/'observed-error-envelope.json').read_text())
    liquidity=pd.read_parquet(calibration/'registered-liquidity.parquet').set_index(['date','symbol']).turnover_mean_20
    rows=[]
    for row in report['cases']:
        key=row['checkpoint'];day=row['signal_date'];date=row['execution_date']
        source=root/'state/style-balance/quality-repair-actions-20261103-a'/key/'paper.sqlite3'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['source_ledger_sha256']:raise ValueError('SOURCE_CHANGED')
        with sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True) as con:
            held=dict(con.execute("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0").fetchall())
            initial=float(con.execute("SELECT cash FROM accounts WHERE account_id='isolated'").fetchone()[0])
        with sqlite3.connect((args.study/key/'paper.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as con:
            fills=pd.read_sql_query('SELECT * FROM fills WHERE trade_date=?',con,params=(date,))
            actual=dict(con.execute("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0").fetchall())
            cash=float(con.execute("SELECT cash FROM accounts WHERE account_id='isolated'").fetchone()[0])
        rebuilt=initial;quantities=dict(held)
        for fill in fills.itertuples():
            if fill.quantity%100 or (fill.side=='BUY' and fill.symbol.startswith(('688','689')) and fill.quantity<200):raise ValueError('LOT_RULE_BREACH')
            if fill.side=='SELL' and fill.quantity>held.get(fill.symbol,0):raise ValueError('T1_PREVIOUS_HOLDINGS_EXCEEDED')
            sign=1 if fill.side=='BUY' else -1;quantities[fill.symbol]=quantities.get(fill.symbol,0)+sign*fill.quantity
            rebuilt-=sign*fill.price*fill.quantity+fill.fee
        quantities={s:q for s,q in quantities.items() if q>0}
        if quantities!=actual or abs(rebuilt-cash)>.01 or len(fills)!=row['fills']:raise ValueError('INDEPENDENT_LEDGER_MISMATCH')
        if not row['repaired']:
            if len(fills):raise ValueError('BLOCKED_REPAIR_HAS_FILLS')
            rows.append({'checkpoint':key,'verified_blocked':True});continue
        raw=pd.read_parquet(version/f'observations-{day}.parquet')
        raw=raw.drop(columns=['turnover_rate']).assign(turnover_mean_20=liquidity.loc[day].reindex(raw.index));c=uncertain_controls(raw,CORE)[0]
        for field in raw.select_dtypes(include='number').columns:
            if field not in CORE:c['unknown_'+field]=raw[field].replace([np.inf,-np.inf],np.nan).isna().astype(float)
        wb=np.exp(raw.log_size);wb/=wb.sum();limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
        lo,hi=quality_upper_only_bounds(c,wb,limits,envelope)
        quote_file=args.study/f'official-daily-{date}.parquet' if date!='20261008' else root/'state/style-balance/quality-repair-evidence-20261103-a/daily-20261008.parquet'
        quotes=pd.read_parquet(quote_file).set_index('ts_code').open
        values=pd.Series(actual)*quotes.reindex(actual);active=(values/values.sum()) @ (c-(wb @ c)).reindex(actual)
        excess=max(float((active-hi).max()),float((lo-active).max()))
        if not np.isfinite(active).all() or excess>1e-5:raise ValueError('INDEPENDENT_REPAIR_RISK_FAILED')
        rows.append({'checkpoint':key,'fills':len(fills),'cash_error':abs(rebuilt-cash),'positions_match':True,'maximum_risk_excess':excess})
    args.output.write_text(json.dumps({'status':'PASS_INDEPENDENT_ENGINEERING_AUDIT','cases':rows,
        'source_report_sha256':hashlib.sha256((args.study/'results.json').read_bytes()).hexdigest(),'orders_enabled':False},indent=2))
    print(json.dumps(rows))


if __name__=='__main__':main()
