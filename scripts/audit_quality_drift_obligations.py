"""Follow actual isolated holdings to next open; no orders or invented repairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.style_missing_controls import uncertain_controls


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_DRIFT_AUDIT')
    root=Path(__file__).resolve().parents[1];parent=root/'state/style-balance/real-archive-group-decomposition-20261102-b'
    report=json.loads((args.study/'results.json').read_text())
    bars=pd.read_parquet(parent/'source-price-history.parquet').set_index(['trade_date','ts_code'])
    liquidity=pd.read_parquet(parent/'registered-liquidity.parquet').set_index(['date','symbol']).turnover_mean_20
    transitions={'20260922':'20260923','20260923':'20260924','20260924':'20260928','20260928':'20260929','20260929':'20260930'}
    following={'20260923':'20260924','20260924':'20260928','20260928':'20260929','20260929':'20260930'}
    rows=[];db_hashes={}
    for case in report['cases']:
        if case['plan_status']!='PASS_REFERENCE_QUOTES_ONLY':continue
        day=transitions[case['signal']];next_day=following.get(day)
        folder=args.study/f"{case['signal']}-{case['probe']}"
        if hashlib.sha256((folder/'results.json').read_bytes()).hexdigest()!=case['case_report_sha256']:raise ValueError('CASE_REPORT_CHANGED')
        db=folder/case['variant']/'paper.sqlite3';db_hashes[str(db)]=hashlib.sha256(db.read_bytes()).hexdigest()
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as con:
            quantities=pd.Series(dict(con.execute("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0").fetchall()),dtype=float)
        raw=pd.read_parquet(root/f'state/style-balance/preperiod-drift-202609-b/observations-{day}.parquet')
        raw=raw.drop(columns=['turnover_rate']).assign(turnover_mean_20=liquidity.loc[day].reindex(raw.index))
        c=uncertain_controls(raw,CORE)[0]
        for field in raw.select_dtypes(include='number').columns:
            if field not in CORE:c['unknown_'+field]=raw[field].replace([np.inf,-np.inf],np.nan).isna().astype(float)
        if set(quantities.index)-set(c.index):raise ValueError('HELD_CLOSE_RISK_UNOBSERVABLE')
        wb=np.exp(raw.log_size);wb/=wb.sum();centered=c-(wb @ c)
        values=quantities*bars.loc[day].close.reindex(quantities.index)
        if not np.isfinite(values).all() or (values<=0).any():raise ValueError('INVALID_CLOSE_MARKS')
        close=(values/values.sum()) @ centered.reindex(quantities.index)
        for field,exposure in close.items():
            limit=.01 if field.startswith('unknown_') else .03 if field.startswith(('industry_','board_')) else .1
            excess=exposure-limit if field.startswith('unknown_') else abs(exposure)-limit
            if excess<=1e-6:continue
            row={'close_date':day,'next_open':next_day,'signal':case['signal'],'probe':case['probe'],'variant':case['variant'],
                'factor':field,'close_active':float(exposure),'limit':limit,'close_excess':float(excess),'repair_executed':False}
            if next_day is None:row['status']='TERMINAL_NEXT_SESSION_OUTSIDE_EVIDENCE'
            else:
                q=bars.loc[next_day].reindex(quantities.index)
                if not np.isfinite(q[['open','adj_factor']]).all().all() or (q.open<=0).any():row['status']='UNMEASURABLE_NEXT_QUOTES'
                elif not np.allclose(q.adj_factor,bars.loc[day].adj_factor.reindex(quantities.index),atol=1e-12,rtol=0):
                    row['status']='ACTION_EVIDENCE_REQUIRED_BEFORE_CARRY_FORWARD'
                else:
                    value=quantities*q.open;active=(value/value.sum()) @ centered.reindex(quantities.index)
                    next_excess=active[field]-limit if field.startswith('unknown_') else abs(active[field])-limit
                    row.update(next_open_active=float(active[field]),next_open_excess=float(max(0.,next_excess)),
                        status='NATURALLY_CLEAR_AT_NEXT_OPEN' if next_excess<=1e-6 else 'REPAIR_REQUIRED')
            rows.append(row)
    for path,digest in db_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('SOURCE_LEDGER_CHANGED')
    summary={}
    for variant in ['quality_unbuffered','quality_buffered']:
        x=[r for r in rows if r['variant']==variant]
        summary[variant]={'close_factor_alerts':len(x),'statuses':{s:sum(r['status']==s for r in x) for s in sorted({r['status'] for r in x})}}
    args.output.mkdir(parents=True)
    (args.output/'results.json').write_text(json.dumps({'summary':summary,'obligations':rows,'source_ledger_hashes':db_hashes,
        'scope':'unchanged_real_quantities_next_open_prior_close_factors_and_reference','orders_enabled':False,
        'corporate_action_limitation':'no_adjustment_change_is_not_complete_official_event_verification','not_claimed':'repair_executed_or_continuous_strategy'},indent=2,allow_nan=False))
    print(json.dumps(summary))


if __name__=='__main__':main()
