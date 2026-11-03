"""Separate turnover infeasibility from integer optimality budget failures."""
import argparse
import json
from pathlib import Path
import sqlite3
import cvxpy as cp
import numpy as np
import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.quality_contract import quality_upper_only_bounds
from moneymore.research.replay_attribution import reconstruct_positions


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_BLOCK_DIAGNOSIS')
    root=Path(__file__).resolve().parents[1];report=json.loads((args.study/'results.json').read_text());store=ParquetStore(root/'data')
    market=pd.read_parquet(args.study/'market-history.parquet');prices=market.pivot(index='trade_date',columns='ts_code',values='total_price');marks=market.set_index(['trade_date','ts_code'])
    foundation=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text());pool=pd.read_parquet(foundation['historical_pool']['path'])
    membership=store.read('baseline_v2_industry_membership');calendar=trading_dates(store,'20240730');rows=[]
    for case in report['cases']:
        folder=args.study/case['account'];eq=pd.read_parquet(folder/'equity.parquet');dates=eq.date.astype(str).tolist()
        with sqlite3.connect((folder/'paper.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as con:
            fills=pd.read_sql_query('SELECT * FROM fills',con);ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger',con)
        positions=reconstruct_positions(fills,ledger,dates)
        for event in case['events']:
            if event['status']!='BLOCKED':continue
            day=event['date'];prior=calendar[calendar.index(day)-1];plan=json.loads((folder/f'plan-{day}.json').read_text())
            row={'account':case['account'],'date':day,'reason':event['reason'],'solver_diagnostics':plan.get('solver_diagnostics'),'warm_start':plan.get('warm_start_diagnostics')}
            if event['reason'] in ['CONTINUOUS_SOLVER_INFEASIBLE','HELD_RISK_OR_POOL_MEMBERSHIP_MISSING']:
                score=pd.read_parquet(root/f"state/style-balance/s3-feature-only-inference-20261103-a/fold0-price_only-{case['model']}-scores.parquet").score.xs(pd.Timestamp(prior),level='datetime')
                current=pool.loc[pool.signal_date.astype(str).eq(prior)].set_index('ts_code');ix=score.index;history=[d for d in calendar if d<=prior][-121:]
                raw,_=named_price_risk_exposures(prices.reindex(index=history,columns=ix),current.total_mv.reindex(ix)*10000)
                raw['book_to_price']=1/current.pb.reindex(ix).where(lambda x:x>0);raw['earnings_yield']=1/current.pe_ttm.reindex(ix).replace(0,np.nan);raw['dividend_yield']=current.dv_ttm.reindex(ix)/100
                raw['industry']=ix.map(industry_as_of(membership,prior).l1_code).fillna('UNKNOWN');raw['listing_board']=['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in ix]
                raw=raw.replace([np.inf,-np.inf],np.nan);core=['log_size','momentum60','volatility20','market_beta120','residual_volatility120']
                c,eligible,_=uncertain_controls(raw,core);raw=raw.loc[eligible]
                for f in raw.select_dtypes(include='number').columns:
                    if f not in core:c['unknown_'+f]=raw[f].isna().astype(float)
                c['unknown_industry']=raw.industry.eq('UNKNOWN').astype(float);wb=current.total_mv.reindex(raw.index);wb/=wb.sum()
                if event['reason']=='HELD_RISK_OR_POOL_MEMBERSHIP_MISSING':
                    held=positions[prior]
                    row['held_exited_dated_pool']=sorted(set(held)-set(current.index))
                    row['held_in_pool_without_inference']=sorted((set(held)&set(current.index))-set(score.index))
                    row['held_with_inference_missing_core_risk']=sorted((set(held)&set(score.index))-set(raw.index))
                    susp=pd.read_parquet(root/f'state/style-balance/s3-cost-evidence-fold0-20261103-a/suspend_d-{prior}.parquet')
                    row['official_signal_day_suspended_held']=sorted(set(held)&set(susp.loc[susp.suspend_type.eq('S'),'ts_code'])) if not susp.empty else []
                    rows.append(row);continue
                limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
                lo,hi=quality_upper_only_bounds(c,wb,limits,{f:{'upper':0.,'lower':0.} for f in c.columns})
                held=positions[prior];values=pd.Series(held)*marks.loc[prior].close.reindex(held)
                w0=(values/values.sum()).reindex(raw.index,fill_value=0.)
                w=cp.Variable(len(raw));active=c.to_numpy().T@(w-wb.to_numpy())
                objective=cp.norm1(w-w0.to_numpy())
                problem=cp.Problem(cp.Minimize(objective),[w>=0,w<=.15,cp.sum(w)==1,active>=lo.to_numpy(),active<=hi.to_numpy()])
                problem.solve(solver=cp.CLARABEL,max_iter=300,tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
                row.update(minimum_turnover_status=problem.status,minimum_required_l1=float(objective.value) if problem.status=='optimal' else None,
                    turnover_budget=.20,contract_changed=False)
            rows.append(row)
    args.output.write_text(json.dumps({'status':'DIAGNOSIS_ONLY_NO_EXECUTION_POLICY_CHANGE','cases':rows,'orders_enabled':False},indent=2,allow_nan=False))
    print(json.dumps(rows))


if __name__=='__main__':main()
