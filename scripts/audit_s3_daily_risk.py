"""Read-only close exposure audit; never confuses decision limits with fills."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.replay_attribution import reconstruct_positions
from moneymore.research.held_evidence import read_held_basic
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.quality_contract import quality_upper_only_bounds


def main():
    p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_DAILY_RISK_AUDIT')
    root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data');calendar=trading_dates(store,'20251231')
    progress=json.loads((args.matrix/'cost-matrix-complete.json').read_text());foundation=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text())
    pool=pd.read_parquet(foundation['historical_pool']['path']);pool['date']=pool.signal_date.astype(str);membership=store.read('baseline_v2_industry_membership')
    rows=[];args.output.mkdir(parents=True)
    for item in progress['studies']:
        study=Path(item['path']);source=json.loads((study/'results.json').read_text())
        binding=json.loads((study/'episode-binding.json').read_text()) if (study/'episode-binding.json').exists() else {'fold':0,'evidence':str(root/'state/style-balance/s3-cost-evidence-fold0-20261103-a'),'audit':str(root/'state/style-balance/s3-cost-evidence-audit-fold0-20261103-a')}
        fold=binding['fold'];basic=read_held_basic(root/f'state/style-balance/s3-attribution-basic-fold{fold}-20261103-a')
        bars=pd.read_parquet(study/'market-history.parquet');prices=bars.pivot(index='trade_date',columns='ts_code',values='total_price');quotes=bars.set_index(['trade_date','ts_code'])
        actions=pd.read_parquet(Path(binding['audit'])/'official-actions.parquet').set_index('action_key');contexts={}
        inference=root/'state/style-balance/s3-feature-only-inference-20261103-a'
        scores={m:pd.read_parquet(inference/f'fold{fold}-price_only-{m}-scores.parquet') for m in ['ridge','lightgbm']}
        for case in source['cases']:
            folder=study/case['account'];equity=pd.read_parquet(folder/'equity.parquet');dates=equity.date.astype(str).tolist();database=folder/'paper.sqlite3';digest=hashlib.sha256(database.read_bytes()).hexdigest()
            with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as con:
                fills=pd.read_sql_query('select * from fills',con);ledger=pd.read_sql_query('select * from corporate_action_ledger',con);rights=pd.read_sql_query('select * from corporate_entitlements',con)
            positions=reconstruct_positions(fills,ledger,dates);last=bars.loc[bars.trade_date.lt(dates[0])].sort_values('trade_date').groupby('ts_code').tail(1).set_index('ts_code').close.to_dict()
            for day in dates:
                marks=quotes.loc[day].close.dropna().to_dict();susp=pd.read_parquet(Path(binding['evidence'])/f'suspend_d-{day}.parquet')
                halted=set(susp.loc[susp.suspend_type.eq('S')&(susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code']) if len(susp) else set()
                for symbol in halted-set(marks):
                    if symbol not in last:continue
                    value=last[symbol]
                    for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():value=(value-action.cash_per_share)/(1+action.stock_ratio)
                    marks[symbol]=value
                held=dict(positions[day]);settled=set(ledger.loc[ledger.trade_date.astype(str).le(day)&ledger.action_type.eq('STOCK_DIVIDEND'),'action_key'])
                for right in rights.itertuples():
                    if right.stock_ratio>0 and right.action_key not in settled and actions.loc[right.action_key,'ex_date']<=pd.Timestamp(day):held[right.symbol]=held.get(right.symbol,0)+int(right.entitled_quantity*right.stock_ratio)
                row={'study':str(study),'account':case['account'],'date':day,'terminal':day==dates[-1]};snapshot=equity.loc[equity.date.eq(day)].iloc[0]
                values=pd.Series({s:q*marks.get(s,np.nan) for s,q in held.items()},dtype=float)
                if not np.isfinite(values).all() or abs(values.sum()-snapshot.market_value-snapshot.unsettled_stock_value)>.01:raise ValueError('ECONOMIC_STOCK_VALUE_NOT_REPRODUCED')
                row['actual_stock_gross']=float(values.sum()/snapshot.equity)
                if not held:row['status']='CASH'
                else:
                    current=pool.loc[pool.date.eq(day)].set_index('ts_code');scored=scores[case['model']].xs(pd.Timestamp(day),level='datetime').index;ix=pd.Index(sorted(set(scored)|set(held)));key=(day,tuple(ix))
                    if key not in contexts:
                        attrs=current.reindex(ix).copy();exits=ix.difference(current.index);extra=basic.loc[basic.trade_date.astype(str).eq(day)].set_index('ts_code')
                        for f in ['total_mv','pb','pe_ttm','dv_ttm']:attrs.loc[exits,f]=extra[f].reindex(exits)
                        history=prices.reindex(index=[d for d in calendar if d<=day][-121:],columns=ix)
                        raw,_=named_price_risk_exposures(history,attrs.total_mv*10000);raw['book_to_price']=1/attrs.pb.where(lambda x:x>0);raw['earnings_yield']=1/attrs.pe_ttm.replace(0,np.nan);raw['dividend_yield']=attrs.dv_ttm/100
                        raw['industry']=ix.map(industry_as_of(membership,day).l1_code).fillna('UNKNOWN');raw['listing_board']=['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in ix]
                        c,eligible,_=uncertain_controls(raw.replace([np.inf,-np.inf],np.nan),['log_size','momentum60','volatility20','market_beta120','residual_volatility120'])
                        for f in ['book_to_price','earnings_yield','dividend_yield']:c['unknown_'+f]=raw.loc[eligible,f].isna().astype(float)
                        c['unknown_industry']=raw.loc[eligible,'industry'].eq('UNKNOWN').astype(float);wb=current.total_mv.reindex(c.index,fill_value=0.);wb/=wb.sum()
                        limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c})
                        contexts[key]=(c,wb,limits)
                    c,wb,limits=contexts[key];missing=sorted(set(held)-set(c.index))
                    if missing:row.update(status='HELD_REQUIRED_RISK_UNOBSERVED',symbols=missing)
                    else:
                        active=(values.reindex(c.index,fill_value=0.)/values.sum()-wb)@c;zero={f:{'lower':0.,'upper':0.} for f in c}
                        lo,hi=quality_upper_only_bounds(c,wb,limits,zero);excess=float(max((active-hi).max(),(lo-active).max()))
                        row.update(status='CLOSE_DRIFT_OUTSIDE_DECISION_BAND' if excess>1e-5 else 'MEASURED_WITHIN_DECISION_BAND',
                            maximum_excess=excess,active_controls=active.to_dict(),account_scaled_active_controls=(active*row['actual_stock_gross']).to_dict())
                rows.append(row);last.update(marks)
            if hashlib.sha256(database.read_bytes()).hexdigest()!=digest:raise ValueError('SOURCE_LEDGER_CHANGED')
        print(json.dumps({'completed_risk_study':str(study)}),flush=True)
    result={'status':'DAILY_RISK_MEASURED_NOT_RELEASE_APPROVAL','rows':rows,'counts':dict(Counter(r['status'] for r in rows)),
        'normal_counts':dict(Counter(r['status'] for r in rows if not r['terminal'])),'orders_enabled':False,'trades_or_equity_modified':False,
        'limits_scope':'frozen decision/reference-execution bands; close drift is monitored, not retrospectively traded or relabelled as a fill violation',
        'terminal_remainders':'liquidity-blocked terminal tails are separate; stock-sleeve and total-account-scaled exposures both reported'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['counts']))


if __name__=='__main__':main()
