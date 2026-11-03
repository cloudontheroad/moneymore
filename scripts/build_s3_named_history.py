"""As-of named risk history from the existing frozen dated pool and local data.

No predictions, refits, price fills, production writes or replacement of S6 risk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.baseline_v2_financials import common_period_features,required_period
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.style_risk_inputs import signed_inverse
from moneymore.research.named_factor_history import named_return,named_covariance


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_NAMED_RISK_HISTORY')
    root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data')
    foundation=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text());source=Path(foundation['historical_pool']['path'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=foundation['historical_pool']['sha256']:raise ValueError('DATED_POOL_CHANGED')
    pool=pd.read_parquet(source);pool['signal_date']=pool.signal_date.astype(str)
    if (pool.trade_date.astype(str)>=pool.signal_date).any():raise ValueError('NONCAUSAL_REFERENCE')
    dates=sorted(pool.signal_date.unique());calendar=trading_dates(store,'20251231');start=calendar[calendar.index(dates[0])-121]
    symbols=sorted(pool.ts_code.unique());filters=[('ts_code','in',symbols),('trade_date','>=',start),('trade_date','<=',calendar[calendar.index(dates[-1])+1])]
    bars=store.read('daily',columns=['ts_code','trade_date','close','amount'],filters=filters).merge(store.read('adj_factor',columns=['ts_code','trade_date','adj_factor'],filters=filters),on=['ts_code','trade_date'],validate='one_to_one')
    bars['total_price']=bars.close*bars.adj_factor
    prices=bars.pivot(index='trade_date',columns='ts_code',values='total_price').reindex([d for d in calendar if start<=d<=bars.trade_date.max()])
    amount=bars.pivot(index='trade_date',columns='ts_code',values='amount').reindex(prices.index)
    basic=store.read('daily_basic',columns=['ts_code','trade_date','turnover_rate'],filters=filters)
    turnover=basic.pivot(index='trade_date',columns='ts_code',values='turnover_rate').reindex(prices.index)
    finances=research_financial_frame(store);membership=store.read('baseline_v2_industry_membership')
    args.output.mkdir(parents=True);(args.output/'exposures').mkdir();returns=[];residuals=[];audits=[]
    inputs={}
    for name,frame in [('prices',bars),('turnover',basic),('financials',finances),('industry',membership)]:
        file=args.output/f'{name}-input.parquet';frame.to_parquet(file,index=False)
        inputs[name]={'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'rows':len(frame)}
    required=['log_size','momentum60','volatility20','market_beta120','residual_volatility120']
    for offset,day in enumerate(dates):
        current=pool.loc[pool.signal_date.eq(day)].set_index('ts_code');ix=current.index;history=prices.loc[:day,ix].tail(121)
        raw,_=named_price_risk_exposures(history,current.total_mv*10000)
        raw['book_to_price']=signed_inverse(current.pb);raw['earnings_yield']=signed_inverse(current.pe_ttm);raw['dividend_yield']=current.dv_ttm/100
        raw['turnover_rate']=turnover.loc[:day].tail(20).reindex(columns=ix).mean().where(turnover.loc[:day].tail(20).reindex(columns=ix).notna().sum().eq(20))/100
        liquid=amount.loc[:day].tail(20).reindex(columns=ix);raw['log_amount20']=np.log(liquid.mean().where(liquid.notna().sum().eq(20)).where(lambda x:x>0)*1000)
        financial=common_period_features(finances,day).set_index('ts_code')
        for field,name in [('roe','quality_roe'),('q_sales_yoy','growth_sales_yoy')]:
            raw[name]=financial[field].where(financial[field+'_end_date'].astype(str).eq(required_period(day))).reindex(ix)/100
        raw['industry']=ix.map(industry_as_of(membership,day).l1_code).fillna('UNKNOWN')
        raw['listing_board']=['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in ix]
        raw=raw.replace([np.inf,-np.inf],np.nan);unobserved=[c for c in raw.select_dtypes(include='number') if raw[c].notna().sum()==0 and c not in required]
        controls,eligible,missing=uncertain_controls(raw.drop(columns=unobserved),required)
        # An industry intercept replaces a redundant all-market intercept.
        controls.to_parquet(args.output/'exposures'/f'{day}.parquet')
        aliases={};kept=[]
        for column in controls:
            identical=next((other for other in kept if np.array_equal(controls[column].to_numpy(),controls[other].to_numpy())),None)
            if identical is not None:aliases[column]=identical
            else:kept.append(column)
        following=calendar[calendar.index(day)+1];economic=prices.loc[following]/prices.loc[day]-1
        row={'signal_date':day,'realized_date':following,'eligible':int(eligible.sum()),'pool':len(ix),'unobserved_optional':unobserved,
             'excluded_by_board':raw.loc[~eligible,'listing_board'].value_counts().to_dict(),'input_missing_counts':missing['missing_counts'],
             'identical_column_aliases_not_independent_effects':aliases}
        try:
            coef,residual,fit=named_return(controls[kept],economic,day,following);coef.name=following;returns.append(coef)
            residual.name=following;residuals.append(residual);row.update(status='MEASURED',fit=fit)
        except ValueError as error:row.update(status='UNAVAILABLE',reason=str(error))
        audits.append(row)
        if offset%25==0:print(json.dumps({'measured_sessions':offset+1,'total':len(dates)}),flush=True)
    factor=pd.DataFrame(returns);factor.index.name='realized_date';factor.to_parquet(args.output/'named-factor-returns.parquet')
    residual_frame=pd.DataFrame(residuals);residual_frame.to_parquet(args.output/'specific-return-residuals.parquet')
    covariance_audits=[]
    for as_of in dates:
        if as_of not in factor.index:continue
        try:
            covariance,meta=named_covariance(factor,as_of);covariance.to_parquet(args.output/f'covariance-{as_of}.parquet')
            past=residual_frame.loc[residual_frame.index.astype(str)<=as_of].tail(120)
            specific=past.var(ddof=1).where(past.notna().sum().ge(120));specific.to_frame('specific_variance').to_parquet(args.output/f'specific-{as_of}.parquet')
            covariance_audits.append(dict(status='MEASURED',specific_assets_observed=int(specific.notna().sum()),**meta))
        except ValueError as error:covariance_audits.append({'as_of':as_of,'status':'UNAVAILABLE','reason':str(error)})
    result={'status':'NAMED_HISTORY_MEASURED_NOT_RELEASE_APPROVAL','sessions':len(dates),'measured_factor_sessions':len(factor),
        'pool_sha256':foundation['historical_pool']['sha256'],'inputs':inputs,'daily':audits,'covariance_audits':covariance_audits,
        'orders_enabled':False,'production_risk_replaced':False,
        'limitations':['latest_vendor_snapshots_not_original_vintage','missing_required_risk_excluded_and_reported_not_filled',
            'incomplete_factor_histories_not_filled_with_zero','OLS_style_measurement_not_causal_alpha_proof','not_full_Barra_or_all_market_PIT_certification']}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['status','sessions','measured_factor_sessions']}))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
