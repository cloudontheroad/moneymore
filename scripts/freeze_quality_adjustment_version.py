"""Reconcile disputed symbols over the whole risk window; immutable research data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.style_price_features import named_price_risk_exposures


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_RISK_INPUT_VERSION')
    root=Path(__file__).resolve().parents[1];diagnosis=root/'state/style-balance/quality-adjustment-diagnosis-20261103-a'
    comparison=pd.read_parquet(diagnosis/'version-comparison.parquet')
    symbols=sorted(comparison.loc[~comparison.same_reported_precision,'symbol'].unique())
    if len(symbols)!=11:raise ValueError('FROZEN_REQUEST_SCOPE_CHANGED')
    source=root/'state/style-balance/real-archive-group-decomposition-20261102-b/source-price-history.parquet'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest();old=pd.read_parquet(source)
    start,end=old.trade_date.min(),old.trade_date.max();args.output.mkdir(parents=True)
    load_dotenv(root/'.env');client=TushareProvider()._client;requests=[];parts=[]
    for symbol in symbols:
        try:
            data=client.adj_factor(ts_code=symbol,start_date=start,end_date=end)
            if data is None or data.empty:raise ValueError('EMPTY_OFFICIAL_FACTOR_HISTORY')
            if not data.ts_code.eq(symbol).all() or data.duplicated(['ts_code','trade_date']).any():raise ValueError('INVALID_OFFICIAL_FACTOR_KEYS')
            file=args.output/(symbol.replace('.','_')+'-official-adj.parquet');data.to_parquet(file,index=False);parts.append(data)
            requests.append({'symbol':symbol,'rows':len(data),'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        except Exception as error:
            requests.append({'symbol':symbol,'error_type':type(error).__name__})
        (args.output/'requests.json').write_text(json.dumps(requests,indent=2));time.sleep(1.1)
    if any('error_type' in r for r in requests):
        (args.output/'results.json').write_text(json.dumps({'status':'BLOCKED_OFFICIAL_HISTORY','requests':requests}))
        print('BLOCKED_OFFICIAL_HISTORY');return
    official=pd.concat(parts).set_index(['trade_date','ts_code']).adj_factor
    fixed=old.copy();mask=fixed.ts_code.isin(symbols)
    values=official.reindex(pd.MultiIndex.from_frame(fixed.loc[mask,['trade_date','ts_code']]))
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('OFFICIAL_HISTORY_MISSING_NOT_FILLED')
    fixed.loc[mask,'adj_factor']=values.to_numpy();fixed['total_price']=fixed.close*fixed.adj_factor
    pd.testing.assert_frame_equal(old[['ts_code','trade_date','open','close']],fixed[['ts_code','trade_date','open','close']])
    fixed.to_parquet(args.output/'source-price-history.parquet',index=False)
    prices_old=old.pivot(index='trade_date',columns='ts_code',values='total_price')
    prices_new=fixed.pivot(index='trade_date',columns='ts_code',values='total_price')
    calendar=trading_dates(ParquetStore(root/'data'),end);daily=[]
    for path in sorted((root/'state/style-balance/preperiod-drift-202609-b').glob('observations-*.parquet')):
        day=path.stem.split('-')[-1];raw=pd.read_parquet(path);days=[d for d in calendar if d<=day][-121:];caps=np.exp(raw.log_size)
        prior,_=named_price_risk_exposures(prices_old.reindex(index=days,columns=raw.index),caps)
        if float((prior-raw[prior.columns]).abs().max().max())>1e-10:raise ValueError('OLD_RISK_BASIS_NOT_REPRODUCED')
        rebuilt,_=named_price_risk_exposures(prices_new.reindex(index=days,columns=raw.index),caps)
        if not np.isfinite(rebuilt).all().all():raise ValueError('REBUILT_CORE_RISK_UNOBSERVABLE')
        result=raw.copy();result[rebuilt.columns]=rebuilt
        result.to_parquet(args.output/path.name)
        daily.append({'date':day,'assets':len(raw),'maximum_changes':(rebuilt-prior).abs().max().to_dict(),
            'raw_observation_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise ValueError('SOURCE_CHANGED')
    report={'status':'FROZEN_RESEARCH_INPUT_V2_NOT_PRODUCTION','source_sha256':source_hash,'reconciled_symbols':symbols,
        'official_history_start':start,'official_history_end':end,'requests':requests,'daily':daily,
        'repaired_rows':int((old.adj_factor-fixed.adj_factor).abs().gt(1e-12).sum()),
        'source_price_history_sha256':hashlib.sha256((args.output/'source-price-history.parquet').read_bytes()).hexdigest(),
        'limitations':['latest_official_reconstruction_not_historical_vintage','11_disputed_symbols_not_full_market_certification',
            'remaining_precision_level_differences_preserved','old_signals_and_old_error_envelope_not_retrained_or_revalidated'],
        'orders_enabled':False,'production_data_updated':False}
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'status':report['status'],'symbols':len(symbols),'risk_days':len(daily),'repaired_rows':report['repaired_rows']}))


if __name__=='__main__':main()
