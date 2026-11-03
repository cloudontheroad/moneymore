"""Four fixed signal-day snapshots, independent of eventual holdings."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(4),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--all-signal-days',action='store_true');p.add_argument('--reuse',type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_DATED_BASIC')
    root=Path(__file__).resolve().parents[1];episode=json.loads((root/'state/style-balance/s3-cost-preflight-20261103-a/results.json').read_text())['episodes'][args.fold]
    offsets=range(21) if args.all_signal_days else [0,5,10,15]
    calendar=trading_dates(ParquetStore(root/'data'),episode['execution_end']);days=[calendar[calendar.index(episode['sessions'][i])-1] for i in offsets]
    load_dotenv(root/'.env');client=TushareProvider()._client;args.output.mkdir(parents=True);rows=[]
    cached={}
    if args.reuse:
        from moneymore.research.held_evidence import read_held_basic
        read_held_basic(args.reuse)
        cached={r['params']['trade_date']:r for r in json.loads((args.reuse/'manifest.json').read_text())['requests'] if r['endpoint']=='daily_basic'}
    for day in days:
        params={'trade_date':day,'fields':'ts_code,trade_date,total_mv,circ_mv,pb,pe_ttm,dv_ttm'};row={'endpoint':'daily_basic','params':params}
        if day in cached and cached[day]['params']==params:
            old=cached[day];file=args.output/old['file'];file.write_bytes((args.reuse/old['file']).read_bytes())
            rows.append(dict(old,reused_snapshot=str((args.reuse/old['file']).resolve())));continue
        try:
            data=client.daily_basic(**params)
            if data is None or data.empty or not data.trade_date.astype(str).eq(day).all() or data.ts_code.duplicated().any():raise ValueError('INVALID_DATED_BASIC_RESPONSE')
            good=pd.to_numeric(data.total_mv,errors='coerce').gt(0);row['excluded_invalid_size']=int((~good).sum());data=data.loc[good]
            if data.empty:raise ValueError('NO_VALID_DATED_SIZE')
            file=args.output/f'daily_basic-{day}.parquet';data.to_parquet(file,index=False)
            row.update(file=file.name,rows=len(data),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        except Exception as error:row['error_type']=type(error).__name__
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2));time.sleep(1.1)
    result={'status':'READY' if all('sha256' in r for r in rows) else 'BLOCKED','fold':args.fold,'requests':rows,
        'request_budget':len(days),'actual_requests':sum('reused_snapshot' not in r for r in rows),'production_updated':False,'orders_enabled':False,'selection':'all_market_not_future_holdings',
        'historical_vintage_certified':False}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'fold':args.fold,'requests':len(rows)}))


if __name__=='__main__':main()
