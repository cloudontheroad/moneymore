"""Six bounded official reads for the diagnosed held-risk gaps; no production writes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
import pandas as pd
from moneymore.data.tushare_provider import TushareProvider


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_HELD_EVIDENCE')
    root=Path(__file__).resolve().parents[1];load_dotenv(root/'.env')
    client=TushareProvider()._client;args.output.mkdir(parents=True);rows=[]
    cases=[('603171.SH','20240722'),('601298.SH','20240715'),('688516.SH','20240715')]
    for symbol,day in cases:
      for endpoint,params in [
          ('daily_basic',{'ts_code':symbol,'trade_date':day,'fields':'ts_code,trade_date,total_mv,pb,pe_ttm,dv_ttm'}),
          ('suspend_d',{'ts_code':symbol,'start_date':'20240102','end_date':day})]:
        row={'endpoint':endpoint,'params':params}
        try:
            data=getattr(client,endpoint)(**params)
            if data is None:raise ValueError('NULL_OFFICIAL_RESPONSE')
            if not data.empty and not data.ts_code.eq(symbol).all():raise ValueError('WRONG_SYMBOL')
            if endpoint=='daily_basic':
                if len(data)!=1 or not data.trade_date.astype(str).eq(day).all():raise ValueError('MISSING_OR_WRONG_DATED_BASIC')
                if not pd.to_numeric(data.total_mv,errors='coerce').gt(0).all():raise ValueError('INVALID_MARKET_CAP')
            elif not data.empty and not data.trade_date.astype(str).between('20240102',day).all():raise ValueError('WRONG_SUSPENSION_RANGE')
            file=args.output/f'{endpoint}-{symbol}-{day}.parquet';data.to_parquet(file,index=False)
            row.update(file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),rows=len(data))
        except Exception as error:
            row['error_type']=type(error).__name__
        rows.append(row);(args.output/'progress.json').write_text(json.dumps(rows,indent=2));time.sleep(1.1)
    result={'status':'READY' if all('sha256' in r for r in rows) else 'BLOCKED',
            'requests':rows,'request_budget':6,'production_updated':False,'orders_enabled':False,
            'source':'tushare_official_api_snapshot_not_historical_vintage_certification'}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
