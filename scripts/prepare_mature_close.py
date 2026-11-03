"""Five bounded official close reads for this family only; no global data rewrite."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider
from moneymore.strategy_universe import active_strategy_universe
from moneymore.data.quality import validate_daily_bars


def main():
    p=argparse.ArgumentParser();p.add_argument('--date',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_OFFICIAL_CLOSE')
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    if args.date>now.strftime('%Y%m%d') or (args.date==now.strftime('%Y%m%d') and now.hour<18):raise ValueError('OFFICIAL_CLOSE_NOT_COMPLETED')
    root=Path(__file__).resolve().parents[1];load_dotenv(root/'.env');client=TushareProvider()._client
    args.output.mkdir(parents=True);hashes={}
    for endpoint in ['daily','adj_factor','daily_basic','suspend_d','stk_limit']:
        frame=getattr(client,endpoint)(trade_date=args.date)
        if frame is None or (endpoint!='suspend_d' and frame.empty):raise ValueError('REQUIRED_OFFICIAL_CLOSE_EMPTY:'+endpoint)
        if not frame.empty and not frame.trade_date.astype(str).eq(args.date).all():raise ValueError('OFFICIAL_CLOSE_DATE_MISMATCH')
        if endpoint=='daily':validate_daily_bars(frame,args.date)
        if endpoint=='daily_basic':frame['source_trade_date']=args.date
        path=args.output/(endpoint+'.parquet');frame.to_parquet(path,index=False);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        time.sleep(.4)
    names=set(active_strategy_universe(ParquetStore(root/'data'),args.date).symbol)
    bars=pd.read_parquet(args.output/'daily.parquet');adj=pd.read_parquet(args.output/'adj_factor.parquet')
    observed=bars.merge(adj,on=['ts_code','trade_date'],validate='one_to_one')
    eligible=observed.loc[observed.close.gt(0)&observed.adj_factor.gt(0)&observed.ts_code.isin(names)]
    if eligible.ts_code.nunique()/len(names)<.95:raise ValueError('OFFICIAL_CLOSE_CURRENT_POOL_COVERAGE')
    (args.output/'results.json').write_text(json.dumps({'status':'OFFICIAL_CLOSE_READY','date':args.date,'hashes':hashes,
        'covered_current_names':eligible.ts_code.nunique(),'required_current_names':len(names),'global_store_modified':False},indent=2))
    print(json.dumps({'status':'OFFICIAL_CLOSE_READY','date':args.date,'covered_current_names':eligible.ts_code.nunique()}))


if __name__=='__main__':main()
