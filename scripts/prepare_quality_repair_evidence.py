"""Thirteen bounded official reads; immutable evidence, no production updates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from dotenv import load_dotenv
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.tushare_provider import TushareProvider


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_EVIDENCE')
    root=Path(__file__).resolve().parents[1];calendar=trading_dates(ParquetStore(root/'data'),'20260930')
    following=calendar[calendar.index('20260930')+1]
    dates=['20260929','20260930',following]
    requests=[]
    for day in dates:
        previous=calendar[calendar.index(day)-1]
        requests.extend([('dividend',{'ex_date':day},f'dividend-ex-{day}'),
            ('dividend',{'record_date':previous},f'dividend-record-{previous}')])
    for endpoint in ['daily','adj_factor','stk_limit','suspend_d']:
        requests.append((endpoint,{'trade_date':following},f'{endpoint}-{following}'))
    for day in ['20260928','20260929','20260930']:
        requests.append(('adj_factor',{'trade_date':day},f'adj_factor-{day}'))
    if len(requests)!=13:raise ValueError('REQUEST_BUDGET_CHANGED')
    load_dotenv(root/'.env');client=TushareProvider()._client;args.output.mkdir(parents=True);rows=[]
    for endpoint,params,name in requests:
        try:
            frame=getattr(client,endpoint)(**params)
            if frame is None or (endpoint in ['daily','adj_factor','stk_limit'] and frame.empty):raise ValueError('REQUIRED_RESPONSE_EMPTY')
            path=args.output/(name+'.parquet');frame.to_parquet(path,index=False)
            rows.append({'endpoint':endpoint,'params':params,'file':path.name,'rows':len(frame),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        except Exception as error:
            rows.append({'endpoint':endpoint,'params':params,'error_type':type(error).__name__})
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
        time.sleep(1.1)
    result={'status':'READY' if all('sha256' in r for r in rows) else 'BLOCKED','max_requests':13,'next_trade_date':following,
        'requests':rows,'orders_enabled':False,'production_data_updated':False}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':result['status'],'next_trade_date':following,'requests':len(rows),'failed':[r for r in rows if 'error_type' in r]}))


if __name__=='__main__':main()
