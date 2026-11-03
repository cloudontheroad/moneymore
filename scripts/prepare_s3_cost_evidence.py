"""Fixed episode official evidence: five limit dates, 21 suspensions/record days."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from dotenv import load_dotenv
import pandas as pd
from moneymore.data.tushare_provider import TushareProvider


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(4),required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_EXECUTION_EVIDENCE')
    root=Path(__file__).resolve().parents[1];preflight=root/'state/style-balance/s3-cost-preflight-20261103-a/results.json'
    episode=json.loads(preflight.read_text())['episodes'][args.fold];sessions=episode['sessions']
    requests=[]
    for index,day in enumerate(sessions):
        if index in [0,5,10,15,20]:requests.append(('stk_limit',{'trade_date':day},'trade_date',day))
        requests.extend([('suspend_d',{'trade_date':day},'trade_date',day),('dividend',{'record_date':day},'record_date',day)])
    if len(requests)!=47:raise ValueError('REQUEST_BUDGET_CHANGED')
    args.output.mkdir(parents=True);load_dotenv(root/'.env');client=TushareProvider()._client;rows=[]
    for endpoint,params,date_field,day in requests:
        try:
            data=getattr(client,endpoint)(**params)
            if data is None or (endpoint=='stk_limit' and data.empty):raise ValueError('REQUIRED_EVIDENCE_EMPTY')
            if not data.empty and not pd.to_datetime(data[date_field],errors='coerce').dt.strftime('%Y%m%d').eq(day).all():raise ValueError('WRONG_OFFICIAL_RESPONSE_DATE')
            file=args.output/f'{endpoint}-{day}.parquet';data.to_parquet(file,index=False)
            rows.append({'endpoint':endpoint,'params':params,'file':file.name,'rows':len(data),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        except Exception as error:rows.append({'endpoint':endpoint,'params':params,'error_type':type(error).__name__})
        (args.output/'progress.json').write_text(json.dumps(rows,indent=2));time.sleep(1.1)
        if len(rows)%10==0:print(json.dumps({'fold':args.fold,'completed':len(rows),'max':47}),flush=True)
    result={'status':'OFFICIAL_SNAPSHOTS_READY_REQUIRES_POOL_AND_EVENT_AUDIT' if all('sha256' in r for r in rows) else 'BLOCKED',
        'fold':args.fold,'episode':episode,'max_requests':47,'requests':rows,'orders_enabled':False,'production_data_updated':False,
        'preflight_sha256':hashlib.sha256(preflight.read_bytes()).hexdigest(),
        'action_scope':'record_date_events_for_holdings_created_after_flat_episode_start;settlement_after_end_may_be_receivable'}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'fold':args.fold,'requests':len(rows),'errors':sum('error_type' in r for r in rows)}))


if __name__=='__main__':main()
