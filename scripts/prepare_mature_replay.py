"""One fixed last-20-heldout-signal execution window; 60 bounded official reads."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv
import pandas as pd
from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider


def finalize(output,result):
    from moneymore.research.replay_actions import official_replay_actions
    for item in result['requests']:
        if hashlib.sha256((output/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('EVIDENCE_CHANGED')
    model=json.loads((Path(result['model'])/'results.json').read_bytes())
    symbols=set(pd.read_parquet(Path(model['inputs'])/'bars-input.parquet',columns=['ts_code']).ts_code)
    raw_actions=pd.concat([pd.read_parquet(output/f'dividend-{d}.parquet') for d in result['sessions']],ignore_index=True)
    # Full-market endpoint also contains B shares with different settlement rules;
    # they are outside this A-share study, not corrupt A-share corporate events.
    actions=official_replay_actions(raw_actions.loc[raw_actions.ts_code.isin(symbols)])
    checked=output/'audit';checked.mkdir(exist_ok=True);actions.to_parquet(checked/'official-actions.parquet',index=False)
    (output/'manifest.json').write_bytes((output/'results.json').read_bytes())
    (checked/'results.json').write_text(json.dumps({'status':'GATES_ACTIONS_AND_QUOTES_AUDITED','requests':60,
        'normalized_actions':len(actions),'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--finalize-existing',action='store_true');args=p.parse_args()
    if args.finalize_existing:
        result=json.loads((args.output/'results.json').read_bytes())
        if result['status']!='READY':raise ValueError('COMPLETE_EXISTING_EVIDENCE_REQUIRED')
        finalize(args.output,result);print('EXISTING_EVIDENCE_FINALIZED_NO_REQUESTS');return
    if args.output.exists():raise FileExistsError('IMMUTABLE_MATURE_REPLAY_EVIDENCE')
    args.output.mkdir(parents=True);root=Path(__file__).resolve().parents[1]
    model=json.loads((args.model/'results.json').read_bytes())
    scores=pd.read_parquet(args.model/'test-predictions.parquet');signals=sorted(scores.index.get_level_values('datetime').strftime('%Y%m%d').unique())[-20:]
    calendar=[d for d in trading_dates(ParquetStore(root/'data'),model['data_cutoff']) if d<=model['data_cutoff']]
    sessions=[calendar[calendar.index(day)+1] for day in signals]
    load_dotenv(root/'.env');client=TushareProvider()._client;requests=[]
    for day in sessions:
        for endpoint,params,field in [('stk_limit',{'trade_date':day},'trade_date'),('suspend_d',{'trade_date':day},'trade_date'),('dividend',{'record_date':day},'record_date')]:
            try:
                data=getattr(client,endpoint)(**params)
                if data is None or (endpoint=='stk_limit' and data.empty):raise ValueError('OFFICIAL_RESPONSE_EMPTY')
                if not data.empty and not data[field].astype(str).eq(day).all():raise ValueError('OFFICIAL_DATE_MISMATCH')
                target=args.output/f'{endpoint}-{day}.parquet';data.to_parquet(target,index=False)
                requests.append({'file':target.name,'rows':len(data),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
            except Exception as error:
                requests.append({'endpoint':endpoint,'date':day,'error_type':type(error).__name__})
            (args.output/'progress.json').write_text(json.dumps({'completed':len(requests),'budget':60,'errors':sum('error_type'in r for r in requests)}))
            time.sleep(1.05)
    result={'status':'READY' if len(requests)==60 and not any('error_type'in r for r in requests) else 'BLOCKED_EVIDENCE',
        'signals':signals,'sessions':sessions,'requests':requests,'model':str(args.model.resolve()),
        'window_rule':'last_20_chronological_heldout_signal_sessions_not_selected_by_return',
        'real_orders_enabled':False,'provider_error_bodies_suppressed':True}
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'requests':len(requests)}))
    if result['status']=='READY':
        finalize(args.output,result)


if __name__=='__main__':main()
