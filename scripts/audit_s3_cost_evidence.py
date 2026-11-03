"""Audit fixed-episode gates and actions against full inference coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.action_versions import latest_known_action_versions


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_EVIDENCE_AUDIT')
    manifest=json.loads((args.evidence/'manifest.json').read_text())
    if manifest['status']!='OFFICIAL_SNAPSHOTS_READY_REQUIRES_POOL_AND_EVENT_AUDIT':raise ValueError('SNAPSHOTS_NOT_READY')
    for item in manifest['requests']:
        if hashlib.sha256((args.evidence/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('SNAPSHOT_CHANGED')
    root=Path(__file__).resolve().parents[1];inference=root/'state/style-balance/s3-feature-only-inference-20261103-a'
    validation=json.loads((inference/'independent-audit.json').read_text())
    for name,digest in validation['artifact_hashes'].items():
        if hashlib.sha256((inference/name).read_bytes()).hexdigest()!=digest:raise ValueError('INFERENCE_CHANGED')
    fold=manifest['fold'];episode=manifest['episode'];sessions=episode['sessions'];store=ParquetStore(root/'data')
    score=pd.read_parquet(inference/f'fold{fold}-price_only-ridge-scores.parquet')
    prior_days=trading_dates(store,sessions[-1]);first=prior_days.index(sessions[0])-1;signal_days=prior_days[first:first+21]
    selected=score.loc[score.index.get_level_values('datetime').isin(pd.to_datetime(signal_days))]
    names=set(selected.index.get_level_values('instrument'))
    limit_checks=[]
    for offset in [0,5,10,15,20]:
        day=sessions[offset];limits=pd.read_parquet(args.evidence/f'stk_limit-{day}.parquet').set_index('ts_code').reindex(sorted(names))
        valid=np.isfinite(limits[['up_limit','down_limit']]).all(axis=1)&limits.up_limit.gt(0)&limits.down_limit.gt(0)&limits.up_limit.ge(limits.down_limit)
        limit_checks.append({'date':day,'required_union_symbols':len(names),'missing_or_invalid':int((~valid).sum()),'examples':limits.index[~valid].tolist()[:10]})
    frames=[pd.read_parquet(args.evidence/f'dividend-{day}.parquet') for day in sessions]
    nonempty=[f for f in frames if not f.empty]
    raw=pd.concat(nonempty,ignore_index=True) if nonempty else frames[0]
    relevant=raw.loc[raw.ts_code.isin(names)] if not raw.empty else raw
    selected,version_audit=latest_known_action_versions(relevant)
    actions=official_replay_actions(selected)
    market=store.read('daily',columns=['ts_code','trade_date','open','close'],filters=[('ts_code','in',sorted(names)),('trade_date','>=',sessions[0]),('trade_date','<=',sessions[-1])])
    quotes=market.set_index(['trade_date','ts_code']);missing=[]
    for day in sessions:
        susp=pd.read_parquet(args.evidence/f'suspend_d-{day}.parquet')
        # A timed intraday suspension does not prove a whole-day quote absence.
        full=susp.loc[susp.suspend_type.eq('S') & (susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code'] if not susp.empty else []
        marks=quotes.reindex(pd.MultiIndex.from_product([[day],sorted(names)],names=['trade_date','ts_code']))
        absent=marks.index[~(np.isfinite(marks[['open','close']]).all(axis=1)&marks.open.gt(0)&marks.close.gt(0))]
        for _,symbol in absent:missing.append({'date':day,'symbol':symbol,'official_full_halt':symbol in set(full)})
    args.output.mkdir(parents=True);actions.to_parquet(args.output/'official-actions.parquet',index=False)
    market.to_parquet(args.output/'market-window.parquet',index=False)
    unexplained=sum(not r['official_full_halt'] for r in missing)
    result={'status':'GATES_ACTIONS_AND_QUOTES_AUDITED' if not unexplained and all(r['missing_or_invalid']==0 for r in limit_checks) else 'BLOCKED_LOCAL_EXECUTION_GAPS',
        'fold':fold,'limit_checks':limit_checks,'implemented_pool_actions':len(actions),'action_version_audit':version_audit,'missing_quotes':missing,
        'unexplained_quote_gaps':unexplained,'manifest_sha256':hashlib.sha256((args.evidence/'manifest.json').read_bytes()).hexdigest(),
        'orders_enabled':False,'cost_replay_executed':False,
        'limitations':['full_halt_quotes_require_explicit_valuation_carry_not_buy_permission','dated_vendor_action_snapshots_not_original_vintage',
            'post_exit_action_rights_and_settlement_must_be_accounted_in_replay']}
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','fold','implemented_pool_actions','unexplained_quote_gaps','limit_checks']}))


if __name__=='__main__':main()
