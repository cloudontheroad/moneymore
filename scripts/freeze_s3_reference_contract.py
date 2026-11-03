"""Freeze the planned circulating-cap research reference; report, never hide gaps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from moneymore.research.held_evidence import read_held_basic


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_REFERENCE_CONTRACT')
    root=Path(__file__).resolve().parents[1];audit=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text());path=Path(audit['historical_pool']['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['historical_pool']['sha256']:raise ValueError('DATED_POOL_CHANGED')
    pool=pd.read_parquet(path);pool['signal_date']=pool.signal_date.astype(str)
    if (pool.trade_date.astype(str)>=pool.signal_date).any():raise ValueError('POOL_SOURCE_NOT_PRIOR')
    episodes=json.loads((root/'state/style-balance/s3-cost-preflight-20261103-a/results.json').read_text())['episodes'];rows=[];weights=[];bindings=[]
    for fold in range(4):
        folder=root/f'state/style-balance/s3-attribution-basic-fold{fold}-20261103-a';data=read_held_basic(folder)
        bindings.append({'path':str((folder/'manifest.json').resolve()),'sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()})
        days=sorted(data.trade_date.astype(str).unique())
        if len(days)!=21:raise ValueError('TWENTY_ONE_REFERENCE_DATES_REQUIRED')
        for signal,execution in zip(days,episodes[fold]['sessions']):
            current=pool.loc[pool.signal_date.eq(signal)].set_index('ts_code');attrs=data.loc[data.trade_date.astype(str).eq(signal)].set_index('ts_code').reindex(current.index)
            cap=pd.to_numeric(attrs.get('circ_mv',pd.Series(np.nan,index=current.index)),errors='coerce');valid=cap.gt(0)&np.isfinite(cap)
            if len(current)!=1000 or signal>=execution:raise ValueError('REFERENCE_UNIVERSE_OR_DATE_INVALID')
            complete=bool(valid.all());rows.append({'fold':fold,'signal_date':signal,'execution_date':execution,'complete':complete,'observed':int(valid.sum()),'required':1000,'missing_symbols':cap.index[~valid].tolist()})
            if complete:
                w=cap/cap.sum()
                if abs(w.sum()-1)>1e-10:raise ValueError('REFERENCE_WEIGHTS_NOT_NORMALIZED')
                weights.extend({'fold':fold,'signal_date':signal,'execution_date':execution,'symbol':s,'weight':float(v)} for s,v in w.items())
    args.output.mkdir(parents=True);pd.DataFrame(weights,columns=['fold','signal_date','execution_date','symbol','weight']).to_parquet(args.output/'circulating-cap-weights.parquet',index=False)
    protocol=root/'configs/style_s3_cost_replay_v1.yaml';risk=yaml.safe_load(protocol.read_text())['risk_contract']
    result={'status':'RESEARCH_REFERENCE_FROZEN' if all(r['complete'] for r in rows) else 'LIMITED_REFERENCE_COVERAGE_FROZEN',
        'method':'signal_close_circulating_cap_weights_within_frozen_dated_Top1000_known_before_next_execution',
        'complete_dates':sum(r['complete'] for r in rows),'required_dates':84,'dates':rows,'bindings':bindings,
        'pool_sha256':audit['historical_pool']['sha256'],'risk_contract':risk,'risk_protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'production_authorized':False,'orders_enabled':False,'unobserved_caps_zero_filled_or_renormalized':False,
        'limits':['not_an_official_exchange_index','latest_historical_vendor_snapshots_not_original_vintage',
            'S6 retains its original lagged_total_cap_risk_reference; this evaluation reference does not rewrite old trades',
            'existing risk budgets frozen_not_optimized_on_this_batch_returns_not_claimed_production_optimal']}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'status':result['status'],'complete_dates':result['complete_dates'],'required_dates':84}))


if __name__=='__main__':main()
