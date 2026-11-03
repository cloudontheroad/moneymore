"""Read-only local evidence inventory for fixed no-future-filter cost episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_PREFLIGHT')
    root=Path(__file__).resolve().parents[1];inference=root/'state/style-balance/s3-feature-only-inference-20261103-a'
    summary=json.loads((inference/'results.json').read_text())
    if summary['future_labels_used_for_eligibility'] or summary['models_retrained']:raise ValueError('INFERENCE_NOT_ACCEPTABLE')
    protocol_path=root/'configs/style_s3_cost_replay_v1.yaml';protocol=yaml.safe_load(protocol_path.read_text())
    store=ParquetStore(root/'data');calendar=trading_dates(store,'20250630');episodes=[]
    limits=store.read('stock_limits');available=set(zip(limits.trade_date.astype(str),limits.ts_code.astype(str)))
    for fold in range(4):
        score=pd.read_parquet(inference/f'fold{fold}-price_only-ridge-scores.parquet')
        signal=score.index.get_level_values('datetime').min().strftime('%Y%m%d');index=calendar.index(signal)
        sessions=calendar[index+1:index+22]
        if len(sessions)!=21:raise ValueError('INCOMPLETE_EPISODE_CALENDAR')
        missing_by_day={};row_count=0
        # Audit the full signal-day pool for local limits; no future labels or
        # retrospective final holdings are used to choose the pool.
        for day in sessions:
            prior=calendar[calendar.index(day)-1]
            mask=score.index.get_level_values('datetime')==pd.Timestamp(prior)
            names=score.loc[mask].index.get_level_values('instrument')
            if not len(names):raise ValueError('MISSING_PRIOR_FEATURE_ONLY_SCORES')
            missing_by_day[day]=sum((day,str(s)) not in available for s in names);row_count+=len(names)
        episodes.append({'fold':fold,'first_signal':signal,'execution_start':sessions[0],'execution_end':sessions[-1],
            'sessions':sessions,'signal_pool_limit_keys':row_count,'missing_limit_rows':sum(missing_by_day.values()),
            'missing_limit_dates':[d for d,n in missing_by_day.items() if n],
            'suspension_evidence':'REQUIRES_OFFICIAL_DATED_SNAPSHOT_OR_VERIFIED_CACHE',
            'corporate_actions':'REQUIRES_EX_DATE_AND_RECORD_DATE_COVERAGE_NOT_EMPTY_CURATED_TABLE_ASSUMPTION'})
    args.output.mkdir(parents=True);(args.output/'protocol.yaml').write_bytes(protocol_path.read_bytes())
    result={'status':'PREFLIGHT_COMPLETED_NOT_EXECUTED','episodes':episodes,'models':protocol['models'],'variants':protocol['signal_variants'],
        'total_independent_accounts':len(episodes)*len(protocol['models'])*len(protocol['signal_variants']),
        'inference_report_sha256':hashlib.sha256((inference/'results.json').read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'orders_enabled':False,
        'next_step':'bounded_official_execution_evidence_and_full_pool_style_components_before_paper_replay'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'accounts':result['total_independent_accounts'],
        'episodes':[{k:e[k] for k in ['fold','execution_start','execution_end','missing_limit_rows']} for e in episodes]}))


if __name__=='__main__':main()
