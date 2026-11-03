"""Feature-only OOS inference with frozen models; no future labels or refitting."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import yaml
from qlib.data.dataset.processor import RobustZScoreNorm
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.qlib_challenger import QlibPanelDataset
from moneymore.research.style_price_features import price_features
from moneymore.research.qlib_feature_preprocessing import transform_features
from moneymore.research.inference_coverage import inference_eligibility


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_INFERENCE')
    root=Path(__file__).resolve().parents[1];study=root/'state/style-balance/s3-csrank-20261103-a'
    audit=json.loads((root/'state/style-balance/s3-rank-joint-audit-20261103-a.json').read_text())
    hashes=audit['artifact_hashes']['qlib_rank_target']
    for name,digest in hashes.items():
        if hashlib.sha256((study/name).read_bytes()).hexdigest()!=digest:raise ValueError('FROZEN_MODEL_OR_PANEL_CHANGED')
    protocol=yaml.safe_load((study/'protocol.yaml').read_text())
    foundation=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text())
    pool_path=Path(foundation['historical_pool']['path'])
    if hashlib.sha256(pool_path.read_bytes()).hexdigest()!=foundation['historical_pool']['sha256']:raise ValueError('HISTORICAL_POOL_CHANGED')
    pool=pd.read_parquet(pool_path)
    if (pool.trade_date.astype(str)>=pool.signal_date.astype(str)).any() or pool.duplicated(['signal_date','ts_code']).any():raise ValueError('NONCAUSAL_POOL_KEYS')
    store=ParquetStore(root/'data');days=[d for d in trading_dates(store,protocol['input_end']) if protocol['input_start']<=d<=protocol['input_end']]
    filters=[('ts_code','in',sorted(pool.ts_code.unique())),('trade_date','>=',days[0]),('trade_date','<=',days[-1])]
    market=store.read('daily',columns=['ts_code','trade_date','open','high','low','close','vol','amount'],filters=filters)
    market=market.merge(store.read('adj_factor',columns=['ts_code','trade_date','adj_factor'],filters=filters),on=['ts_code','trade_date'],how='left',validate='one_to_one')
    parts=[]
    for symbol,g in market.groupby('ts_code',sort=True):
        f=price_features(g.set_index('trade_date').reindex(days),'adjusted')
        f['datetime']=pd.to_datetime(f.index);f['instrument']=symbol
        parts.append(f.reset_index(drop=True).set_index(['datetime','instrument']))
    features=pd.concat(parts).sort_index()
    members=pd.MultiIndex.from_arrays([pd.to_datetime(pool.signal_date),pool.ts_code],names=['datetime','instrument'])
    features=features.reindex(features.index.intersection(members));names=list(features.columns)
    eligible=inference_eligibility(features,names);available=features.loc[eligible]
    args.output.mkdir(parents=True);available.to_parquet(args.output/'available-features.parquet')
    # Retain reasons for omissions. No label column is loaded or computed above.
    missing=features.loc[~eligible].isna().sum(axis=1).rename('missing_feature_count')
    missing.to_frame().to_parquet(args.output/'unavailable-features.parquet')
    results=[];coverage=[];new_source=Path(inspect.getsourcefile(RobustZScoreNorm))
    for fold,(start,end) in enumerate(protocol['folds']):
        dates=available.index.get_level_values('datetime');selected=available.loc[(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))]
        for model_name in ['ridge','lightgbm']:
            key=f'fold{fold}-price_only-{model_name}'
            frozen=pickle.loads((study/(key+'-processor.pkl')).read_bytes())['normalizer']
            if frozen['schema']!=names or hashlib.sha256(new_source.read_bytes()).hexdigest()!=frozen['source_sha256']:raise ValueError('PROCESSOR_SCHEMA_OR_IMPLEMENTATION_CHANGED')
            frame=selected.copy();frame.columns=pd.MultiIndex.from_product([['feature'],names]);frame=transform_features(frozen,frame)
            model=pickle.loads((study/(key+'-model.pkl')).read_bytes())
            dataset=QlibPanelDataset(frame,{'test':(start,end)})
            score=model.predict(dataset,'test') if model_name=='ridge' else pd.Series(model.predict(frame['feature']),index=frame.index)
            if not score.index.equals(selected.index) or not np.isfinite(score).all():raise ValueError('INVALID_FROZEN_INFERENCE')
            previous=pd.read_parquet(study/(key+'-predictions.parquet')).score
            if not previous.index.isin(score.index).all():raise ValueError('LOST_PREVIOUS_PREDICTIONS')
            difference=float(abs(score.reindex(previous.index)-previous).max())
            if difference>1e-10:raise ValueError('FROZEN_PREDICTION_PARITY_FAILED')
            score.rename('score').to_frame().to_parquet(args.output/(key+'-scores.parquet'))
            old_days=previous.index.get_level_values('datetime').unique()
            same_days=score.loc[score.index.get_level_values('datetime').isin(old_days)]
            added_same=len(same_days.index.difference(previous.index))
            results.append({'fold':fold,'model':model_name,'prediction_rows':len(score),'days':score.index.get_level_values('datetime').nunique(),
                'original_evaluation_rows':len(previous),'new_rows_on_original_test_days':added_same,
                'new_rows_on_previously_purged_tail_days':len(score)-len(same_days),'maximum_common_score_error':difference})
            if model_name=='ridge':
                for day,group in score.groupby(level='datetime'):
                    expected=int(pool.signal_date.astype(str).eq(day.strftime('%Y%m%d')).sum())
                    coverage.append({'date':str(day.date()),'fold':fold,'pool':expected,'feature_eligible':len(group),'coverage':len(group)/expected})
    pd.DataFrame(coverage).to_parquet(args.output/'daily-coverage.parquet',index=False)
    result={'status':'FROZEN_FEATURE_ONLY_INFERENCE_READY_NOT_TRADED','cases':results,'orders_enabled':False,'models_retrained':False,
        'future_labels_used_for_eligibility':False,'normalization_refitted':False,'source_artifact_hashes':hashes,
        'pool_sha256':foundation['historical_pool']['sha256'],
        'mean_daily_coverage':float(pd.DataFrame(coverage).coverage.mean()),
        'limitations':['historical_dated_pool_not_original_vintage_proof','historical_price_adjustments_not_full_event_certification',
            'added_tail_dates_are_inference_not_previously_evaluated_rank_ic','no_execution_or_cost_result_yet']}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['cases']))


if __name__=='__main__':main()
