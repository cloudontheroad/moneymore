"""Bind complete inference signals and verify common-score parity independently."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_COVERAGE_AUDIT')
    root=Path(__file__).resolve().parents[1];old=root/'state/style-balance/s3-csrank-20261103-a'
    features=pd.read_parquet(args.study/'available-features.parquet')
    if not features.index.is_unique or not np.isfinite(features).all().all():raise ValueError('INFERENCE_FEATURES_INVALID')
    hashes={};rows=[]
    for fold in range(4):
        model_index=None
        for model in ['ridge','lightgbm']:
            key=f'fold{fold}-price_only-{model}';file=args.study/(key+'-scores.parquet')
            score=pd.read_parquet(file).score;previous=pd.read_parquet(old/(key+'-predictions.parquet')).score
            if not score.index.is_unique or not score.index.isin(features.index).all() or not np.isfinite(score).all():raise ValueError('SCORE_COVERAGE_INVALID')
            if model_index is not None and not model_index.equals(score.index):raise ValueError('MODEL_COVERAGE_MISMATCH')
            model_index=score.index
            if not previous.index.isin(score.index).all():raise ValueError('LOST_PREVIOUS_SCORES')
            delta=float(abs(score.reindex(previous.index)-previous).max())
            if delta>1e-10:raise ValueError('PREDICTION_PARITY_FAILED')
            hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            rows.append({'fold':fold,'model':model,'rows':len(score),'new_rows':len(score)-len(previous),'maximum_common_delta':delta})
    for name in ['available-features.parquet','daily-coverage.parquet','results.json']:hashes[name]=hashlib.sha256((args.study/name).read_bytes()).hexdigest()
    coverage=pd.read_parquet(args.study/'daily-coverage.parquet')
    result={'status':'PASS_FEATURE_ONLY_INFERENCE_COVERAGE','cases':rows,'artifact_hashes':hashes,'days':len(coverage),
        'mean_coverage':float(coverage.coverage.mean()),'minimum_coverage':float(coverage.coverage.min()),
        'models_changed':False,'orders_enabled':False,'not_claimed':['performance_on_new_tail_dates','cost_replay_completed','full_point_in_time_certification']}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','days','mean_coverage','minimum_coverage']}))


if __name__=='__main__':main()
