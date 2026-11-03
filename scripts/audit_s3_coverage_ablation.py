"""Joint six-candidate uncertainty and exact score comparison, no selection."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from moneymore.research.walkforward_protocol import block_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank-study',type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_JOINT_AUDIT')
    root=Path(__file__).resolve().parents[1]/'state/style-balance'
    studies={'original':root/'s3-walkforward-20261103-a','without_sparse_turnover':root/'s3-coverage-ablation-20261103-a'}
    if args.rank_study is not None:studies['qlib_rank_target']=args.rank_study
    family=10 if args.rank_study is not None else 6
    summary=[];paired=[];comparisons=[];inputs={}
    original=pd.read_parquet(studies['original']/'daily-ic.parquet').pivot(index='date',columns=['variant','model'],values='rank_ic')
    for label,folder in studies.items():
        daily=pd.read_parquet(folder/'daily-ic.parquet')
        for (variant,model),group in daily.groupby(['variant','model']):
            if model=='shuffled_ridge':continue
            interval=block_interval(group.rank_ic,comparisons=family)
            summary.append({'study':label,'variant':variant,'model':model,'interval':interval,'screen_pass':interval['mean']>=.01 and interval['lower']>0})
        inputs[label]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    reduced=pd.read_parquet(studies['without_sparse_turnover']/'daily-ic.parquet').pivot(index='date',columns='model',values='rank_ic')
    for model in ['ridge','lightgbm']:
        delta=reduced[model]-original[('price_plus_market_factors',model)]
        paired.append({'model':model,'comparison':'without_sparse_field_minus_original_augmented','interval':block_interval(delta,comparisons=family)})
        for fold in range(4):
            key=f'fold{fold}-price_plus_market_factors-{model}'
            a=pd.read_parquet(studies['original']/(key+'-predictions.parquet')).score
            b=pd.read_parquet(studies['without_sparse_turnover']/(key+'-predictions.parquet')).score
            if not a.index.equals(b.index):raise ValueError('COVERAGE_ABLATION_CHANGED_ROWS')
            row={'fold':fold,'model':model,'maximum_score_change':float(abs(a-b).max())}
            if model=='lightgbm':
                estimator=pickle.loads((studies['original']/(key+'-model.pkl')).read_bytes())
                importance=dict(zip(estimator.feature_name(),map(int,estimator.feature_importance())))
                row['original_sparse_field_split_count']=sum(importance.get(f,0) for f in ['turnover_mean_20','missing_turnover_mean_20'])
            comparisons.append(row)
    rank_effect=[]
    if args.rank_study is not None:
        rank=pd.read_parquet(args.rank_study/'daily-ic.parquet').pivot(index='date',columns=['variant','model'],values='rank_ic')
        for variant in ['price_only','price_plus_market_factors']:
            for model in ['ridge','lightgbm']:
                baseline=original[(variant,model)] if variant=='price_only' else reduced[model]
                delta=rank[(variant,model)]-baseline
                if delta.isna().any():raise ValueError('RANK_LABEL_PAIRED_DAYS_MISMATCH')
                rank_effect.append({'variant':variant,'model':model,'rank_target_minus_same_features_raw_target':block_interval(delta,comparisons=family)})
    result={'status':'EXPLORATION_COMPLETE_NOT_RELEASED','candidate_count':family,'candidate_intervals':summary,'paired_coverage_effect':paired,'paired_rank_effect':rank_effect,
        'score_comparisons':comparisons,'artifact_hashes':inputs,'orders_enabled':False,'publication_allowed':False,
        'limitations':['six_candidates_include_data_quality_ablation_proposed_after_first_run','no_claim_of_original_pre_registration',
            'single_shuffled_seed_per_fold_is_not_full_permutation_test','no_fees_or_execution_return_validation']}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'summary':summary,'coverage_effect':paired,'score_comparisons':comparisons}))


if __name__=='__main__':main()
