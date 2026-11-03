"""P2: one native Qlib LGBModel, chronological evaluation then latest refit."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml

from moneymore.qlib_challenger import QlibPanelDataset
from moneymore.research.mature_multifactor import temporal_segments
from moneymore.research.qlib_feature_preprocessing import fit_training_processor, transform_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-metric',choices=['mse','rank_ic'],default='mse')
    parser.add_argument('--label-normalization',choices=['zscore','rank'],default='zscore')
    parser.add_argument('--rank-objective',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('IMMUTABLE_MATURE_MODEL')
    args.output.mkdir(parents=True)
    manifest = json.loads((args.inputs / 'results.json').read_bytes())
    if manifest['status'] != 'P1_NATIVE_ALPHA158_READY' or manifest['smoke_only'] or manifest.get('prediction_only'):
        raise ValueError('FULL_P1_REQUIRED')
    for name, digest in manifest['input_hashes'].items():
        if hashlib.sha256((args.inputs / name).read_bytes()).hexdigest() != digest:
            raise ValueError('P1_INPUT_CHANGED')
    protocol = yaml.safe_load((args.inputs / 'protocol.yaml').read_text(encoding='utf-8'))
    if args.selection_metric=='rank_ic':
        protocol['version']='mature_alpha158_lgbm_rankic_v2'
        protocol['lightgbm']['metric']='None'
    protocol['selection_metric']=args.selection_metric
    protocol['label_normalization']=args.label_normalization
    if args.label_normalization=='rank':protocol['version']='mature_alpha158_lgbm_csrank_v2'
    if args.rank_objective:
        protocol['version']='mature_alpha158_lambdarank_v2'
        protocol['lightgbm']={'loss':'mse','objective':'lambdarank','metric':'None','learning_rate':.1,'num_leaves':31,
            'lambda_l1':0.,'lambda_l2':0.,'label_gain':list(range(20)),'lambdarank_truncation_level':53,'num_threads':4}
        protocol['rank_groups']='one_trading_day';protocol['rank_relevance_bins']=20
    (args.output / 'protocol.yaml').write_text(yaml.safe_dump(protocol,allow_unicode=True),encoding='utf-8')
    def status(stage, **extra):
        (args.output / 'progress.json').write_text(json.dumps({'stage':stage, **extra},indent=2))
        print(json.dumps({'stage':stage, **extra}), flush=True)
    status('PREPARE_NATIVE_TRAINING')
    panel = pd.read_parquet(args.inputs / 'alpha158-panel.parquet')
    days = sorted(panel.index.get_level_values('datetime').unique())
    segments = temporal_segments(days, protocol['label_horizon'], protocol['validation_sessions'], protocol['test_sessions'])
    # Calendar rows with no eligible features must not compress a five-session label.
    calendar = pd.DatetimeIndex(pd.to_datetime((args.inputs / 'provider/calendars/day.txt').read_text().splitlines()))
    calendar = calendar[calendar >= pd.Timestamp(protocol['train_start'])]
    segments = temporal_segments(calendar, protocol['label_horizon'], protocol['validation_sessions'], protocol['test_sessions'])
    (args.output / 'time-contract.json').write_text(json.dumps(segments,indent=2))
    dates = panel.index.get_level_values('datetime')
    feature_names = manifest['native_features']
    import qlib
    from qlib.contrib.model.gbdt import LGBModel
    if args.rank_objective:
        from moneymore.research.qlib_ranker import RankingLGBModel
        LGBModel=RankingLGBModel
    from qlib.data.dataset.processor import CSZScoreNorm,CSRankNorm
    from qlib.workflow import R
    qlib.init(provider_uri=str((args.inputs / 'provider').resolve()), region='cn', kernels=1, logging_level=30,
        exp_manager={'class':'MLflowExpManager','module_path':'qlib.workflow.expm',
                     'kwargs':{'uri':'sqlite:///' + (args.output / 'mlflow.db').resolve().as_posix(),'default_exp_name':'mature_multifactor'}})
    def prepared(start, end):
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        medians = panel.loc[mask, feature_names].median()
        if not np.isfinite(medians).all():
            raise ValueError('FEATURE_HAS_NO_TRAINING_REFERENCE')
        frame = panel[feature_names].fillna(medians).copy()
        frame.columns = pd.MultiIndex.from_product([['feature'],feature_names])
        frame[('label','TOTAL_RETURN')] = panel.economic_label
        processor = fit_training_processor(frame,start,end)
        frame = transform_features(processor,frame)
        # Native Alpha158 learning convention, applied only to mature labels.
        mature = frame.loc[np.isfinite(frame[('label','TOTAL_RETURN')])].copy()
        normalizer=CSRankNorm if args.label_normalization=='rank' else CSZScoreNorm
        mature = normalizer(fields_group='label')(mature)
        return frame, mature, {'normalizer':processor,'medians':medians}
    frame, mature, processor = prepared(*segments['train'])
    dataset = QlibPanelDataset(mature, {k:segments[k] for k in ['train','valid','test']})
    model = LGBModel(**protocol['lightgbm'], seed=protocol['seed'], num_boost_round=protocol['max_rounds'],early_stopping_rounds=protocol['early_stop'])
    status('NATIVE_LGBMODEL_RESEARCH_FIT', segments=segments, rows=len(mature))
    with R.start(experiment_name='mature_alpha158_chronological'):
        from moneymore.research.prediction_quality import daily_rank_ic_evaluator
        evaluation={}
        model.fit(dataset,verbose_eval=50,evals_result=evaluation,
            **({'feval':daily_rank_ic_evaluator(dataset)} if args.selection_metric=='rank_ic' else {}))
    (args.output/'validation-metrics.json').write_text(json.dumps(evaluation,indent=2))
    test_data = QlibPanelDataset(frame, {'test':segments['test']})
    score = model.predict(test_data)
    predicted = pd.concat([score.rename('score'),panel.economic_label.reindex(score.index)],axis=1)
    predicted.to_parquet(args.output / 'test-predictions.parquet')
    ic = predicted.groupby(level='datetime').apply(lambda x:x.score.corr(x.economic_label,method='spearman'),include_groups=False)
    ic.to_frame('rank_ic').to_parquet(args.output / 'test-daily-ic.parquet')
    rounds = int(model.model.best_iteration or protocol['max_rounds'])
    (args.output / 'research-model.pkl').write_bytes(pickle.dumps(model))
    (args.output / 'research-processor.pkl').write_bytes(pickle.dumps(processor))
    status('LATEST_MATURE_LABEL_REFIT', selected_rounds=rounds, deployment_fit=segments['deployment_fit'])
    deploy_frame, deploy_mature, deploy_processor = prepared(*segments['deployment_fit'])
    # Fit only the training segment: no fake validation and no test-based early stop.
    deploy_options={**protocol['lightgbm'],'metric':'ndcg' if args.rank_objective else 'l2'}
    deploy = LGBModel(**deploy_options,seed=protocol['seed'],num_boost_round=rounds,early_stopping_rounds=protocol['early_stop'])
    with R.start(experiment_name='mature_alpha158_latest_refit'):
        deploy.fit(QlibPanelDataset(deploy_mature,{'train':segments['deployment_fit']}),verbose_eval=0)
    latest_data = QlibPanelDataset(deploy_frame,{'latest':segments['latest']})
    latest = deploy.predict(latest_data,'latest').rename('score')
    (args.output / 'model.pkl').write_bytes(pickle.dumps(deploy))
    (args.output / 'processor.pkl').write_bytes(pickle.dumps(deploy_processor))
    restored = pickle.loads((args.output / 'model.pkl').read_bytes()).predict(latest_data,'latest')
    reload_error = float(np.max(np.abs(latest.to_numpy()-restored.to_numpy())))
    if reload_error > 1e-12 or not np.isfinite(latest).all() or latest.std() <= 1e-10:
        raise ValueError('MODEL_RELOAD_OR_PREDICTION_INVALID')
    latest.to_frame().to_parquet(args.output / 'latest-predictions.parquet')
    from moneymore.research.prediction_quality import prediction_quality
    quality=prediction_quality(latest.to_numpy(),protocol.get('topk',50))
    result = {'status':'P2_NATIVE_MODEL_AND_LATEST_REFIT_READY','model_id':protocol['version'],
        'inputs':str(args.inputs.resolve()),'data_cutoff':manifest['cutoff'],'segments':segments,'selected_rounds':rounds,
        'test_days':len(ic),'test_rank_ic':float(ic.mean()),'test_positive_day_ratio':float((ic>0).mean()),
        'latest_prediction_rows':len(latest),'reload_max_abs_error':reload_error,
        'gate_scope':'data_time_model_integrity_for_independent_paper_observation_not_profit_or_promotion',
        'real_orders_enabled':False,'promotion_authorized':False,
        'prediction_quality':quality,'selection_metric':args.selection_metric,
        'historical_limits':['known_snapshot_membership_not_original_vintage','last_known_membership_carried_2026_gap',
                            'historical_research_not_new_market_evidence','Alpha158_price_volume_not_fundamental_forecaster'],
        'input_manifest_sha256':hashlib.sha256((args.inputs / 'results.json').read_bytes()).hexdigest(),
        'artifact_hashes':{name:hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in ['model.pkl','processor.pkl','latest-predictions.parquet','research-model.pkl','test-predictions.parquet','protocol.yaml','time-contract.json']}}
    if not quality['passed']:result['status']='P2_MODEL_DISCRIMINATION_BLOCKED'
    (args.output / 'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    status('P2_COMPLETE' if quality['passed'] else 'P2_BLOCKED_DISCRIMINATION',test_rank_ic=result['test_rank_ic'],latest_prediction_rows=len(latest),selected_rounds=rounds)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(limits=4):
            main()
    except Exception as error:
        import sys
        destination = Path(sys.argv[sys.argv.index('--output') + 1]) / 'progress.json'
        if destination.exists():
            prior = json.loads(destination.read_bytes())
            destination.write_text(json.dumps({'stage':'FAILED','failed_stage':prior.get('stage'),'error_type':type(error).__name__}),encoding='utf-8')
        raise
