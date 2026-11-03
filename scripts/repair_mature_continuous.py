"""One fixed native Qlib Ridge control, not a parameter or model search."""
from pathlib import Path
import pickle,json,hashlib
import numpy as np
import pandas as pd
import yaml
from qlib.contrib.model.linear import LinearModel
from qlib.data.dataset.processor import CSZScoreNorm
from moneymore.qlib_challenger import QlibPanelDataset
from moneymore.research.qlib_feature_preprocessing import transform_features
from moneymore.research.prediction_quality import prediction_quality
from threadpoolctl import threadpool_limits

def main():
    root=Path.cwd();old=root/'state/mature-multifactor/p2-model-20261103-b'
    output=root/'state/mature-multifactor/p2-ridge-repair-20261103-b'
    output.mkdir(exist_ok=False);meta=json.loads((old/'results.json').read_bytes());segments=meta['segments']
    panel=pd.read_parquet(Path(meta['inputs'])/'alpha158-panel.parquet')
    def prepare(frame,processor):
        names=processor['normalizer']['schema'];x=frame[names].fillna(processor['medians']).copy()
        x.columns=pd.MultiIndex.from_product([['feature'],names]);x[('label','TOTAL_RETURN')]=frame.economic_label if 'economic_label' in frame else np.nan
        x=transform_features(processor['normalizer'],x).astype('float64')
        mature=CSZScoreNorm(fields_group='label')(x.loc[np.isfinite(x[('label','TOTAL_RETURN')])].copy())
        return x,mature
    processor=pickle.loads((old/'research-processor.pkl').read_bytes());frame,mature=prepare(panel,processor)
    model=LinearModel(estimator='ridge',alpha=1.,fit_intercept=False,include_valid=False)
    model.fit(QlibPanelDataset(mature,{'train':segments['train']}))
    metrics={}
    for segment in ['valid','test']:
        scores=model.predict(QlibPanelDataset(frame,{segment:segments[segment]}),segment)
        paired=pd.concat([scores.rename('score'),panel.economic_label.reindex(scores.index)],axis=1)
        ic=paired.groupby(level='datetime').apply(lambda g:g.score.corr(g.economic_label,method='spearman'),include_groups=False)
        paired.to_parquet(output/(segment+'-predictions.parquet'))
        metrics[segment+'_rank_ic']=float(ic.mean())
    deploy_processor=pickle.loads((old/'processor.pkl').read_bytes());deploy_frame,deploy_mature=prepare(panel,deploy_processor)
    deploy=LinearModel(estimator='ridge',alpha=1.,fit_intercept=False,include_valid=False)
    deploy.fit(QlibPanelDataset(deploy_mature,{'train':segments['deployment_fit']}))
    latest_panel=pd.read_parquet(root/'state/mature-multifactor/runtime/features/20261103-a/alpha158-panel.parquet')
    latest_frame,_=prepare(latest_panel,deploy_processor);latest_segment=['2026-11-03','2026-11-03']
    dataset=QlibPanelDataset(latest_frame,{'latest':latest_segment});scores=deploy.predict(dataset,'latest')
    from moneymore.strategy_universe import active_strategy_universe
    from moneymore.data.store import ParquetStore
    members=set(active_strategy_universe(ParquetStore(root/'data'),'20261103').symbol)
    scores=scores.loc[scores.index.get_level_values('instrument').isin(members)]
    quality=prediction_quality(scores.to_numpy())
    (output/'model.pkl').write_bytes(pickle.dumps(deploy));(output/'processor.pkl').write_bytes(pickle.dumps(deploy_processor))
    scores.rename('score').to_frame().to_parquet(output/'latest-predictions.parquet')
    reload_error=float(abs(pickle.loads((output/'model.pkl').read_bytes()).predict(dataset,'latest').reindex(scores.index)-scores).max())
    protocol=yaml.safe_load((old/'protocol.yaml').read_text(encoding='utf-8'));protocol.update(version='mature_alpha158_ridge_v2',model_class='qlib.contrib.model.linear.LinearModel',estimator='ridge',alpha=1.,selection='single_fixed_native_control_validation_rank_ic_not_test_selected')
    (output/'protocol.yaml').write_text(yaml.safe_dump(protocol),encoding='utf-8')
    result={**meta,**metrics,'status':'P2_CONTINUOUS_REPAIR_EVALUATED','model_id':protocol['version'],'model_class':protocol['model_class'],
        'data_cutoff':'20261103','segments':{**segments,'latest':latest_segment},'selected_rounds':None,'prediction_quality':quality,
        'reload_max_abs_error':reload_error,'test_rank_ic':metrics['test_rank_ic'],'real_orders_enabled':False,'promotion_authorized':False,
        'validation_reference_rank_ic':0.005096210920902666,'validation_improved':metrics['valid_rank_ic']>0.005096210920902666,
        'source_model':str(old),'artifact_hashes':{n:hashlib.sha256((output/n).read_bytes()).hexdigest() for n in ['model.pkl','processor.pkl','latest-predictions.parquet','protocol.yaml']}}
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
