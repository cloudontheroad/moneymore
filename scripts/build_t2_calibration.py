"""Fixed-family, purged chronological OOF calibration; no 2026 tuning."""
from pathlib import Path
import json,pickle,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from qlib.contrib.model.linear import LinearModel
from qlib.data.dataset.processor import CSZScoreNorm
from moneymore.qlib_challenger import QlibPanelDataset
from moneymore.research.qlib_feature_preprocessing import fit_training_processor,transform_features

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'state/cost-aware-ab-v2/t2';OUT.mkdir(parents=True,exist_ok=True)

def main():
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a'
    panel=pd.read_parquet(source/'alpha158-panel.parquet');manifest=json.loads((source/'results.json').read_bytes());names=manifest['native_features']
    calendar=pd.DatetimeIndex(pd.to_datetime((source/'provider/calendars/day.txt').read_text().splitlines()))
    dates=panel.index.get_level_values('datetime');bars=pd.read_parquet(source/'bars-input.parquet')
    bars['day']=pd.to_datetime(bars.trade_date);bars['c']=bars.close*bars.adj_factor;bars['o']=bars.open*bars.adj_factor
    close=bars.pivot(index='day',columns='ts_code',values='c').reindex(calendar)
    opening=bars.pivot(index='day',columns='ts_code',values='o').reindex(calendar)
    target=(close.shift(-5)/opening.shift(-1)-1).stack(future_stack=True)
    target.index=target.index.set_names(['datetime','instrument'])
    rows=[]
    for start in pd.date_range('2024-07-01','2026-10-01',freq='QS'):
        last=min(start+pd.offsets.QuarterEnd(),pd.Timestamp('2026-10-26'))
        end_train=calendar[calendar<start][-6]
        folder=OUT/start.strftime('%Y%m%d');folder.mkdir(exist_ok=True)
        path=folder/'predictions.parquet'
        if path.exists():rows.append(pd.read_parquet(path));continue
        mask=(dates>=pd.Timestamp('2024-01-31'))&(dates<=end_train)&np.isfinite(panel.economic_label)
        medians=panel.loc[mask,names].median();assert medians.notna().all()
        frame=panel[names].fillna(medians);frame.columns=pd.MultiIndex.from_product([['feature'],names]);frame[('label','TOTAL_RETURN')]=panel.economic_label
        processor=fit_training_processor(frame,'20240131',end_train.strftime('%Y%m%d'))
        frame=transform_features(processor,frame).astype('float64')
        train=CSZScoreNorm(fields_group='label')(frame.loc[mask].copy())
        model=LinearModel(estimator='ridge',alpha=1.,fit_intercept=False,include_valid=False)
        model.fit(QlibPanelDataset(train,{'train':('20240131',end_train.strftime('%Y%m%d'))}))
        predicted=model.predict(QlibPanelDataset(frame,{'test':(start.strftime('%Y%m%d'),last.strftime('%Y%m%d'))}),'test')
        result=predicted.rename('score').to_frame();result['return']=target.reindex(result.index);result['fold']=start.strftime('%Y%m%d');result.to_parquet(path)
        proof={'train_end':end_train.strftime('%Y%m%d'),'prediction_start':start.strftime('%Y%m%d'),'prediction_end':last.strftime('%Y%m%d'),
            'last_training_label_observed':calendar[calendar.get_loc(end_train)+5].strftime('%Y%m%d'),'alpha':1.,'rows':len(result)}
        assert proof['last_training_label_observed']<proof['prediction_start']
        (folder/'contract.json').write_text(json.dumps(proof,indent=2));rows.append(result)
        (OUT/'progress.json').write_text(json.dumps({'folds':len(rows),'last_fold':str(start.date())}));print(json.dumps(proof),flush=True)
    combined=pd.concat(rows).sort_index();assert combined.index.is_unique
    combined.to_parquet(OUT/'oof.parquet')
    for label,cutoff in [('historical','20251231'),('deployment','20261102')]:
        last_signal=calendar[calendar<=pd.Timestamp(cutoff)][-6]
        f=combined.loc[combined.index.get_level_values('datetime')<=last_signal].copy()
        records=[]
        for day,g in f.groupby(level='datetime'):
            finite=g['return'].notna();z=(g.score-g.score.mean())/g.score.std(ddof=0)
            y=g['return']-g.loc[finite,'return'].mean();z=z.loc[finite];y=y.loc[finite]
            if len(z)<100:continue
            records.append({'day':day.strftime('%Y%m%d'),'beta':float(z@y/(z@z)),'rows':len(z)})
        daily=pd.DataFrame(records);v=daily.beta.to_numpy();rng=np.random.default_rng(17);n=len(v);means=[]
        for _ in range(1000):
            starts=rng.integers(0,n,size=(n+4)//5);sample=np.concatenate([v[(np.arange(5)+s)%n] for s in starts])[:n];means.append(sample.mean())
        estimate=float(v.mean());uncertainty=float(np.std(means,ddof=1));conservative=max(0.,estimate-uncertainty)
        artifact={'schema':'COST_CALIBRATION_V1','family':'qlib_alpha158_native_ridge_alpha1','horizon':5,'score_transform':'same_day_cross_section_zscore',
            'entry':'next_session_open','exit':'fifth_session_close','return_unit':'decimal_relative_cross_section_return',
            'beta_estimate':estimate,'mean_estimation_se_block5':uncertainty,'beta_conservative':conservative,
            'shrink_rule':'one_standard_error_toward_zero_predeclared_not_significance_claim','minimum_expected_advantage':0.,
            'label_cutoff':cutoff,'last_signal':last_signal.strftime('%Y%m%d'),'oof_days':n,'oof_rows':int(daily.rows.sum()),
            'forecast_noise_separate_from_mean_uncertainty':True,'no_2026_parameter_tuning':True,
            'oof_sha256':hashlib.sha256((OUT/'oof.parquet').read_bytes()).hexdigest(),'optional_alpha_enabled':conservative>0}
        (OUT/(label+'.json')).write_text(json.dumps(artifact,indent=2));daily.to_parquet(OUT/(label+'-daily.parquet'),index=False);print(json.dumps(artifact),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
