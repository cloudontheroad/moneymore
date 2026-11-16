"""R3: fixed quarterly refits, with label-end purging and full failed-result retention."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
from moneymore.research.pareto_rebuild import mature_training_mask
from audit_ashare_pareto_inputs import ROOT,OUT

FEATURES=['value_quality_comparable','improvement_comparable','momentum_comparable']
# Fixed before any new walk-forward model results. No test-driven configuration search.
SPECS=[('ridge_raw20','ridge',20,'raw'),('ridge_residual20','ridge',20,'residual'),('ridge_residual5','ridge',5,'residual'),('ridge_residual60','ridge',60,'residual'),('tree_residual20','tree',20,'residual'),('nonnegative_residual20','positive_ridge',20,'residual')]


def main(panel_path=None, output=None, year=2026):
    output=Path(output) if output else OUT/'walkforward';output.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(panel_path or OUT/'signals/panel.parquet')
    predictions=[];fits=[]
    quarters=[(f'{year}0101',f'{year}0331'),(f'{year}0401',f'{year}0630'),(f'{year}0701',f'{year}0930')]
    if year==2025:quarters.append(('20251001','20251231'))
    evaluation_end=quarters[-1][1]
    for start,end in quarters:
        # Predictions for the prior close are used at the first open of each quarter.
        days=sorted(panel.date.unique());previous=max(d for d in days if d<start)
        next_previous=max(d for d in days if d<end) if end==evaluation_end else end
        test=panel.loc[panel.date.between(previous,next_previous)].copy()
        if end!=evaluation_end:test=test.loc[test.date<max(d for d in days if d<=end)]
        # Quarter-end close belongs to next quarter's refit; Sep 30 signal not needed for Jan-Sep execution.
        allowed=test[FEATURES].notna().sum(axis=1).ge(2)
        for name,kind,horizon,target in SPECS:
            label=('label_' if target=='raw' else 'residual_label_')+str(horizon)
            mask=mature_training_mask(panel,start,horizon,'20240131')&np.isfinite(panel[label])&panel[FEATURES].notna().sum(axis=1).ge(2)
            train=panel.loc[mask]
            if train.date.nunique()<100:raise ValueError('INSUFFICIENT_TRAINING_DATES')
            assert pd.to_datetime(train[f'label_end_{horizon}']).max()<pd.Timestamp(start)
            # Normalized-score imputation, NEVER raw financial zero filling. Fit on past only.
            imputer=SimpleImputer(strategy='median',add_indicator=True)
            x=imputer.fit_transform(train[FEATURES]);xt=imputer.transform(test[FEATURES])
            y=train[label].to_numpy(dtype=float)
            lo,hi=np.quantile(y,[.01,.99]);y=np.clip(y,lo,hi)
            ymean=float(y.mean());ystd=float(y.std());assert ystd>1e-10
            y=(y-ymean)/ystd
            if kind=='tree':model=LGBMRegressor(n_estimators=80,learning_rate=.03,max_depth=3,num_leaves=7,min_child_samples=200,reg_lambda=10,n_jobs=2,verbosity=-1,random_state=20261111)
            else:model=Ridge(alpha=1.,positive=kind=='positive_ridge')
            model.fit(x,y)
            pred=model.predict(xt)*ystd+ymean
            result=test[['date','symbol']].copy();result['strategy']=name;result['score']=np.where(allowed,pred,np.nan);result['refit_date']=start
            predictions.append(result)
            metadata={'strategy':name,'refit_date':start,'horizon':horizon,'target':target,'training_rows':len(train),'training_dates':int(train.date.nunique()),'last_feature_date':str(train.date.max()),'last_label_end':str(train[f'label_end_{horizon}'].max()),'prediction_start':str(test.date.min()),'prediction_end':str(test.date.max()),'feature_names':imputer.get_feature_names_out().tolist(),'coefficients':model.coef_.tolist() if hasattr(model,'coef_') else None,'feature_importances':model.feature_importances_.tolist() if hasattr(model,'feature_importances_') else None,'prediction_std':float(np.std(pred)),'distinct_predictions':int(pd.Series(pred).nunique()),'training_target_clip_quantiles':[float(lo),float(hi)],'production_modified':False}
            if metadata['prediction_std']<1e-10:metadata['status']='DEGENERATE_NOT_ELIGIBLE'
            else:metadata['status']='FIT_AND_PREDICTED_NOT_PERFORMANCE_ACCEPTANCE'
            fits.append(metadata)
            with (output/f'{name}-{start}.pkl').open('wb') as handle:pickle.dump({'model':model,'imputer':imputer,'y_mean':ymean,'y_std':ystd,'metadata':metadata},handle)
            (output/'progress.json').write_text(json.dumps({'completed_fits':len(fits),'total_fits':len(quarters)*len(SPECS),'latest':metadata},ensure_ascii=False,indent=2),encoding='utf-8')
            print({'model':name,'quarter':start,'rows':len(train),'last_label_end':metadata['last_label_end']},flush=True)
    joined=pd.concat(predictions,ignore_index=True)
    assert not joined.duplicated(['strategy','date','symbol']).any()
    joined.to_parquet(output/'predictions.parquet',index=False)
    rows=[]
    for spec in SPECS:
        name,_,h,target=spec;label=('label_' if target=='raw' else 'residual_label_')+str(h)
        sample=joined.loc[joined.strategy.eq(name)].merge(panel[['date','symbol',label,f'label_end_{h}']],on=['date','symbol'],validate='one_to_one')
        sample=sample.loc[sample[f'label_end_{h}'].notna()&sample[f'label_end_{h}'].le(evaluation_end)]
        bydate=[]
        for day,g in sample.groupby('date'):
            good=g[['score',label]].dropna()
            if len(good)<100:continue
            bydate.append({'date':day,'rank_ic':float(good.score.rank().corr(good[label].rank())),'assets':len(good)})
        f=pd.DataFrame(bydate)
        rows.append({'strategy':name,'horizon':h,'target':target,'mature_test_days':len(f),'rank_ic_mean':float(f.rank_ic.mean()),'positive_day_fraction':float(f.rank_ic.gt(0).mean()),'not_comparable_horizons_without_common_date_intersection':True})
        f.to_parquet(output/f'{name}-ic.parquet',index=False)
    report={'status':'R3_FITS_COMPLETE_COSTED_REPLAY_PENDING','fits':fits,'diagnostics':rows,'models':len(SPECS),'quarterly_refits':len(quarters),'evaluation_year':year,'panel_source':str(panel_path or OUT/'signals/panel.parquet'),'production_modified':False,'test_period_previously_examined':True,'no_return_or_sharpe_claim':True}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--year',type=int,choices=[2025,2026],default=2026)
    args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.panel,args.output,args.year)
