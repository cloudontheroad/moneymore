"""Freeze a small 2025-only validation choice; never inspect 2026 performance."""
import hashlib
import json
import numpy as np
import pandas as pd
from audit_ashare_pareto_inputs import OUT


def main():
    validation=OUT/'validation2025'
    report=json.loads((validation/'results.json').read_bytes())
    if report['evaluation_year']!=2025 or len(report['fits'])!=24:raise ValueError('INCOMPLETE_VALIDATION_FITS')
    for fit in report['fits']:
        if fit['last_label_end']>=fit['refit_date']:raise ValueError('LABEL_LEAKAGE')
    forecasts=pd.read_parquet(validation/'predictions.parquet')
    forecasts=forecasts.loc[forecasts.strategy.isin(['tree_residual20','nonnegative_residual20'])]
    table=forecasts.pivot(index=['date','symbol'],columns='strategy',values='score')
    labels=pd.read_parquet(OUT/'label-repair-v2/panel.parquet',columns=['date','symbol','residual_label_20','label_end_20']).set_index(['date','symbol'])
    table=table.join(labels,how='inner')
    table=table.loc[table.index.get_level_values('date').str.startswith('2025')&table.label_end_20.notna()&table.label_end_20.le('20251231')].dropna()
    # Both models use the same residual20 return target. No 2026 metric is read.
    candidates=[];daily=[]
    for tree_weight in (0.,.5,1.):
        score=tree_weight*table.tree_residual20+(1-tree_weight)*table.nonnegative_residual20
        f=table.assign(score=score).reset_index()
        points=[]
        for day,g in f.groupby('date'):
            if len(g)<100 or g.score.nunique()<10:continue
            points.append({'date':day,'rank_ic':float(g.score.rank().corr(g.residual_label_20.rank())),'assets':len(g)})
        bydate=pd.DataFrame(points)
        bydate['quarter']=pd.to_datetime(bydate.date).dt.to_period('Q').astype(str)
        quarter=bydate.groupby('quarter').rank_ic.mean()
        complete=set(quarter.index)=={'2025Q1','2025Q2','2025Q3','2025Q4'}
        candidates.append({'tree_weight':tree_weight,'nonnegative_weight':1-tree_weight,'quarter_rank_ic':quarter.to_dict(),'median_quarter_rank_ic':float(quarter.median()),'all_quarters_positive':bool(complete and quarter.gt(0).all()),'validation_days':len(bydate)})
        bydate['tree_weight']=tree_weight;daily.append(bydate)
    eligible=[c for c in candidates if c['all_quarters_positive']]
    selected=max(eligible,key=lambda c:(c['median_quarter_rank_ic'],-abs(c['tree_weight']-.5))) if eligible else None
    destination=OUT/'blend-selection';destination.mkdir(exist_ok=True)
    pd.concat(daily,ignore_index=True).to_parquet(destination/'validation-dates.parquet',index=False)
    result={'selection_rule':'three fixed convex weights 0/0.5/1; all four quarterly mean IC positive; maximize median quarterly IC; tie prefer equal blend','candidates':candidates,'selected':selected,'status':'VALIDATION_SIGNAL_SELECTED_NOT_PORTFOLIO_ACCEPTANCE' if selected else 'NO_ALL_QUARTER_POSITIVE_BLEND','validation_predictions_sha256':hashlib.sha256((validation/'predictions.parquet').read_bytes()).hexdigest(),'uses_2026_performance':False,'costed_validation_completed':False,'production_modified':False}
    (destination/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
