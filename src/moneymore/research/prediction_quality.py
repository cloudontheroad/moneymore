"""Tie-aware admission checks and native LightGBM cross-sectional evaluation."""
import numpy as np
import pandas as pd


def prediction_quality(scores,top_k=50):
    values=pd.Series(scores,dtype=float)
    if not np.isfinite(values).all():return {'passed':False,'reason':'NONFINITE_PREDICTIONS'}
    values=values.sort_values(ascending=False)
    if len(values)<top_k:return {'passed':False,'reason':'INSUFFICIENT_PREDICTIONS'}
    cutoff=float(values.iloc[top_k-1]);above=int(values.gt(cutoff).sum());tied=int(values.eq(cutoff).sum())
    return {'passed':above+tied<=top_k,'reason':'PASS' if above+tied<=top_k else 'UNRESOLVED_TOPK_BOUNDARY_TIE',
        'rows':len(values),'distinct_scores':int(values.nunique()),'top_score_count':int(values.eq(values.iloc[0]).sum()),
        'topk':top_k,'boundary_tied':tied,'boundary_remaining_slots':top_k-above}


def daily_rank_ic_evaluator(dataset):
    dates_by_length={};labels_by_length={};cache={}
    for segment in ('train','valid'):
        frame=dataset.prepare(segment);dates=frame.index.get_level_values('datetime')
        if len(frame) in dates_by_length:raise ValueError('AMBIGUOUS_EVALUATION_SEGMENT_LENGTH')
        dates_by_length[len(frame)]=pd.factorize(dates)[0]
        if 'label' in frame.columns.get_level_values(0):labels_by_length[len(frame)]=frame['label'].iloc[:,0].to_numpy()
    def evaluate(predictions,data):
        codes=dates_by_length[len(predictions)];key=id(data)
        if key not in cache:
            counts=np.bincount(codes);y=pd.Series(labels_by_length.get(len(predictions),data.get_label())).groupby(codes).rank().to_numpy()
            y-=np.bincount(codes,weights=y)[codes]/counts[codes]
            cache[key]=(counts,y,np.bincount(codes,weights=y*y))
        counts,y,y2=cache[key];x=pd.Series(predictions).groupby(codes).rank().to_numpy()
        x-=np.bincount(codes,weights=x)[codes]/counts[codes]
        denominator=np.sqrt(np.bincount(codes,weights=x*x)*y2)
        # Constant prediction days count as zero, never silently disappear.
        ic=np.divide(np.bincount(codes,weights=x*y),denominator,out=np.zeros_like(denominator),where=denominator>0)
        return 'daily_rank_ic',float(ic.mean()),True
    return evaluate
