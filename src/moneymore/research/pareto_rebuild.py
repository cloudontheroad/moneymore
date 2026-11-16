"""Comparable research metrics: a Pareto front is not evidence of alpha."""
import numpy as np
import pandas as pd

OBJECTIVES={'annualized_return':1,'sharpe':1,'maximum_drawdown':-1,'annualized_one_way_turnover':-1}
CONTRACT=('window_start','window_end','sessions','initial_cash','slippage_bps','execution_version')


def pareto_front(records, tolerance=1e-10):
    if tolerance < 0:raise ValueError('NEGATIVE_TOLERANCE')
    if not records:return []
    if len({r['strategy'] for r in records})!=len(records):raise ValueError('DUPLICATE_STRATEGY')
    if any(key not in row for row in records for key in CONTRACT):raise ValueError('MISSING_COMPARABILITY_CONTRACT')
    expected=tuple(records[0][k] for k in CONTRACT)
    if any(tuple(row[k] for k in CONTRACT)!=expected for row in records):raise ValueError('INCOMPARABLE_EXPERIMENTS')
    matrix=np.array([[r.get(k,np.nan)*sign if r.get(k) is not None else np.nan for k,sign in OBJECTIVES.items()] for r in records],dtype=float)
    result=[]
    for i,row in enumerate(records):
        if not np.isfinite(matrix[i]).all() or row.get('complete') is not True:
            result.append({**row,'pareto_status':'INELIGIBLE','dominated_by':[]});continue
        dominators=[]
        for j,other in enumerate(records):
            if i==j or other.get('complete') is not True or not np.isfinite(matrix[j]).all():continue
            if np.all(matrix[j]>=matrix[i]-tolerance) and np.any(matrix[j]>matrix[i]+tolerance):dominators.append(other['strategy'])
        result.append({**row,'pareto_status':'DOMINATED' if dominators else 'NONDOMINATED','dominated_by':dominators})
    return result


def mature_training_mask(frame, refit_date, horizon, start=None):
    dates=pd.to_datetime(frame['date'])
    ends=pd.to_datetime(frame[f'label_end_{horizon}'],errors='coerce')
    mask=dates.lt(pd.Timestamp(refit_date)) & ends.lt(pd.Timestamp(refit_date)) & ends.gt(dates)
    if start is not None:mask &= dates.ge(pd.Timestamp(start))
    return mask


def residualize_cross_section(values, industries, market_caps, min_group=5):
    """Only signal-date controls; missing controls cannot silently become raw labels."""
    frame=pd.concat([values.rename('y'),industries.rename('industry'),market_caps.rename('cap')],axis=1)
    valid=np.isfinite(frame.y) & np.isfinite(frame.cap) & frame.cap.gt(0) & frame.industry.notna()
    f=frame.loc[valid].copy()
    counts=f.groupby('industry').y.transform('size')
    f=f.loc[counts.ge(min_group)]
    result=pd.Series(np.nan,index=frame.index,dtype=float)
    if len(f)<max(20,f.industry.nunique()+3):return result
    size=np.log(f.cap);std=size.std(ddof=0)
    if not np.isfinite(std) or std<1e-12:return result
    design=pd.get_dummies(f.industry,dtype=float)
    design['logsize']=(size-size.mean())/std
    x=design.to_numpy();y=f.y.to_numpy()
    coef=np.linalg.lstsq(x,y,rcond=None)[0]
    result.loc[f.index]=y-x@coef
    return result
