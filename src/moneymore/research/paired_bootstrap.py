"""Paired circular date-block intervals, not independent stock-row inference."""
import numpy as np
import pandas as pd


def paired_block_intervals(candidate, reference, *, block=20, draws=2000, seed=20261111):
    if not isinstance(candidate,pd.Series) or not isinstance(reference,pd.Series):
        raise ValueError('DATED_RETURNS_REQUIRED')
    if not candidate.index.equals(reference.index) or not candidate.index.is_unique:
        raise ValueError('UNPAIRED_DATES')
    a=candidate.to_numpy(float);b=reference.to_numpy(float);n=len(a)
    if n<2*block or block<1 or draws<100 or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<=-1:
        raise ValueError('INVALID_BOOTSTRAP_SAMPLE')
    def stats(x):
        wealth=np.cumprod(1+x)
        peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
        sd=x.std(ddof=1)
        return np.array([wealth[-1]**(252/len(x))-1, np.sqrt(252)*x.mean()/sd if sd>1e-12 else np.nan, np.max(1-wealth/peak)])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        indices=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        delta=stats(a[indices])-stats(b[indices]);delta[2]*=-1
        values.append(delta)
    values=np.asarray(values);point=stats(a)-stats(b);point[2]*=-1
    names=['annualized_return_gain','sharpe_gain','drawdown_reduction']
    result={name:{'point':float(point[i]),'lower_95':float(np.nanquantile(values[:,i],.025)),'upper_95':float(np.nanquantile(values[:,i],.975))} for i,name in enumerate(names)}
    return {'metrics':result,'block_sessions':block,'draws':draws,'seed':seed,'sessions':n,'all_three_lower_bounds_positive':all(v['lower_95']>0 for v in result.values()),'turnover_not_inferred_from_returns':True,'retrospective_not_out_of_sample_proof':True}
