"""Purged chronological folds and day-block uncertainty, no row bootstrap."""
import numpy as np
import pandas as pd


def purged_fold(calendar, first_train, test_start, test_end, horizon=5, validation=20):
    days=pd.DatetimeIndex(calendar)
    if not days.is_unique or not days.is_monotonic_increasing:raise ValueError('INVALID_CALENDAR')
    start=days.searchsorted(pd.Timestamp(test_start));end=days.searchsorted(pd.Timestamp(test_end),side='right')-1
    valid_end=start-horizon-1;valid_start=valid_end-validation+1;train_end=valid_start-horizon-1
    if train_end<0 or days[train_end]<pd.Timestamp(first_train) or end-horizon<start:raise ValueError('INSUFFICIENT_PURGED_HISTORY')
    return {'train':(str(pd.Timestamp(first_train).date()),str(days[train_end].date())),
        'valid':(str(days[valid_start].date()),str(days[valid_end].date())),
        'test':(str(days[start].date()),str(days[end-horizon].date()))}


def block_interval(values, block=21, repetitions=2000, seed=20261102, comparisons=4):
    x=np.asarray(values,dtype=float)
    if len(x)<block or not np.isfinite(x).all():raise ValueError('INSUFFICIENT_FINITE_DAILY_OBSERVATIONS')
    rng=np.random.default_rng(seed);blocks=int(np.ceil(len(x)/block))
    starts=rng.integers(0,len(x),size=(repetitions,blocks))
    indices=(starts[:,:,None]+np.arange(block))%len(x)
    samples=x[indices.reshape(repetitions,-1)[:,:len(x)]].mean(axis=1)
    alpha=.05/comparisons
    return {'mean':float(x.mean()),'lower':float(np.quantile(samples,alpha/2)),
        'upper':float(np.quantile(samples,1-alpha/2)),'days':len(x),'block_sessions':block,
        'method':'circular_day_blocks_bonferroni_percentiles','comparisons':comparisons}
