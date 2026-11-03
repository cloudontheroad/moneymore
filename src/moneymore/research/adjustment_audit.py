"""Distinguish reporting precision, factor units, and changed return paths."""
import numpy as np
import pandas as pd


def audit_adjustment_versions(frame, decimals=4):
    required=['date','symbol','cached','official']
    data=frame[required].copy().sort_values(['symbol','date'])
    if data.duplicated(['date','symbol']).any():raise ValueError('DUPLICATE_ADJUSTMENT_KEYS')
    if not np.isfinite(data[['cached','official']]).all().all() or (data[['cached','official']]<=0).any().any():
        raise ValueError('INVALID_ADJUSTMENT_FACTORS')
    data['same_reported_precision']=np.isclose(data.cached.round(decimals),data.official.round(decimals),atol=1e-12,rtol=0)
    data['version_ratio']=data.cached/data.official
    data['ratio_path_change']=False
    data['constant_scale_candidate']=False
    # Precision-consistent tolerance is determined by each observation's
    # rounding interval, not chosen from realized portfolio returns.
    half_unit=.5*10.**(-decimals)
    for _,group in data.groupby('symbol'):
        low=(group.cached-half_unit)/(group.official+half_unit)
        high=(group.cached+half_unit)/(group.official-half_unit)
        if (group.official<=half_unit).any():raise ValueError('FACTOR_BELOW_REPORTING_PRECISION')
        constant=bool(low.max()<=high.min()+1e-12)
        data.loc[group.index,'ratio_path_change']=not constant
        data.loc[group.index,'constant_scale_candidate']=constant and len(group)>=2 and not group.same_reported_precision.all()
    return data
