"""Measured named-factor returns and strictly as-of shrinkage covariance.

Uses sklearn OLS/LedoitWolf; this is risk measurement, not a forecast strategy.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf


def named_return(exposures,returns,signal_date,realized_date):
    if str(signal_date)>=str(realized_date):raise ValueError('NONCAUSAL_FACTOR_EXPOSURE_DATE')
    aligned=exposures.join(returns.rename('__return'),how='inner').replace([np.inf,-np.inf],np.nan).dropna()
    x=aligned.drop(columns='__return');active=x.columns[x.abs().sum().gt(0)]
    x=x[active]
    if len(x)<len(active)+30:raise ValueError('INSUFFICIENT_NAMED_FACTOR_CROSS_SECTION')
    model=LinearRegression(fit_intercept=False).fit(x,aligned['__return'])
    if model.rank_<len(active):raise ValueError('NAMED_FACTORS_NOT_IDENTIFIABLE')
    residual=aligned['__return']-model.predict(x)
    return pd.Series(model.coef_,index=active),residual,{'assets':len(x),'rank':int(model.rank_),
        'condition_number':float(model.singular_[0]/model.singular_[-1]),'exposure_date':signal_date,'realized_date':realized_date}


def named_covariance(factor_returns,as_of,window=120):
    observed=factor_returns.loc[factor_returns.index.astype(str)<=str(as_of)].tail(window)
    if len(observed)<window:raise ValueError('INSUFFICIENT_ASOF_FACTOR_HISTORY')
    selected=observed.dropna(axis=1)
    if selected.empty:raise ValueError('NO_COMPLETE_NAMED_FACTOR_HISTORY')
    if not np.isfinite(selected.to_numpy()).all():raise ValueError('NONFINITE_FACTOR_HISTORY')
    cov=LedoitWolf().fit(selected).covariance_
    if np.linalg.eigvalsh(cov).min()<-1e-12:raise ValueError('NAMED_COVARIANCE_NOT_PSD')
    return pd.DataFrame(cov,index=selected.columns,columns=selected.columns),{
        'as_of':as_of,'observations':len(selected),'first':str(selected.index[0]),'last':str(selected.index[-1]),
        'excluded_incomplete_factors':observed.columns.difference(selected.columns).tolist(),'estimator':'sklearn.LedoitWolf'}
