"""Comparable net-of-cost daily NAV statistics, including initial deployment day."""
import numpy as np
import pandas as pd

def period_statistics(equity, initial_cash, annual_sessions=252, annual_risk_free=0.):
    values=pd.Series(equity,dtype=float).reset_index(drop=True)
    if values.empty or not np.isfinite(values).all() or not values.gt(0).all() or initial_cash<=0:
        raise ValueError('INVALID_NAV_SERIES')
    previous=values.shift(1);previous.iloc[0]=initial_cash
    returns=values/previous-1
    daily_std=float(returns.std(ddof=1)) if len(returns)>1 else 0.
    excess=returns-((1+annual_risk_free)**(1/annual_sessions)-1)
    peaks=values.cummax().clip(lower=initial_cash)
    return {'sessions':len(values),'net_return':float(values.iloc[-1]/initial_cash-1),
        'annualized_return':float((values.iloc[-1]/initial_cash)**(annual_sessions/len(values))-1),
        'annualized_volatility':daily_std*np.sqrt(annual_sessions),
        'sharpe':float(excess.mean()/daily_std*np.sqrt(annual_sessions)) if daily_std>0 else None,
        'maximum_drawdown':float((1-values/peaks).max()),'annual_sessions':annual_sessions,
        'annual_risk_free':annual_risk_free,'first_day_included':True}
