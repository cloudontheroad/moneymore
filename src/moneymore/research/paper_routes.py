"""Predeclared, long-only adaptations of three published mechanisms (not replications)."""
import numpy as np
import pandas as pd


def earnings_yield(basic, members):
    """Use dated daily_basic EP; remove bottom 30% of the FULL supplied market by cap."""
    frame = basic.drop_duplicates('ts_code').set_index('ts_code')
    size = pd.to_numeric(frame.total_mv, errors='coerce')
    cutoff = size.loc[size.gt(0)].quantile(.30)
    pe = pd.to_numeric(frame.pe_ttm, errors='coerce')
    ep = (1 / pe).replace([np.inf, -np.inf], np.nan)
    return ep.loc[frame.index.isin(members) & size.ge(cutoff) & pe.ne(0)].dropna()


def buffered_targets(scores, held_weights, topk=50, exit_rank=100, n_drop=5):
    """Buy in top K, retain until outside 2K; never refill from outside top K."""
    scores = scores.sort_index().sort_values(ascending=False, kind='stable')
    rank = pd.Series(np.arange(1, len(scores)+1), index=scores.index)
    # Match the reference's five replacements/day; otherwise a rank buffer could
    # silently remove its turnover cap and confound the intended comparison.
    held_ranks = rank.reindex(held_weights.index).fillna(np.inf).sort_values(ascending=False)
    exits = held_ranks.loc[held_ranks.gt(exit_rank)].index[:n_drop]
    keep = held_weights.drop(index=exits)
    entrants = [s for s in scores.index[:topk] if s not in keep.index][:max(0, topk-len(keep))]
    targets = keep.to_dict()
    if entrants:
        targets.update({s: max(0., 1-float(keep.sum()))/len(entrants) for s in entrants})
    return targets


def monthly_inverse_variance(market_returns, signal_dates):
    """Previous completed month's variance; c frozen at pre-2026 median, capped 100%."""
    ret = market_returns.copy()
    ret.index = pd.to_datetime(ret.index)
    variance = ret.groupby(ret.index.to_period('M')).var(ddof=1)
    count = ret.groupby(ret.index.to_period('M')).count()
    variance = variance.where(count.ge(10))
    calibration = variance.loc[(variance.index >= pd.Period('2024-01')) & (variance.index <= pd.Period('2025-12'))].dropna()
    if len(calibration) < 12:
        raise ValueError('INSUFFICIENT_PRE2026_VARIANCE_CALIBRATION')
    c = float(calibration.median())
    result = {}
    for day in signal_dates:
        # The signal is at a close: use that month only if this is its last market session.
        stamp = pd.Timestamp(day)
        month = stamp.to_period('M')
        month_sessions = ret.index[ret.index.to_period('M') == month]
        completed = month if stamp == month_sessions.max() else month-1
        v = variance.get(completed, np.nan)
        if not np.isfinite(v) or v <= 0:
            raise ValueError('MISSING_PREVIOUS_MONTH_VARIANCE:'+day)
        result[day] = min(1., c/float(v))
    return result, c
