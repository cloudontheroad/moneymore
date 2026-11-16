"""Pure, time-explicit N4 allocation utilities; no broker or data-source side effects."""
import numpy as np
import pandas as pd


def past_returns(frame, signal, lookback=120, minimum=100):
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError('INVALID_RETURN_TIMELINE')
    past = frame.loc[frame.index <= signal].tail(lookback)
    if len(past) < minimum or not np.isfinite(past.to_numpy()).all():
        raise ValueError('INSUFFICIENT_REALIZED_NET_HISTORY')
    if past.std(ddof=1).le(1e-12).any():
        raise ValueError('CONSTANT_SLEEVE_RETURN')
    return past


def validate_weights(weights, sources, maximum=.6):
    w = pd.Series(weights, dtype=float).reindex(sources, fill_value=0.)
    if not np.isfinite(w).all() or (w < -1e-10).any() or abs(w.sum()-1)>1e-8:
        raise ValueError('INVALID_SLEEVE_WEIGHTS')
    if int(w.gt(1e-10).sum()) > 1 and w.max() > maximum+1e-8:
        raise ValueError('SLEEVE_CAP')
    return w


def constrain_mix(targets, industries, benchmark, locked, stock_exposure=.995, stock_cap=.05, active_cap=.08):
    """Only reduce tradable targets to current caps; never re-rank or force reinvestment."""
    w = pd.Series(targets, dtype=float)
    lock = pd.Series(locked, dtype=float)
    universe = sorted(set(w.index)|set(lock.index))
    w = w.reindex(universe, fill_value=0.)
    ind = pd.Series(industries).reindex(universe)
    if ind.isna().any() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('INVALID_STOCK_TARGETS')
    if not np.isfinite(lock).all() or (lock<0).any() or (lock>stock_cap+1e-10).any() or lock.sum()>stock_exposure+1e-10:
        raise ValueError('LOCKED_BUDGET_OR_STOCK_CAP')
    free_names = w.index.difference(lock.index)
    w.loc[free_names] = w.loc[free_names].clip(upper=stock_cap)
    w.loc[lock.index] = lock
    before = float(w.sum())
    for sector in ind.unique():
        names = ind.index[ind.eq(sector)]
        frozen_names = names.intersection(lock.index)
        free = names.difference(lock.index)
        limit = float(benchmark.get(sector,0.))+active_cap
        frozen_weight = float(w.loc[frozen_names].sum())
        if frozen_weight > limit+1e-10:
            raise ValueError('LOCKED_INDUSTRY_CAP')
        total = float(w.loc[free].sum())
        if total > limit-frozen_weight and total>0:
            w.loc[free] *= max(0.,limit-frozen_weight)/total
    free_total = float(w.loc[free_names].sum())
    budget = max(0.,stock_exposure-float(lock.sum()))
    if free_total>budget:
        w.loc[free_names] *= budget/free_total
    if (w.groupby(ind).sum()-pd.Series(benchmark).reindex(ind.unique(),fill_value=0)-active_cap).max()>1e-9:
        raise ValueError('TARGET_PROJECTION_FAILED')
    return w.loc[w.gt(0)].to_dict(), {'before_exposure':before,'after_exposure':float(w.sum()),'reduced_exposure':before-float(w.sum())}


def netting_audit(desired_by_source, ownership, held, prices):
    """Fractional virtual ownership sums to actual shares; net once before global lotting."""
    sources = sorted(desired_by_source)
    symbols = sorted(set(held)|set().union(*(set(v) for v in desired_by_source.values())))
    deltas = []
    for source in sources:
        row = {}
        for s in symbols:
            portions = ownership.get(s,{})
            total = sum(portions.values())
            previous = held.get(s,0)*(portions.get(source,0)/total if total>0 else (1 if source==sources[0] else 0))
            row[s] = (desired_by_source[source].get(s,0)-previous)*prices[s]
        deltas.append(row)
    frame = pd.DataFrame(deltas,index=sources).fillna(0)
    gross = float(frame.abs().to_numpy().sum())
    net = float(frame.sum().abs().sum())
    if net > gross+1e-6:
        raise ValueError('NETTING_INCREASED_GROSS')
    return {'virtual_gross_notional':gross,'net_notional_before_lot_cash_and_permissions':net,'crossed_notional_both_sides':max(0.,gross-net),'theoretical_not_actual_fee_savings':True}
