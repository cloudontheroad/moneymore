"""Small data/time adapters for the native Qlib Alpha158 observation family."""
import numpy as np
import pandas as pd


def native_bar_fields(frame):
    """Tushare daily: hundred shares, thousand CNY; no missing-day filling."""
    frame = frame.sort_values('trade_date').copy()
    factor = pd.to_numeric(frame.adj_factor, errors='coerce').where(lambda x: x > 0)
    if factor.notna().sum() == 0:
        raise ValueError('NO_OBSERVED_ADJUSTMENT_FACTOR')
    normalized = factor / factor.dropna().iloc[0]
    result = pd.DataFrame(index=frame.index)
    for field in ['open', 'high', 'low', 'close']:
        result[field] = pd.to_numeric(frame[field], errors='coerce').where(lambda x: x > 0) * normalized
    shares = pd.to_numeric(frame.vol, errors='coerce').where(lambda x: x > 0) * 100
    currency = pd.to_numeric(frame.amount, errors='coerce').where(lambda x: x > 0) * 1000
    raw_vwap = currency / shares
    result['vwap'] = raw_vwap * normalized
    result['volume'] = shares / normalized
    result['factor'] = normalized
    invalid_vwap = raw_vwap.notna() & ~raw_vwap.between(frame.low * .98, frame.high * 1.02)
    result.loc[invalid_vwap, 'vwap'] = np.nan
    result.index = frame.trade_date.astype(str)
    return result.replace([np.inf, -np.inf], np.nan), {'invalid_vwap': int(invalid_vwap.sum())}


def dated_membership(calendar, historical, snapshots, start):
    """Use only observed memberships; explicitly carry the last known snapshot."""
    historical = historical.copy()
    historical['signal_date'] = historical.signal_date.astype(str)
    snapshots = snapshots.copy()
    snapshots['effective_date'] = snapshots.effective_date.astype(str)
    if (snapshots.market_data_date.astype(str) > snapshots.effective_date).any():
        raise ValueError('FUTURE_UNIVERSE_SOURCE')
    if (historical.trade_date.astype(str) >= historical.signal_date).any():
        raise ValueError('FUTURE_HISTORICAL_POOL')
    known = {str(d): sorted(g.ts_code.astype(str)) for d, g in historical.groupby('signal_date')}
    for day, group in snapshots.groupby('effective_date'):
        known[str(day)] = sorted(group.symbol.astype(str))
    rows, carried = [], []
    keys = sorted(known)
    for day in calendar:
        if day < start:
            continue
        prior = [key for key in keys if key <= day]
        if not prior:
            continue
        source = prior[-1]
        rows.extend((pd.Timestamp(day), symbol) for symbol in known[source])
        if source != day:
            carried.append({'date': day, 'membership_as_of': source, 'size': len(known[source])})
    return pd.MultiIndex.from_tuples(rows, names=['datetime', 'instrument']), carried


def temporal_segments(calendar, horizon=5, validation=60, test=63):
    """Label endpoints must precede the next evaluation feature date."""
    days = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    mature_end = len(days) - horizon - 1
    test_first = mature_end - test + 1
    valid_last = test_first - horizon - 1
    valid_first = valid_last - validation + 1
    train_last = valid_first - horizon - 1
    if train_last < 120:
        raise ValueError('INSUFFICIENT_CHRONOLOGICAL_TRAINING_HISTORY')
    return {
        'train': (str(days[0].date()), str(days[train_last].date())),
        'valid': (str(days[valid_first].date()), str(days[valid_last].date())),
        'test': (str(days[test_first].date()), str(days[mature_end].date())),
        'deployment_fit': (str(days[0].date()), str(days[mature_end].date())),
        'latest': (str(days[-1].date()), str(days[-1].date())),
    }


def topk_dropout_targets(scores, held_weights, topk=50, n_drop=5):
    """Qlib TopkDropout top/bottom ranking and new-buy cash allocation semantics.

    Selection-only adapter to the local A-share PaperBroker, not Qlib's exchange.
    Reference: microsoft/qlib, contrib/strategy/signal_strategy.py (MIT).
    """
    if not scores.index.is_unique or not np.isfinite(scores).all() or scores.std() <= 1e-12:
        raise ValueError('INVALID_RAW_CONTROL_SCORE')
    if set(held_weights.index) - set(scores.index):
        raise ValueError('HELD_SCORE_MISSING_NOT_AN_EXIT')
    last = scores.reindex(held_weights.index).sort_values(ascending=False).index
    today = scores.loc[~scores.index.isin(last)].sort_values(ascending=False).index[:n_drop+topk-len(last)]
    combined = scores.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
    sell = last[last.isin(combined[-n_drop:])]
    buy = today[:len(sell)+topk-len(last)]
    retained = held_weights.drop(index=sell)
    result = retained.to_dict()
    if len(buy):
        budget = max(0., 1-float(retained.sum())) / len(buy)
        result.update({symbol:budget for symbol in buy})
    return result


def current_risk_inputs(store, bars, scores, day, *, suspension_proof=None, candidate_members=None, basic_overlay=None):
    """Prior-known daily risk, explicit missing controls; no sector capital sleeves."""
    from moneymore.challenger_diagnostics import trading_dates
    from moneymore.research.style_price_features import named_price_risk_exposures
    from moneymore.research.style_risk_inputs import signed_inverse
    from moneymore.research.style_missing_controls import uncertain_controls
    from moneymore.research.baseline_v2_industry import industry_as_of
    from moneymore.data.fundamental_research import research_financial_frame
    from moneymore.research.baseline_v2_financials import common_period_features, required_period
    calendar = [date for date in trading_dates(store,day) if date <= day][-121:]
    if len(calendar) != 121 or calendar[-1] != day:
        raise ValueError('INCOMPLETE_CURRENT_RISK_CALENDAR')
    sample = bars.loc[bars.trade_date.astype(str).isin(calendar) & bars.ts_code.isin(scores.index)].copy()
    sample['adjusted'] = sample.close * sample.adj_factor
    prices = sample.pivot(index='trade_date',columns='ts_code',values='adjusted').reindex(calendar)
    carried = []
    if suspension_proof is not None and not suspension_proof.empty:
        from moneymore.research.enhanced_indexing import carry_confirmed_suspensions
        prices, carried = carry_confirmed_suspensions(prices, suspension_proof.loc[suspension_proof.trade_date.astype(str).le(day)])
    basic = store.read('daily_basic',filters=[('trade_date','<=',day),('trade_date','>=',calendar[-21]),('ts_code','in',list(scores.index))])
    if basic_overlay is not None:
        extra=basic_overlay.loc[basic_overlay.trade_date.astype(str).between(calendar[-21],day)&basic_overlay.ts_code.isin(scores.index)]
        basic=pd.concat([basic,extra],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
    if 'source_trade_date' in basic:
        basic = basic.loc[basic.source_trade_date.isna() | basic.source_trade_date.astype(str).le(basic.trade_date.astype(str))]
    attrs = basic.sort_values('trade_date').groupby('ts_code').tail(1).set_index('ts_code').reindex(scores.index)
    raw, price_audit = named_price_risk_exposures(prices.reindex(columns=scores.index),attrs.total_mv*10000)
    raw['book_to_price'] = signed_inverse(attrs.pb)
    raw['earnings_yield'] = signed_inverse(attrs.pe_ttm)
    raw['dividend_yield'] = attrs.dv_ttm / 100
    finance = common_period_features(research_financial_frame(store),day).set_index('ts_code')
    for field,name in [('roe','quality_roe'),('q_sales_yoy','growth_sales_yoy')]:
        raw[name] = finance[field].where(finance[field+'_end_date'].astype(str).eq(required_period(day))).reindex(scores.index) / 100
    industry = industry_as_of(store.read('baseline_v2_industry_membership'),day)
    raw['industry'] = raw.index.map(industry.l1_code).fillna('UNKNOWN')
    raw['listing_board'] = ['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in raw.index]
    controls, eligible, missing = uncertain_controls(raw.replace([np.inf,-np.inf],np.nan),
        ['log_size','momentum60','volatility20','market_beta120','residual_volatility120'])
    # Board is observed, not a mandatory allocation floor or technology quota.
    controls = controls.drop(columns=[c for c in controls if c.startswith('board_')])
    returns = prices.reindex(columns=controls.index).pct_change(fill_method=None).iloc[1:]
    cap = attrs.total_mv.reindex(controls.index)
    if not np.isfinite(returns.to_numpy()).all() or not cap.gt(0).all():
        raise ValueError('INVALID_RETURN_OR_REFERENCE_CAP')
    reference = cap if candidate_members is None else cap.where(cap.index.isin(candidate_members),0.)
    benchmark = reference/reference.sum()
    limits = {c:.01 if c.startswith('unknown_') else .03 if c.startswith('industry_') else .10 for c in controls}
    return returns, cap, raw, controls, benchmark, limits, {'price':price_audit,'missing':missing,
        'excluded_symbols':sorted(set(scores.index)-set(controls.index)),
        'benchmark':'latest_known_cap_weights_within_shared_risk_eligible_pool_not_official_index',
        'cap_source_dates':attrs.trade_date.reindex(controls.index).astype(str).value_counts().to_dict(),
        'official_halt_carries':carried,'unknown_price_gaps_filled':False}
