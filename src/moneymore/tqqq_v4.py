"""Frozen V4 research rules and their next-open allocation."""

import numpy as np
import pandas as pd

SECTORS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")


def weekly_risk(history, context, alternate, *, week_complete=False):
    weekly = history.groupby(history.index.to_period("W-FRI")).tail(1)
    if not week_complete and weekly.index[-1].dayofweek != 4:
        weekly = weekly.iloc[:-1]
    close = weekly.qqqClose
    price = np.log(close)
    sma = close.rolling(10).mean()
    below = close.lt(sma).where(sma.notna()).rolling(4).mean()
    features = pd.DataFrame({
        "gap26": price - price.rolling(26).max(),
        "change4": price.diff(4), "change13": price.diff(13), "change26": price.diff(26),
        "below10_share": below, "vol13": price.diff().rolling(13).std(),
        "one_week": price.diff(), "recovery_from_low": price - price.rolling(4).min(),
        "distance10": np.log(close / sma),
    })
    # Keep the same data-availability mask as the original price forecast.
    source = context.reindex(weekly.index)
    alt = alternate.reindex(weekly.index)
    chips = np.log(source.SOXX / close)
    valid = features.notna().all(axis=1) & below.gt(0)
    valid &= pd.concat([
        chips.diff(4), chips.diff(13), np.log(alt.RSPD / alt.RSPS).diff(4),
        np.log(alt.RSPD / alt.RSPS).diff(13), np.log(source.RSP / source.SPY).diff(13),
        np.log(weekly.HYG / weekly.IEI).diff(4), np.log(weekly.VIX / weekly.VIX3M),
        source["^TNX"].diff(13),
    ], axis=1).notna().all(axis=1)
    offsets = history.index.get_indexer(weekly.index) + 1
    opens = history.tqqqOpen.to_numpy()
    closes = history.tqqqClose.to_numpy()
    settled = pd.Series(pd.NaT, index=weekly.index, dtype=history.index.dtype)
    loss = pd.Series(np.nan, index=weekly.index)
    for date, start, end in zip(weekly.index, offsets[:-4], offsets[4:]):
        if end >= len(history):
            break
        settled.at[date] = history.index[end]
        loss.at[date] = -np.log(min(opens[start:end + 1].min(), closes[start:end].min()) / opens[start])

    forecast = pd.Series(np.nan, index=weekly.index)
    for date in weekly.index[valid]:
        past = valid & settled.le(date)
        if past.sum() < 52:
            continue
        values = features.loc[past].to_numpy()
        center = np.median(values, axis=0)
        scale = np.maximum(np.subtract(*np.quantile(values, (0.75, 0.25), axis=0)), 1e-8)
        x = np.clip((values - center) / scale, -4, 4)
        query = np.clip((features.loc[date].to_numpy() - center) / scale, -4, 4)
        design = np.column_stack((np.ones(len(x)), x))
        penalty = np.diag([0.0, *([26.0] * x.shape[1])])
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ loss.loc[past].to_numpy())
        forecast.at[date] = max(0., np.r_[1, query] @ weights)
    return forecast.gt(forecast.shift().rolling(104, min_periods=26).quantile(.80))


def close_signals(history, context, sectors):
    close = history.qqqClose
    q = np.log(close)
    high_age = q.rolling(63).apply(np.argmax, raw=True).rsub(62)
    shock = -q.diff(5) / (q.diff().rolling(63).std().shift(5) * np.sqrt(5))
    cross = pd.DataFrame({
        "credit": -np.log(history.HYG / history.IEI).diff(5),
        "volatility": np.log(history.VIX / history.VIX3M).diff(5),
    })
    elevated = cross.gt(cross.shift().rolling(504, min_periods=126).quantile(.80))
    confirmed = shock.ge(1.5) & elevated.credit & elevated.volatility
    industry = np.log(sectors[list(SECTORS)].reindex(q.index))
    gap = industry - industry.rolling(63).mean()
    weak = gap.lt(0).where(gap.notna()).mean(axis=1).where(gap.count(axis=1).ge(8))
    relative = q - np.log(context.SPY.reindex(q.index))
    leadership = relative.diff(126).rolling(756, min_periods=252).rank(pct=True)
    weakening = weak.diff(20).rolling(756, min_periods=252).rank(pct=True)
    reversal = (-relative.diff(5)).rolling(756, min_periods=252).rank(pct=True)
    narrowing = leadership.ge(.80) & weakening.ge(.80) & reversal.ge(.65)
    return pd.DataFrame({"peak_age": high_age, "exit": high_age.le(21) & (confirmed | narrowing)}, index=q.index)


def v4_targets(history, context, alternate, sectors, *, next_session=None):
    """Return allocations effective at each open, optionally including the next opening."""
    complete = next_session is not None and next_session.to_period("W-FRI") != history.index[-1].to_period("W-FRI")
    risk = weekly_risk(history, context, alternate, week_complete=complete)
    index = history.index if next_session is None else history.index.append(pd.DatetimeIndex([next_session]))
    safe = ~risk.reindex(index).ffill().shift().eq(True)
    clock = pd.Series(risk.index, index=risk.index).reindex(index).ffill().shift()
    average = history.qqqClose.rolling(175).mean()
    trend = pd.Series(np.select([history.qqqClose.gt(average), history.qqqClose.lt(.99 * average)],
                                [1., 0.], default=np.nan), index=history.index)
    trend = trend.reindex(index).ffill().shift().fillna(0.)
    signals = close_signals(history, context, sectors).reindex(index).shift()
    weights = []
    core = previous = 0.
    used_anchor = waiting = None
    for i, (up, permitted, week, signal) in enumerate(zip(trend, safe, clock, signals.itertuples(), strict=True)):
        core = max(core, float(permitted)) if up else 0.
        anchor = i - 1 - int(signal.peak_age) if pd.notna(signal.peak_age) else None
        if not core:
            waiting = None
        elif pd.notna(week) and signal.exit == True and anchor is not None and anchor != used_anchor and previous:
            used_anchor, waiting = anchor, week
        elif waiting is not None and week != waiting and permitted:
            waiting = None
        previous = float(bool(core) and waiting is None)
        weights.append(previous)
    return pd.Series(weights, index=index, name="v4")
