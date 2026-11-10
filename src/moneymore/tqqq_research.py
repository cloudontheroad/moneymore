"""Causal position-sizing candidates for research. All signals are close-known."""

import numpy as np
import pandas as pd


def fast_defense(qqq: pd.Series) -> pd.Series:
    """Reduce exposure on abnormal weakness; restore on a distinct recovery signal."""
    returns = qqq.pct_change()
    weekly_vol = returns.ewm(span=40, min_periods=40).std().shift(5) * np.sqrt(5)
    shock = qqq.pct_change(5) < -1.5 * weekly_vol
    trend = qqq.ewm(span=30, min_periods=30).mean()
    weak = qqq < trend
    recovery = (qqq > qqq.shift().rolling(5).max()) & (qqq.pct_change(5) > 0)
    defending = False
    scales = []
    for enter, exit_ in zip(shock & weak, recovery, strict=True):
        if enter:
            defending = True
        elif exit_:
            defending = False
        scales.append(0.5 if defending else 1.0)
    return pd.Series(scales, index=qqq.index, name="defense_scale")


def layered_components(qqq: pd.Series, tqqq_open: pd.Series, tqqq: pd.Series) -> pd.DataFrame:
    """Separate trend, a risk ceiling, and a capped recovery overlay."""
    fast = qqq.ewm(span=30, min_periods=30).mean()
    slow = qqq.ewm(span=150, min_periods=150).mean()
    trend = (0.25 + 0.35 * (qqq > fast) + 0.40 * (fast > slow)).where(slow.notna())
    daily = tqqq.pct_change()
    overnight = tqqq_open / tqqq.shift() - 1
    daily_risk = pd.concat([daily.ewm(span=n, min_periods=n).std() for n in (10, 60)], axis=1).max(axis=1)
    gap_risk = np.sqrt(overnight.clip(upper=0).pow(2).ewm(span=40, min_periods=40).mean())
    risk = pd.concat([daily_risk, 2 * gap_risk], axis=1).max(axis=1) * np.sqrt(252)
    # Recent weakness followed by price recovery, with a short holding window.
    weak = qqq.pct_change(20).lt(0).rolling(5).max().fillna(0).astype(bool)
    recovery = weak & qqq.gt(qqq.shift().rolling(5).max()) & qqq.pct_change(3).gt(0)
    overlay = 0.20 * recovery.rolling(5, min_periods=1).max()
    return pd.DataFrame({"trend": trend, "risk": risk, "recovery": overlay})


def layered_target(components: pd.DataFrame, risk_budget: float, *, recovery: bool) -> pd.Series:
    cap = (risk_budget / components.risk).clip(upper=1)
    desired = components.trend + (components.recovery if recovery else 0)
    # Update immediately if the cap falls, and only for meaningful increases.
    values = []
    prior = 0.0
    for weight, limit in zip(desired, cap, strict=True):
        if not np.isfinite(weight) or not np.isfinite(limit):
            values.append(np.nan)
            continue
        target = min(weight, limit)
        if prior > limit or abs(target - prior) >= 0.05:
            prior = target
        values.append(min(prior, limit))
    return pd.Series(values, index=components.index)
