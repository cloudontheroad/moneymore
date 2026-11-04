"""WildWest-inspired TQQQ alpha candidate, independently implemented.

The benchmark is 100% TQQQ.  Every cash allocation is therefore an active bet
that must earn positive net excess return after next-open execution costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V3Parameters:
    momentum_days: int = 20
    volatility_days: int = 20
    volatility_median_days: int = 504
    volatility_floor: float = 1.0
    volatility_ceiling: float = 2.5
    negative_momentum_scale: float = 0.30
    rsi_days: int = 14
    oversold_rsi: float = 30.0
    dip_buy_allocation: float = 0.80
    rebalance_band: float = 0.10
    use_relative_volatility: bool = True
    use_dip_buy: bool = True
    use_vix_confirmation: bool = False
    use_reversal_confirmation: bool = False


def rsi(close: pd.Series, days: int) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / days, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / days, adjust=False).mean()
    relative = gain / loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative)
    return result.where(loss > 0.0, 100.0).fillna(50.0)


def raw_targets(
    qqq_close: pd.Series,
    tqqq_close: pd.Series,
    vix_close: pd.Series | None = None,
    vix3m_close: pd.Series | None = None,
    params: V3Parameters | None = None,
) -> pd.DataFrame:
    """Calculate close-known targets; callers execute them at the next open."""
    params = params or V3Parameters()
    qqq = qqq_close.astype(float)
    tqqq = tqqq_close.reindex(qqq.index).astype(float)
    momentum = qqq.pct_change(params.momentum_days)
    daily_vol = qqq.pct_change().rolling(params.volatility_days).std(ddof=1)
    normal_vol = daily_vol.rolling(
        params.volatility_median_days,
        min_periods=max(126, params.volatility_median_days // 4),
    ).median()
    relative_vol = daily_vol / normal_vol.replace(0.0, np.nan)
    if params.use_relative_volatility:
        volatility_allocation = (
            (params.volatility_ceiling - relative_vol.clip(
                params.volatility_floor, params.volatility_ceiling
            ))
            / (params.volatility_ceiling - params.volatility_floor)
        ).clip(0.0, 1.0)
    else:
        volatility_allocation = pd.Series(1.0, index=qqq.index)

    positive_momentum = momentum > 0.0
    target = volatility_allocation.where(
        positive_momentum,
        volatility_allocation * params.negative_momentum_scale,
    )
    tqqq_rsi = rsi(tqqq, params.rsi_days)
    dip_buy = (~positive_momentum) & (tqqq_rsi < params.oversold_rsi)

    if params.use_vix_confirmation and vix_close is not None:
        vix = vix_close.reindex(qqq.index).ffill().astype(float)
        if vix3m_close is not None:
            term = vix / vix3m_close.reindex(qqq.index).ffill().astype(float)
            panic_confirmed = term >= 1.0
        else:
            panic_confirmed = vix >= vix.rolling(252, min_periods=126).quantile(0.80)
        dip_buy &= panic_confirmed.fillna(False)
    if params.use_reversal_confirmation:
        reversal = (tqqq.pct_change() > 0.0) & (tqqq.pct_change(3) > 0.0)
        dip_buy &= reversal.fillna(False)
    if params.use_dip_buy:
        target = target.where(~dip_buy, params.dip_buy_allocation)

    return pd.DataFrame(
        {
            "raw_target": target.clip(0.0, 1.0),
            "momentum": momentum,
            "relative_volatility": relative_vol,
            "volatility_allocation": volatility_allocation,
            "rsi": tqqq_rsi,
            "dip_buy": dip_buy,
            "regime": np.where(positive_momentum, "BULL", "BEAR"),
        },
        index=qqq.index,
    )


def executable_targets(raw: pd.Series, band: float, initial: float = 1.0) -> pd.Series:
    """Apply a causal drift band to the desired allocation."""
    values = np.full(len(raw), np.nan)
    current = initial
    for offset, desired in enumerate(raw.to_numpy(dtype=float)):
        if np.isfinite(desired) and abs(desired - current) > band:
            current = float(desired)
        values[offset] = current
    return pd.Series(values, index=raw.index, name="target")


def signal_frame(
    qqq_close: pd.Series,
    tqqq_close: pd.Series,
    vix_close: pd.Series | None = None,
    vix3m_close: pd.Series | None = None,
    params: V3Parameters | None = None,
) -> pd.DataFrame:
    params = params or V3Parameters()
    result = raw_targets(qqq_close, tqqq_close, vix_close, vix3m_close, params)
    result["target"] = executable_targets(result["raw_target"], params.rebalance_band)
    return result
