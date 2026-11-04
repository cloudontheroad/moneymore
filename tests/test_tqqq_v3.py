from __future__ import annotations

import numpy as np
import pandas as pd

from moneymore.tqqq_v3 import V3Parameters, executable_targets, raw_targets


def prices(periods: int = 800):
    dates = pd.bdate_range("2020-01-02", periods=periods)
    qqq = pd.Series(300 * np.cumprod(np.full(periods, 1.0007)), index=dates)
    tqqq = pd.Series(50 * np.cumprod(np.full(periods, 1.0021)), index=dates)
    return qqq, tqqq


def test_positive_momentum_keeps_full_exposure_when_volatility_is_normal():
    qqq, tqqq = prices()
    result = raw_targets(qqq, tqqq, params=V3Parameters(use_relative_volatility=False))
    assert result.iloc[-1].regime == "BULL"
    assert result.iloc[-1].raw_target > 0.95


def test_negative_momentum_softly_derisks_instead_of_forcing_cash():
    qqq, tqqq = prices()
    qqq.iloc[-40:] = qqq.iloc[-40] * np.cumprod(np.full(40, 0.998))
    tqqq.iloc[-40:] = tqqq.iloc[-40] * np.cumprod(np.full(40, 0.994))
    result = raw_targets(
        qqq,
        tqqq,
        params=V3Parameters(use_dip_buy=False, use_relative_volatility=False),
    )
    assert result.iloc[-1].regime == "BEAR"
    assert 0.0 < result.iloc[-1].raw_target <= 0.30


def test_oversold_bear_raises_allocation_for_dip_buy():
    qqq, tqqq = prices()
    qqq.iloc[-30:] = qqq.iloc[-30] * np.cumprod(np.full(30, 0.997))
    tqqq.iloc[-30:] = tqqq.iloc[-30] * np.cumprod(np.full(30, 0.985))
    with_dip = raw_targets(qqq, tqqq)
    without = raw_targets(qqq, tqqq, params=V3Parameters(use_dip_buy=False))
    assert with_dip.iloc[-1].dip_buy
    assert with_dip.iloc[-1].raw_target == 0.80
    assert with_dip.iloc[-1].raw_target > without.iloc[-1].raw_target


def test_rebalance_band_prevents_noise_trades():
    dates = pd.bdate_range("2026-01-01", periods=5)
    raw = pd.Series([1.0, 0.95, 0.91, 0.89, 0.88], index=dates)
    target = executable_targets(raw, band=0.10)
    assert list(target) == [1.0, 1.0, 1.0, 0.89, 0.89]


def test_future_prices_do_not_rewrite_old_targets():
    qqq, tqqq = prices()
    old = raw_targets(qqq, tqqq).iloc[-1]
    future = pd.bdate_range(qqq.index[-1] + pd.Timedelta(1, unit="D"), periods=30)
    extended = raw_targets(
        pd.concat([qqq, pd.Series(qqq.iloc[-1] * 0.5, index=future)]),
        pd.concat([tqqq, pd.Series(tqqq.iloc[-1] * 0.2, index=future)]),
    ).loc[qqq.index[-1]]
    assert extended.raw_target == old.raw_target
    assert extended.relative_volatility == old.relative_volatility
