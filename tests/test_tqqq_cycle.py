from __future__ import annotations

import numpy as np
import pandas as pd

from moneymore.tqqq_cycle import CycleParameters, cycle_decisions, latest_cycle_decision


def market(periods: int = 800):
    dates = pd.bdate_range("2020-01-02", periods=periods)
    qqq = pd.Series(300 * np.cumprod(np.full(periods, 1.0005)), index=dates)
    vix = pd.Series(np.linspace(12, 32, periods), index=dates)
    vix3m = pd.Series(np.full(periods, 20.0), index=dates)
    rate = pd.Series(np.full(periods, 4.0), index=dates)
    return qqq, vix, rate, vix3m


def test_fear_is_contrarian_not_an_automatic_sell_signal():
    qqq, vix, rate, vix3m = market()
    vix.iloc[:-80] = 16.0
    vix.iloc[-80:] = np.linspace(16.0, 40.0, 80)
    decisions = cycle_decisions(qqq, vix, rate, vix3m)

    assert decisions.iloc[-1].fear_rank > decisions.iloc[-100].fear_rank
    assert decisions.iloc[-1].raw_target > decisions.iloc[-100].raw_target


def test_rising_rates_reduce_the_cycle_target():
    qqq, vix, rate, vix3m = market()
    stable = latest_cycle_decision(qqq, vix, rate, vix3m)
    rising_rate = rate.copy()
    rising_rate.iloc[-120:] = np.linspace(4.0, 7.0, 120)
    pressured = latest_cycle_decision(qqq, vix, rising_rate, vix3m)

    assert pressured.rate_rank < stable.rate_rank
    assert pressured.raw_target < stable.raw_target


def test_only_structural_break_activates_crash_guard():
    qqq, vix, rate, vix3m = market()
    qqq.iloc[-1] = qqq.rolling(250).mean().iloc[-2] * 0.70
    decision = latest_cycle_decision(qqq, vix, rate, vix3m)

    assert decision.crash_guard_active
    assert decision.regime == "STRUCTURAL_CRASH"


def test_future_values_cannot_rewrite_an_old_target():
    qqq, vix, rate, vix3m = market()
    original = cycle_decisions(qqq, vix, rate, vix3m).iloc[-1]
    future_dates = pd.bdate_range(qqq.index[-1] + pd.Timedelta(1, unit="D"), periods=20)
    extended = cycle_decisions(
        pd.concat([qqq, pd.Series(qqq.iloc[-1] * 0.5, index=future_dates)]),
        pd.concat([vix, pd.Series(80.0, index=future_dates)]),
        pd.concat([rate, pd.Series(10.0, index=future_dates)]),
        pd.concat([vix3m, pd.Series(30.0, index=future_dates)]),
    ).loc[qqq.index[-1]]

    assert extended.raw_target == original.raw_target
    assert extended.combined_rank == original.combined_rank


def test_parameter_object_exposes_research_contract():
    params = CycleParameters()
    assert params.rank_window == 1260
    assert params.rebalance_band == 0.12
    assert params.rebalance_frequency == "monthly"
    assert params.crash_ma_days == 250


def test_live_state_changes_only_at_month_open():
    qqq, vix, rate, vix3m = market()
    midmonth = latest_cycle_decision(
        qqq.iloc[:-3], vix.iloc[:-3], rate.iloc[:-3], vix3m.iloc[:-3], previous_target=0.80
    )
    assert midmonth.target_tqqq == 0.80

    next_month = qqq.index[-1] + pd.offsets.MonthBegin(1)
    qqq2 = pd.concat([qqq, pd.Series(qqq.iloc[-1], index=[next_month])])
    vix2 = pd.concat([vix, pd.Series(vix.iloc[-1], index=[next_month])])
    rate2 = pd.concat([rate, pd.Series(7.0, index=[next_month])])
    vix3m2 = pd.concat([vix3m, pd.Series(vix3m.iloc[-1], index=[next_month])])
    month_open = latest_cycle_decision(
        qqq2, vix2, rate2, vix3m2, previous_target=0.80
    )
    assert month_open.target_tqqq != 0.80
