"""Keep normal TQQQ exposure at 100% and test recovery after repeat losses."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from review_tqqq_credit import adjusted_close
from review_tqqq_macro_defense import cash_state, load_history, risk_metrics, signal_inputs
from review_tqqq_swing import COSTS, START, replay

FAILURE_WINDOW = 20
CHOP_START = "2014-09-01"
CHOP_END = "2016-12-31"


def failure_aware(exit_, standard, durable):
    """Require a quarterly breakout only after a recent re-entry fails."""
    invested, stricter, last_entry = True, False, None
    targets = []
    for i, (reduce, recover, stronger) in enumerate(zip(exit_, standard, durable, strict=True)):
        if reduce and invested:
            stricter = last_entry is not None and i - last_entry <= FAILURE_WINDOW
            invested = False
        elif not reduce and not invested and (stronger if stricter else recover):
            invested, last_entry = True, i
        targets.append(float(invested))
    return pd.Series(targets, index=exit_.index).shift(1, fill_value=1.0)


def reclaim_exit_reference(exit_, entry, qqq):
    """When flat, also require QQQ to reclaim the close before the first sell signal."""
    invested, reference = True, None
    targets = []
    for reduce, recover, price, prior in zip(exit_, entry, qqq, qqq.shift(), strict=True):
        if reduce and invested:
            invested, reference = False, prior
        elif not reduce and not invested and recover and price >= reference:
            invested = True
        targets.append(float(invested))
    return pd.Series(targets, index=exit_.index).shift(1, fill_value=1.0)


def separate_recoveries(exit_, slow_down, standard, quick):
    """Re-enter acute-only pullbacks sooner. Require the standard rule after a slow break."""
    invested, slow_broken = True, False
    targets = []
    for reduce, weak, recover, rebound in zip(exit_, slow_down, standard, quick, strict=True):
        if reduce and invested:
            invested, slow_broken = False, weak
        elif not invested:
            slow_broken = slow_broken or weak
            if not reduce and (recover or (rebound and not slow_broken)):
                invested = True
        targets.append(float(invested))
    return pd.Series(targets, index=exit_.index).shift(1, fill_value=1.0)


def rearm_after_missed_rally(flags, qqq, recover, quick, new_quarter_high):
    """After an acute false exit, require stress confirmation until a new quarter high."""
    invested, armed, slow_broken, sale_close = True, True, False, 0.0
    targets = []
    inputs = zip(flags.slow_down, flags.sharp_shock, flags.vol_stress, qqq,
                 recover, quick, new_quarter_high, strict=True)
    for weak, shock, stressed, price, normal_buy, quick_buy, reset in inputs:
        armed = armed or reset
        if invested:
            if weak or (shock and (armed or stressed)):
                invested, slow_broken, sale_close = False, weak, price
        else:
            slow_broken = slow_broken or weak
            if not (weak or shock) and (normal_buy or (quick_buy and not slow_broken)):
                invested = True
                if not slow_broken and price > sale_close:
                    armed = reset
        targets.append(float(invested))
    return pd.Series(targets, index=qqq.index).shift(1, fill_value=1.0)


def opening_weights(history):
    flags = signal_inputs(history)
    exit_ = flags.slow_down | flags.sharp_shock
    recover = flags.slow_up & flags.fast_up
    qqq = history.qqqClose
    new_quarter_high = qqq.gt(qqq.shift().rolling(63).max())
    quarterly = flags.slow_up & new_quarter_high
    sustained = recover & flags.slow_up.rolling(5).sum().eq(5)
    month_low = qqq.lt(qqq.shift().rolling(20).min())
    broad_low = history.SPY.lt(history.SPY.shift().rolling(20).min())
    quick = flags.slow_up & qqq.gt(qqq.shift().rolling(10).max())
    return {
        "hold": pd.Series(1.0, index=history.index),
        "slow_control": cash_state(flags.slow_down, flags.slow_up),
        "base": cash_state(exit_, recover),
        "five_day_stability": cash_state(exit_, sustained),
        "always_quarterly": cash_state(exit_, quarterly),
        "failed_then_quarterly": failure_aware(exit_, recover, quarterly),
        "reclaim_exit_reference": reclaim_exit_reference(exit_, recover, qqq),
        "shock_with_qqq_break": cash_state(flags.slow_down | (flags.sharp_shock & month_low), recover),
        "shock_with_spy_break": cash_state(flags.slow_down | (flags.sharp_shock & broad_low), recover),
        "faster_recovery": cash_state(exit_, quick),
        "acute_only_faster_recovery": separate_recoveries(exit_, flags.slow_down, recover, quick),
        "rearm_after_missed_rally": rearm_after_missed_rally(flags, qqq, recover, quick, new_quarter_high),
    }


def round_trips(target, prices, bps):
    change = target.diff().fillna(0)
    reductions = np.flatnonzero(change.lt(0).to_numpy())
    recoveries = np.flatnonzero(change.gt(0).to_numpy())
    ledger = []
    friction = (1 - bps / 10_000) / (1 + bps / 10_000)
    for buy in recoveries:
        later = reductions[reductions > buy]
        if not len(later):
            continue
        sell = later[0]
        value = prices.tqqqOpen.iloc[sell] / prices.tqqqOpen.iloc[buy] * friction - 1
        ledger.append({
            "buy": str(prices.index[buy].date()), "sell": str(prices.index[sell].date()),
            "sessions": int(sell - buy), "net_return": float(value),
            "failed_within_one_month": bool(sell - buy <= FAILURE_WINDOW and value < 0),
        })
    return ledger


def cash_episodes(target, prices, bps):
    change = target.diff().fillna(0)
    reductions = np.flatnonzero(change.lt(0).to_numpy())
    recoveries = np.flatnonzero(change.gt(0).to_numpy())
    ledger = []
    friction = (1 - bps / 10_000) / (1 + bps / 10_000)
    for sell in reductions:
        later = recoveries[recoveries > sell]
        if not len(later):
            continue
        buy = later[0]
        gross = prices.tqqqOpen.iloc[buy] / prices.tqqqOpen.iloc[sell]
        ledger.append({
            "sell": str(prices.index[sell].date()), "buy": str(prices.index[buy].date()),
            "cash_sessions": int(buy - sell), "tqqq_while_flat": float(gross - 1),
            "net_relative_vs_staying_in": float(friction / gross - 1),
        })
    return ledger


def chop_metrics(nav, ledger):
    window = nav.loc[CHOP_START:CHOP_END]
    prior = nav.iloc[nav.index.get_loc(window.index[0]) - 1]
    path = np.r_[prior, window.to_numpy()]
    return {
        "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
        "return": float(path[-1] / path[0] - 1),
        "failed_reentries": sum(CHOP_START <= row["sell"] <= CHOP_END and row["failed_within_one_month"]
                                for row in ledger),
    }


def first_recoveries(target):
    changes = target.diff().gt(0)
    return {
        label: [str(day.date()) for day in changes.loc[start:end].index[changes.loc[start:end]]]
        for label, start, end in (
            ("2015–2016", CHOP_START, CHOP_END),
            ("2020", "2020-02-01", "2020-06-30"),
            ("2022", "2022-01-01", "2023-04-30"),
            ("2025", "2025-02-01", "2025-06-30"),
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        market = pool.submit(load_history, args.end)
        broad = pool.submit(adjusted_close, "SPY", args.end)
        history = market.result()
        history["SPY"] = broad.result()
    if history.loc[START:, "SPY"].isna().any():
        raise ValueError("SPY must cover every common trading session")
    weights = opening_weights(history)
    for cutoff in ("2015-08-24", "2016-06-27", "2020-03-16", "2022-10-13", "2025-04-09"):
        for name, target in opening_weights(history.loc[:cutoff]).items():
            pd.testing.assert_series_equal(target, weights[name].loc[:cutoff])
    prices = history.loc[START:]
    results = {}
    for cost, delay in [*((cost, 0) for cost in COSTS), (25, 1)]:
        group = {}
        for name, targets in weights.items():
            target = targets.shift(delay, fill_value=1.0).loc[prices.index]
            frame, fills = replay(prices, target, cost)
            ledger = round_trips(target, prices, cost)
            cash_ledger = cash_episodes(target, prices, cost)
            chop_exits = [row for row in cash_ledger if CHOP_START <= row["sell"] <= CHOP_END]
            failed = [row for row in ledger if row["failed_within_one_month"]]
            group[name] = {
                **risk_metrics(frame, target, fills, prices, delay),
                "chop_2014_to_2016": chop_metrics(frame.nav, ledger),
                "failed_reentries": len(failed),
                "failed_reentries_combined_loss": float(np.prod([1 + row["net_return"] for row in failed]) - 1),
                "reentries_by_wave": first_recoveries(target),
                "round_trip_ledger": ledger,
                "cash_ledger": cash_ledger,
                "chop_cash_exits": len(chop_exits),
                "chop_cash_exits_hurt": sum(row["net_relative_vs_staying_in"] < 0 for row in chop_exits),
            }
        results[f"{cost}bps_delay{delay}"] = group
    print(json.dumps({
        "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
