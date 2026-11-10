"""Separate the effects of earlier exits and phased re-entry on a frozen swing strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from review_tqqq_execution import bootstrap
from review_tqqq_external import daily_prices
from review_tqqq_swing import COSTS, START, WAVES, assess, changepoint_weights, replay
from review_tqqq_switch_cost import leverage_rules

POLICIES = {
    "parent": (False, False, False),
    "early_exit": (True, False, False),
    "phased_reentry": (False, True, False),
    "both": (True, True, False),
    "both_require_confirmation": (True, True, True),
}


def opening_weights(close):
    parent = changepoint_weights(close, slow_fallback=True)
    previous = close.shift()

    def crosses(length, *, above):
        channel = close.shift(2).rolling(length)
        return previous.gt(channel.max()) if above else previous.lt(channel.min())

    early_break = crosses(63, above=False)
    full_recovery = crosses(63, above=True)
    partial_recovery = crosses(20, above=True)
    failed_recovery = crosses(20, above=False)
    targets = {}
    calendar_week = pd.Series(close.index.to_period("W-FRI"), index=close.index)
    new_week = calendar_week.ne(calendar_week.shift())
    for name, (early, phased, require_confirmation) in POLICIES.items():
        allocation, locked = 1.0, False
        prior_week_on = False
        path = []
        for risk_on, sell, restore, partial, fail, week in zip(
            parent, early_break, full_recovery, partial_recovery, failed_recovery, new_week, strict=True,
        ):
            if not risk_on:
                locked = False
            elif early and sell:
                locked = True
            elif restore:
                locked = False
            if risk_on and not locked:
                if require_confirmation and allocation < 1.0:
                    confirmed = allocation == 0.5 and (restore or (week and prior_week_on))
                    allocation = 1.0 if confirmed else 0.5
                else:
                    allocation = 1.0
            elif allocation == 1 or fail or not phased:
                allocation = 0.0
            elif partial:
                allocation = 0.5
            if week:
                prior_week_on = risk_on and not locked
            path.append(allocation)
        targets[name] = pd.Series(path, index=close.index)
    pd.testing.assert_series_equal(parent, targets["parent"])
    return targets


def log_increments(nav):
    values = np.log(nav)
    return values.diff().fillna(values.iloc[0])


def relative_effect(candidate, baseline):
    active = log_increments(candidate.nav / baseline.nav)
    waves = {name: float(active.loc[start:end].sum()) for name, (start, end) in WAVES.items()}
    return {
        "annual_log_excess": float(active.mean() * 252),
        "net_relative_return": float(np.expm1(active.sum())),
        "annual_log_excess_interval": bootstrap([active.to_numpy()]),
        "windows_log_excess": waves,
        "outside_windows_log_excess": float(active.sum() - sum(waves.values())),
        "net_relative_without_best_wave": float(np.expm1(active.sum() - max(0, max(waves.values())))),
    }


def tranche_ledger(frame, weights, tqqq_open):
    ledger = []
    changes = weights.diff().fillna(weights.iloc[0] - 1)
    for i in np.flatnonzero(changes.to_numpy() > 0):
        day = weights.index[i]
        prior = weights.iloc[i - 1] if i else 1.0
        ledger.append({
            "date": str(day.date()),
            "target_from": float(prior), "target_to": float(weights.iloc[i]),
            "tqqq_open": float(tqqq_open.iloc[i]),
            "trade_weight_delta": float(frame.weight_change.iloc[i]),
        })
    return ledger


def trial_entries(weights, prices, alternative):
    """Evaluate the market move while holding the intermediate 50% target."""
    target = weights.to_numpy()
    trials = []
    for start in np.flatnonzero((target == 0.5) & ~weights.shift().eq(0.5).to_numpy()):
        changes = np.flatnonzero(target[start + 1:] != 0.5)
        if not len(changes):
            continue
        end = start + 1 + changes[0]
        entry, exit_ = prices.iloc[start], prices.iloc[end]
        tqqq = exit_.tqqqOpen / entry.tqqqOpen
        residual = exit_.qqqOpen / entry.qqqOpen if alternative == "qqq" else 1.0
        trials.append({
            "start": str(weights.index[start].date()), "end": str(weights.index[end].date()),
            "next_target": float(target[end]),
            "gross_relative_vs_no_tqqq": float((0.5 * tqqq + 0.5 * residual) / residual - 1),
        })
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        qqq, tqqq = pool.map(lambda symbol: daily_prices(symbol, args.end), ("QQQ", "TQQQ"))
    history = pd.concat([
        qqq[["Open", "Close"]].add_prefix("qqq"), tqqq[["Open", "Close"]].add_prefix("tqqq"),
    ], axis=1, join="inner")
    close = history.qqqClose
    targets = opening_weights(close)
    for cutoff in ("2018-12-24", "2020-03-16", "2022-10-13", "2025-04-09"):
        for name, truncated in opening_weights(close.loc[:cutoff]).items():
            pd.testing.assert_series_equal(truncated, targets[name].loc[:cutoff])
    prices = history.loc[START:]
    targets = {name: weights.loc[prices.index] for name, weights in targets.items()}
    results = {}
    for cost in COSTS:
        hold, _ = replay(prices, pd.Series(1.0, index=prices.index), cost)
        buffer_target = ((leverage_rules(close).buffer_1pct - 1) / 2).shift().loc[prices.index]
        buffer, fills = replay(prices, buffer_target, cost, alternative="qqq")
        output = {
            "hold": assess(hold, 1, hold, prices.tqqqOpen),
            "buffer": assess(buffer, fills, hold, prices.tqqqOpen),
        }
        for alternative in ("cash", "qqq"):
            runs = {name: replay(prices, weights, cost, alternative=alternative)
                    for name, weights in targets.items()}
            output[alternative] = {
                name: assess(path, fills, hold, prices.tqqqOpen)
                for name, (path, fills) in runs.items()
            }
            output[alternative]["paired_effects"] = {
                "early_alone": relative_effect(runs["early_exit"][0], runs["parent"][0]),
                "phased_alone": relative_effect(runs["phased_reentry"][0], runs["parent"][0]),
                "early_with_phased": relative_effect(runs["both"][0], runs["phased_reentry"][0]),
                "phased_with_early": relative_effect(runs["both"][0], runs["early_exit"][0]),
                "require_confirmation": relative_effect(runs["both_require_confirmation"][0], runs["both"][0]),
            }
            if cost == 25:
                output[alternative]["reentry_ledger"] = {
                    name: tranche_ledger(path, targets[name], prices.tqqqOpen) for name, (path, _) in runs.items()
                }
                output[alternative]["trial_entries"] = {
                    name: trial_entries(targets[name], prices, alternative)
                    for name in ("phased_reentry", "both", "both_require_confirmation")
                }
        results[cost] = output
    print(json.dumps({
        "start": START, "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(), "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
