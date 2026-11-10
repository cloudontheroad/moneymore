"""Test higher-low retests without moving the parent reduction dates."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from review_tqqq_external import daily_prices
from review_tqqq_recovery import defense_windows
from review_tqqq_swing import COSTS, START, assess, replay
from review_tqqq_swing_reentry import log_increments, opening_weights, relative_effect
from review_tqqq_switch_cost import leverage_rules

WAIT = 10
HORIZON = 20


def recovery_features(history):
    close = history.qqqClose
    prior = close.shift()
    true_range = pd.concat([
        history.qqqHigh - history.qqqLow,
        (history.qqqHigh - prior).abs(),
        (history.qqqLow - prior).abs(),
    ], axis=1).max(axis=1)
    return pd.DataFrame({
        "close": prior,
        "breakout": close.gt(prior.rolling(10).max()).shift(1, fill_value=False),
        # Exclude the pullback bar from its own volatility threshold.
        "atr": true_range.rolling(20).mean().shift(2),
    })


def retest_flags(features, parent):
    """Price-only state based on closes observed before each tradable open."""
    close, atr = features.close.to_numpy(), features.atr.to_numpy()
    breakout = features.breakout.to_numpy()
    ready = np.zeros(len(parent), dtype=bool)
    confirmed = np.zeros(len(parent), dtype=bool)
    delayed = np.zeros(len(parent), dtype=bool)
    for sell, _, end in defense_windows(parent):
        floor = close[sell]
        peak, pullback, first_break = None, None, None
        stage = "search"
        for i in range(sell + 1, end):
            price = close[i]
            if first_break is None and breakout[i]:
                first_break = i
            if first_break is not None and i >= first_break + WAIT:
                delayed[i] = True
            if price <= floor or (stage == "ready" and price <= pullback):
                floor = price
                stage, peak, pullback = "search", None, None
                continue
            if stage == "search" and breakout[i]:
                stage, peak = "advance", price
            elif stage == "advance":
                peak = max(peak, price)
                if peak - price >= atr[i]:
                    stage, pullback = "pullback", price
            elif stage == "pullback":
                pullback = min(pullback, price)
                if price > peak:
                    stage = "ready"
                    confirmed[i] = True
            ready[i] = stage == "ready"
    return pd.DataFrame({"ready": ready, "confirmed": confirmed, "delayed": delayed}, index=parent.index)


def choose_paths(parent, features, flags):
    variants = {
        "early_breakout": (features.breakout, "early"),
        "early_retest": (flags.ready, "early"),
        "early_fixed_wait": (flags.delayed, "early"),
        "wait_retest_then_parent": (flags.ready, "bounded"),
        "retest_only": (flags.ready, "unbounded"),
    }
    windows = defense_windows(parent)
    for name, (eligible, mode) in variants.items():
        target = parent.copy()
        rows = []
        for sell, baseline, end in windows:
            if mode == "early":
                first, stop, fallback = sell + 1, baseline, baseline
            elif mode == "bounded":
                first, stop = baseline, min(baseline + WAIT, end)
                fallback = stop
            else:
                first, stop, fallback = sell + 1, end, end
            choices = np.flatnonzero(eligible.iloc[first:stop].to_numpy())
            entry = first + choices[0] if len(choices) else fallback
            target.iloc[sell:entry] = 0.0
            target.iloc[entry:end] = 1.0
            rows.append({
                "sell": str(parent.index[sell].date()),
                "parent_buy": str(parent.index[baseline].date()) if baseline < end else None,
                "buy": str(parent.index[entry].date()) if entry < end else None,
                "sessions_vs_parent": int(entry - baseline) if max(entry, baseline) < end else None,
                "changed": bool(entry != baseline),
            })
        cuts = np.flatnonzero(target.eq(0).to_numpy() & target.shift(fill_value=0).eq(1).to_numpy())
        if not set(cuts).issubset(sell for sell, _, _ in windows):
            raise AssertionError("A candidate moved a reduction date")
        yield name, target, rows


def matched_events(prices, features, flags, parent, bps):
    """Use a common end date for each first breakout and its waiting alternatives."""
    swap = (1 - bps / 10_000) / (1 + bps / 10_000)
    ratio = (prices.tqqqOpen / prices.qqqOpen).to_numpy()
    events = []
    for sell, baseline, next_sell in defense_windows(parent):
        breakouts = np.flatnonzero(features.breakout.iloc[sell + 1:baseline].to_numpy())
        if not len(breakouts):
            continue
        entry = sell + 1 + breakouts[0]
        end = entry + HORIZON
        if end >= len(prices) or end > next_sell:
            continue
        confirmed = np.flatnonzero(flags.confirmed.iloc[entry:end].to_numpy())
        retest = entry + confirmed[0] if len(confirmed) else end
        fixed = entry + WAIT

        def net(start):
            return float(ratio[end] / ratio[start] * swap**2 - 1) if start < end else 0.0

        events.append({
            "date": str(prices.index[entry].date()),
            "retest": str(prices.index[retest].date()) if retest < end else None,
            "breakout": net(entry), "fixed_wait": net(fixed), "higher_low": net(retest),
        })
    groups = {}
    for rule in ("breakout", "fixed_wait", "higher_low"):
        values = [row[rule] for row in events]
        groups[rule] = {
            "mean_net_relative": float(np.mean(values)) if values else None,
            "wins": sum(value > 0 for value in values), "losses": sum(value < 0 for value in values),
        }
    return {"events": events, "summary": groups}


def attribute_entries(rows, candidate, baseline):
    active = log_increments(candidate.nav / baseline.nav)
    attributed = []
    for i, row in enumerate(rows):
        start = active.index.get_loc(row["sell"])
        end = active.index.get_loc(rows[i + 1]["sell"]) if i + 1 < len(rows) else len(active)
        attributed.append({**row, "net_log_excess_vs_parent": float(active.iloc[start:end].sum())})
    values = [row["net_log_excess_vs_parent"] for row in attributed]
    np.testing.assert_allclose(sum(values), active.sum(), atol=1e-11)
    without_best = np.expm1(active.sum() - max([0.0, *values]))
    return attributed, float(without_best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        qqq, tqqq = pool.map(lambda symbol: daily_prices(symbol, args.end), ("QQQ", "TQQQ"))
    history = pd.concat([
        qqq.add_prefix("qqq"), tqqq[["Open", "Close"]].add_prefix("tqqq"),
    ], axis=1, join="inner")
    prices = history.loc[START:]
    features = recovery_features(history).loc[prices.index]
    parent = opening_weights(history.qqqClose)["early_exit"].loc[prices.index]
    flags = retest_flags(features, parent)
    variants = {name: (target, rows) for name, target, rows in choose_paths(parent, features, flags)}
    for cutoff in ("2018-12-24", "2020-03-16", "2022-10-13", "2025-04-09"):
        partial_features = recovery_features(history.loc[:cutoff]).loc[START:]
        partial_parent = parent.loc[:cutoff]
        partial_flags = retest_flags(partial_features, partial_parent)
        pd.testing.assert_frame_equal(partial_flags, flags.loc[:cutoff])
        for name, weights, _ in choose_paths(partial_parent, partial_features, partial_flags):
            pd.testing.assert_series_equal(weights, variants[name][0].loc[:cutoff])

    control = ((leverage_rules(history.qqqClose).buffer_1pct - 1) / 2).shift().loc[prices.index]
    results = {}
    for cost in COSTS:
        hold, _ = replay(prices, pd.Series(1.0, index=prices.index), cost)
        base, base_fills = replay(prices, parent, cost, alternative="qqq")
        buffer, buffer_fills = replay(prices, control, cost, alternative="qqq")
        group = {
            "hold": assess(hold, 1, hold, prices.tqqqOpen),
            "buffer": assess(buffer, buffer_fills, hold, prices.tqqqOpen),
            "parent": assess(base, base_fills, hold, prices.tqqqOpen),
        }
        for name, (target, rows) in variants.items():
            run, fills = replay(prices, target, cost, alternative="qqq")
            entries, without_best = attribute_entries(rows, run, base)
            group[name] = {
                **assess(run, fills, hold, prices.tqqqOpen),
                "vs_parent": relative_effect(run, base),
                "without_best_defense_episode_vs_parent": without_best,
                "entries": entries,
            }
        results[cost] = group
    print(json.dumps({
        "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "defense_episodes": len(defense_windows(parent)),
        "matched_20_session_study": matched_events(prices, features, flags, parent, 25),
        "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
