"""Test Kelly and risk-constrained Kelly only during TQQQ recovery."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from review_tqqq_full_defense import cash_episodes, rearm_after_missed_rally, round_trips
from review_tqqq_macro_defense import load_history, risk_metrics, signal_inputs
from review_tqqq_swing import COSTS, START, replay
from review_tqqq_swing_reentry import relative_effect

HORIZON = 21
RECOVERY_SESSIONS = 21
ALPHA = 0.75
BETA = 0.20
LAMBDA = np.log(BETA) / np.log(ALPHA)
BUCKETS = np.array([0.0, 0.5, 1.0])


def prior_market_state(prices):
    qqq = prices.qqqClose
    above_trend = qqq.gt(qqq.rolling(200).mean())
    vol = np.log(qqq).diff().rolling(20).std()
    high_vol = vol.gt(vol.rolling(252, min_periods=63).median())
    return (2 * above_trend.astype(int) + high_vol.astype(int)).shift(1, fill_value=0).to_numpy()


def scenario_payoffs(prices):
    opens = prices.tqqqOpen.to_numpy()
    closes = prices.tqqqClose.to_numpy()
    scenario_end = np.arange(HORIZON, len(prices))
    gross = np.full(len(prices), np.nan)
    low = np.full(len(prices), np.nan)
    gross[:-HORIZON] = opens[scenario_end] / opens[:-HORIZON]
    for i in range(len(prices) - HORIZON):
        # Only opening and closing prices, rather than an intraday loss guarantee.
        window_open = opens[i:i + HORIZON + 1]
        window_close = closes[i:i + HORIZON]
        low[i] = min(window_open.min(), window_close.min()) / opens[i]
    return gross, low


def scenario_weights(same):
    pooled = np.full(len(same), 1.0 / len(same))
    count = same.sum()
    shrink = count / (count + 12)
    weights = pooled * (1 - shrink)
    if count:
        weights[same] += shrink / count
    return weights


def recovery_schedule(prices, parent):
    weeks = prices.index.to_period("W-FRI")
    regimes = prior_market_state(prices)
    gross, low = scenario_payoffs(prices)

    recent_buy = None
    schedule = []
    for i, allocation in enumerate(parent.to_numpy()):
        if allocation == 0:
            recent_buy = None
            continue
        if i and parent.iloc[i - 1] == 0:
            recent_buy = i
        if recent_buy is None or i - recent_buy >= RECOVERY_SESSIONS:
            continue
        if i != recent_buy and weeks[i] == weeks[i - 1]:
            continue
        # Non-overlapping past 21-session blocks, all ending strictly before this open.
        known = np.arange(i - HORIZON - 1, -1, -HORIZON)
        if not len(known):
            continue
        same = regimes[known] == regimes[i]
        schedule.append((i, gross[known], low[known], scenario_weights(same), int(same.sum()), regimes[i]))
    return schedule


def kelly_allocation(gross, low, probabilities, bps, *, constrained=False, path_check=False, buckets=BUCKETS):
    spread = (1 - bps / 10_000) / (1 + bps / 10_000)
    log_payoff = np.log1p(np.outer(gross * spread - 1, buckets))
    growth = probabilities @ log_payoff
    risk_moment = probabilities @ np.exp(-LAMBDA * log_payoff)
    trough = 1 + np.outer(low * spread - 1, buckets)
    path_probability = probabilities @ (trough < ALPHA)
    permitted = np.ones(len(buckets), dtype=bool)
    if constrained:
        permitted &= risk_moment <= 1 + 1e-12
    if path_check:
        permitted &= path_probability <= BETA + 1e-12
    selected = int(np.argmax(np.where(permitted, growth, -np.inf)))
    return float(buckets[selected]), float(growth[selected]), float(risk_moment[selected]), float(path_probability[selected])


def transition_targets(parent, schedule, bps):
    modes = {
        "fixed_half": {},
        "plain_kelly": {},
        "risk_constrained": {"constrained": True},
        "risk_constrained_binary": {"constrained": True, "buckets": np.array([0.0, 1.0])},
        "risk_and_path": {"constrained": True, "path_check": True},
    }
    for name, options in modes.items():
        decisions = {}
        ledger = []
        for i, gross, low, probability, n, regime in schedule:
            if name == "fixed_half":
                decisions[i] = 0.5
                continue
            weight, growth, moment, path_risk = kelly_allocation(gross, low, probability, bps, **options)
            decisions[i] = weight
            ledger.append({
                "date": str(parent.index[i].date()), "weight": weight,
                "scenarios": len(gross), "same_regime": n, "regime": int(regime),
                "expected_log_growth": growth, "risk_moment": moment, "path_floor_frequency": path_risk,
            })
        path = parent.copy()
        recent_buy, active = None, 1.0
        for i, target in enumerate(parent.to_numpy()):
            if target == 0:
                recent_buy = None
            elif i and parent.iloc[i - 1] == 0:
                recent_buy, active = i, 1.0
            if recent_buy is not None and i - recent_buy < RECOVERY_SESSIONS:
                active = decisions.get(i, active)
                path.iloc[i] = active
        yield name, path, ledger


def baseline_weights(history, flags=None):
    if flags is None:
        flags = signal_inputs(history)
    close = history.qqqClose
    normal = flags.slow_up & flags.fast_up
    quick = flags.slow_up & close.gt(close.shift().rolling(10).max())
    reset = close.gt(close.shift().rolling(63).max())
    return rearm_after_missed_rally(flags, close, normal, quick, reset)


def acute_gate(history, bps, *, constrained):
    """Test whether weak acute signals are worth giving up a full TQQQ position."""
    flags = signal_inputs(history)
    gross, low = scenario_payoffs(history)
    regimes = prior_market_state(history)
    soft = flags.sharp_shock & ~flags.slow_down & ~flags.vol_stress
    anchors = []
    decisions = []
    binary = np.array([0.0, 1.0])
    cash_payoff = (1 - bps / 10_000) / (1 + bps / 10_000)
    for i in np.flatnonzero(soft.to_numpy()):
        # The decision follows close i and trades at open i+1.
        opening = i + 1
        if opening == len(history):
            break
        recent = np.asarray([j for j in anchors if j + HORIZON < opening], dtype=int)
        pooled = np.arange(opening - HORIZON - 1, -1, -HORIZON)
        if len(pooled):
            # Remove overlapping pooled scenarios before appending acute analogs.
            if len(recent):
                pooled = pooled[(np.abs(pooled[:, None] - recent) >= HORIZON).all(axis=1)]
            known = np.r_[pooled, recent]
            comparable = np.isin(known, recent) & (regimes[known] == regimes[opening])
            if len(known):
                # Holding already-open shares incurs no incremental cost. Selling and
                # rebuying incurs two legs. Normalize payoffs to that cash alternative.
                weight, growth, moment, path_risk = kelly_allocation(
                    gross[known] / cash_payoff, low[known] / cash_payoff,
                    scenario_weights(comparable), 0, constrained=constrained, buckets=binary,
                )
                flags.iat[i, flags.columns.get_loc("sharp_shock")] = weight == 0
                decisions.append({
                    "signal_date": str(history.index[i].date()), "keep_full": weight == 1,
                    "scenarios": len(known), "same_regime_acute": int(comparable.sum()),
                    "expected_log_growth_vs_cash": growth, "risk_moment": moment,
                    "path_floor_frequency": path_risk,
                })
        if not anchors or opening - anchors[-1] >= HORIZON:
            anchors.append(opening)
    return baseline_weights(history, flags), decisions


def verify_paths(history, parent, full):
    for cutoff in ("2015-08-24", "2016-06-27", "2020-03-16", "2022-10-13", "2025-04-09"):
        past = history.loc[:cutoff]
        known = baseline_weights(past)
        pd.testing.assert_series_equal(known, parent.loc[:cutoff])
        for name, weights, _ in transition_targets(known, recovery_schedule(past, known), 25):
            pd.testing.assert_series_equal(weights, full[name].loc[:cutoff])
        for constrained in (False, True):
            name = "acute_risk_constrained" if constrained else "acute_plain_kelly"
            pd.testing.assert_series_equal(acute_gate(past, 25, constrained=constrained)[0], full[name].loc[:cutoff])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    history = load_history(args.end)
    parent = baseline_weights(history)
    schedule = recovery_schedule(history, parent)
    check_paths = {name: weights for name, weights, _ in transition_targets(parent, schedule, 25)}
    check_paths["acute_plain_kelly"] = acute_gate(history, 25, constrained=False)[0]
    check_paths["acute_risk_constrained"] = acute_gate(history, 25, constrained=True)[0]
    verify_paths(history, parent, check_paths)
    prices = history.loc[START:]
    results = {}
    for bps, delay in [*((bps, 0) for bps in COSTS), (25, 1)]:
        hold_weights = pd.Series(1.0, index=history.index)
        hold, _ = replay(prices, hold_weights.loc[prices.index], bps)
        base, _ = replay(prices, parent.shift(delay, fill_value=1).loc[prices.index], bps)
        group = {}
        variants = [("hold", hold_weights, []), ("prior_defense", parent, []),
                    *transition_targets(parent, schedule, bps),
                    ("acute_plain_kelly", *acute_gate(history, bps, constrained=False)),
                    ("acute_risk_constrained", *acute_gate(history, bps, constrained=True))]
        for name, weights, decisions in variants:
            target = weights.shift(delay, fill_value=1).loc[prices.index]
            frame, fills = replay(prices, target, bps)
            metrics = risk_metrics(frame, target, fills, prices, delay)
            in_transition = parent.eq(1) & weights.ne(1)
            group[name] = {
                **metrics,
                "partial_target_days": int(target.between(0, 1, inclusive="neither").sum()),
                "days_below_full_during_parent_risk_on": int(in_transition.loc[START:].sum()),
                "to_cash_transitions": int((target.eq(0) & target.shift(fill_value=1).gt(0)).sum()),
                "vs_prior_defense": relative_effect(frame, base),
                "vs_hold": relative_effect(frame, hold),
                "decisions": [row for row in decisions if (row.get("date") or row["signal_date"]) >= START],
            }
            if target.isin((0, 1)).all():
                group[name]["cash_ledger"] = cash_episodes(target, prices, bps)
                group[name]["round_trip_ledger"] = round_trips(target, prices, bps)
        results[f"{bps}bps_delay{delay}"] = group
    print(json.dumps({
        "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "parameters": {"floor": ALPHA, "frequency": BETA, "lambda": LAMBDA, "scenario_sessions": HORIZON},
        "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
