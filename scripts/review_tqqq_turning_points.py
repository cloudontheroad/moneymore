"""Test early TQQQ exits and recoveries with walk-forward net trade hurdles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_kelly_forecast import standardize
from review_tqqq_kelly_rebuild import matched_quarterly_weight
from review_tqqq_macro_defense import load_history
from review_tqqq_sharpe import daily_returns, paired_interval, sharpe
from review_tqqq_swing import START, replay
from review_tqqq_swing_round_trips import open_signal, path_metrics, state_targets


def forecasts(history):
    """Predict next-week TQQQ log return using only previously settled weeks."""
    qqq = np.log(history.qqqClose)
    credit = np.log(history.HYG / history.IEI)
    vol = qqq.diff().rolling(63).std()
    price = {
        "fast": qqq.diff(5) / (vol * np.sqrt(5)),
        "medium": qqq.diff(21) / (vol * np.sqrt(21)),
        "slow": qqq.diff(63) / (vol * np.sqrt(63)),
    }
    cross = {
        "credit": credit.diff(5) / (credit.diff().rolling(63).std() * np.sqrt(5)),
        "vix_curve": np.log(history.VIX / history.VIX3M),
    }
    week = np.flatnonzero(~history.index.to_period("W-FRI").duplicated())
    week = week[week > 0]
    opens = history.tqqqOpen.to_numpy()
    returns = np.diff(np.log(opens[week]))
    output = pd.DataFrame(index=history.index[week], columns=["price", "cross", "tail"], dtype=float)
    for name, predictors in (("price", price), ("cross", {**price, **cross})):
        features = pd.DataFrame(predictors).iloc[week - 1].to_numpy()
        valid = np.isfinite(features).all(axis=1)
        for i in np.flatnonzero(valid):
            past = np.flatnonzero(valid[:max(0, i - 1)])
            # A weekly label ending at today's open is unavailable at the signal close.
            if len(past) < 104:
                continue
            x, query = standardize(features[past], features[i])
            design = np.column_stack((np.ones(len(x)), x))
            penalty = np.diag([0.0, *([26.0] * x.shape[1])])
            fit = np.linalg.solve(design.T @ design + penalty, design.T @ returns[past])
            output.iat[i, output.columns.get_loc(name)] = np.r_[1, query] @ fit
            if name == "cross":
                # Separate downside-frequency forecast. The tail threshold uses
                # only returns already available at the forecast date.
                realized = returns[past]
                tail = realized <= np.quantile(realized, 0.20)
                tail_fit = np.linalg.solve(design.T @ design + penalty, design.T @ tail)
                output.iat[i, output.columns.get_loc("tail")] = np.clip(np.r_[1, query] @ tail_fit, 0, 1)
    return output


def price_signals(history):
    close = history.qqqClose
    average = close.rolling(175).mean()
    trend = state_targets(open_signal(close.gt(average)), open_signal(close.lt(0.99 * average)))
    # Weekly attempts distinguish a short rebound from weak medium-term momentum.
    weak = close.lt(close.rolling(50).mean()) & close.lt(close.shift(21))
    rebound = close.gt(close.shift(5))
    return pd.DataFrame({"trend": trend, "weak": open_signal(weak), "rebound": open_signal(rebound)})


def elevated_tail(score):
    return score.gt(score.shift().rolling(104, min_periods=52).quantile(0.80))


def overlay(signals, exit_ok, recovery_ok):
    cash = False
    rows = []
    for date, (trend, weak, rebound) in zip(signals.index, signals.to_numpy(), strict=True):
        if not trend:
            cash = False
        elif date in exit_ok.index:
            if cash:
                cash = not (rebound and recovery_ok.at[date])
            elif weak:
                cash = exit_ok.at[date]
        rows.append(float(bool(trend) and not cash))
    return pd.Series(rows, index=signals.index)


def opportunities(history, signals, expected, raw, high_tail, bps):
    """Score both decisions on the same raw path, using one completed week per event."""
    weekly = signals.reindex(expected.index)
    previous = raw.shift().reindex(expected.index)
    gross = history.tqqqOpen.reindex(expected.index).shift(-1) / history.tqqqOpen.reindex(expected.index)
    market_log = np.log(gross)
    observed_tail = market_log.le(market_log.shift(2).expanding(min_periods=52).quantile(0.20))
    fee = bps / 10_000
    hurdle = np.log((1 + fee) / (1 - fee))
    result = {}
    for side, active, direction in (
        ("exit", weekly.trend.eq(1) & previous.eq(1) & weekly.weak, -1),
        ("recovery", weekly.trend.eq(1) & previous.eq(0) & weekly.rebound, 1),
    ):
        # Recovery samples are actual early-overlay buy signals; trend restarts after
        # being in long-term cash are excluded because the overlay cannot delay them.
        if side == "recovery":
            active &= signals.trend.shift().reindex(expected.index).eq(1)
        sample = active & gross.notna() & expected.notna().all(axis=1) & (expected.index >= pd.Timestamp(START))
        net = direction * market_log[sample] - hurdle
        result[side] = {}
        selections = (
            ("all", pd.Series(True, index=net.index)),
            *((model, direction * expected.loc[net.index, model] > hurdle) for model in expected),
            ("tail_high", high_tail.reindex(net.index)),
            ("tail_other", ~high_tail.reindex(net.index)),
        )
        for name, mask in selections:
            values = net[mask]
            result[side][name] = {
                "count": len(values), "wins": int(values.gt(0).sum()),
                "net_mean": float(values.mean()) if len(values) else None,
                "net_mean_without_2022": float(values[values.index.year != 2022].mean())
                if (values.index.year != 2022).any() else None,
                "actual_tail_weeks": int(observed_tail.loc[values.index].sum()),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--history-cache", type=Path)
    args = parser.parse_args()
    history = (pd.read_csv(args.history_cache, index_col=0, parse_dates=True)
               if args.history_cache else load_history(args.end)).loc[:args.end]
    expected, signals = forecasts(history), price_signals(history)
    tail = expected.pop("tail")
    high_tail = elevated_tail(tail)
    for cutoff in ("2018-12-24", "2020-03-24", "2022-10-14", "2025-04-09"):
        past = forecasts(history.loc[:cutoff])
        pd.testing.assert_frame_equal(past, expected.assign(tail=tail).loc[:cutoff])
        pd.testing.assert_frame_equal(price_signals(history.loc[:cutoff]), signals.loc[:cutoff])
        pd.testing.assert_series_equal(elevated_tail(past["tail"]), high_tail.loc[:cutoff])
    prices = history.loc[START:]
    results = {}
    for bps in (25, 50):
        fee = bps / 10_000
        hurdle = np.log((1 + fee) / (1 - fee))
        always = pd.Series(True, index=expected.index)
        targets = {
            "hold": pd.Series(1.0, index=history.index),
            "sma175": signals.trend,
            "raw": overlay(signals, always, always),
            "tail_exit": overlay(signals, high_tail, always),
            "tail_recovery": overlay(signals, always, ~high_tail),
            "tail_both": overlay(signals, high_tail, ~high_tail),
        }
        for name in expected:
            for leg, exit_gate, recovery_gate in (
                ("exit", True, False), ("recovery", False, True), ("both", True, True)
            ):
                targets[f"{name}_{leg}"] = overlay(
                    signals, expected[name].lt(-hurdle) if exit_gate else always,
                    expected[name].gt(hurdle) if recovery_gate else always,
                )
        baseline, _ = replay(prices, signals.trend.loc[START:], bps)
        results[bps] = {"events": opportunities(history, signals, expected, targets["raw"], high_tail, bps),
                        "systems": {}}
        quarter = prices.index[~prices.index.to_period("Q").duplicated()]
        for name, target in targets.items():
            frame, fills = replay(prices, target.loc[START:], bps)
            weight = matched_quarterly_weight(prices, frame.weight.mean())
            matched, matched_fills = replay(prices, pd.Series(weight, index=prices.index), bps,
                                            rebalance_on=quarter)
            returns = daily_returns(frame.nav)
            mask = prices.index.year != 2022
            results[bps]["systems"][name] = {
                **path_metrics(frame, fills),
                "sharpe_without_2022": float(sharpe(returns[mask])),
                "annual_log_edge_without_2022": float(
                    np.log((1 + returns[mask]) / (1 + daily_returns(baseline.nav)[mask])).mean() * 252
                ),
                "sharpe_delta_interval": paired_interval(returns, daily_returns(baseline.nav)),
                "matched": path_metrics(matched, matched_fills),
            }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
