"""Isolate TQQQ signal contributions under the MoneyMore execution policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_tqqq_execution import ROOT, bootstrap, metrics

from moneymore.leveraged_etf import DATA_SYMBOLS, LeveragedEtfPaper
from moneymore.leveraged_etf_v3 import LeveragedEtfV3Paper, v3_target
from moneymore.tqqq_cycle import CycleParameters, cycle_decisions, cycle_target
from moneymore.tqqq_v3 import V3Parameters, raw_targets

V2_MODES = ("v2", "v2_no_fear", "v2_no_rates", "v2_neutral")
V3_MODES = ("v3", "v3_no_momentum_cut", "v3_no_dip", "v3_no_vol", "v3_vol_only")
EXPLORATORY = {"v2": "v2_fear_guard", "v3": "v3_dip_hold5"}
HORIZONS = (1, 5, 20)


def raw_signals(data: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    def close(symbol: str) -> pd.Series:
        return pd.Series(data[symbol].close.to_numpy(), index=dates)

    qqq, tqqq = close("QQQ"), close("TQQQ")
    vix, vix3m = close("^VIX"), close("^VIX3M")
    cycle = cycle_decisions(qqq, vix, close("^TNX"), vix3m)
    p = CycleParameters()
    signals = pd.DataFrame(index=dates)
    signals["guard"] = cycle.crash_guard_active
    weak_trend = qqq < qqq.ewm(span=50, min_periods=50).mean()
    for name in (*V2_MODES, EXPLORATORY["v2"]):
        fear = 50 if name in ("v2_no_fear", "v2_neutral") else cycle.fear_rank
        rates = 50 if name in ("v2_no_rates", "v2_neutral") else cycle.rate_rank
        if name == EXPLORATORY["v2"]:
            fear = fear.where(~weak_trend, fear.clip(upper=50))
        desired = p.base_exposure + p.cycle_slope * ((fear + rates) / 200 - 0.5)
        desired = pd.Series(desired, index=dates).clip(p.minimum_exposure, p.maximum_exposure)
        desired = desired.where(~signals.guard, desired * p.crash_multiplier)
        signals[name] = desired.clip(p.minimum_exposure, p.maximum_exposure).where(cycle.raw_target.notna())

    p3 = V3Parameters()
    variants = {
        "v3": p3,
        "v3_no_momentum_cut": replace(p3, negative_momentum_scale=1.0),
        "v3_no_dip": replace(p3, use_dip_buy=False),
        "v3_no_vol": replace(p3, use_relative_volatility=False),
        "v3_vol_only": replace(p3, negative_momentum_scale=1.0, use_dip_buy=False),
    }
    for name, params in variants.items():
        decision = raw_targets(qqq, tqqq, vix, vix3m, params)
        signals[name] = decision.raw_target
        if name == "v3":
            signals["dip"] = decision.dip_buy
            signals["momentum"] = decision.momentum
    recent_dip = signals.dip.rolling(5, min_periods=1).max().astype(bool)
    signals[EXPLORATORY["v3"]] = signals.v3.where(
        ~(recent_dip & signals.momentum.le(0)), signals.v3.clip(lower=p3.dip_buy_allocation)
    )
    return signals


def targets(signals: pd.DataFrame, start: int, end: int, mode: str) -> pd.DataFrame:
    values = np.full(len(signals), np.nan)
    prior = None
    for i in range(start, end + 1):
        if np.isfinite(signals[mode].iat[i]):
            prior = target_at(signals, i, mode, prior)
            values[i] = prior
    return pd.DataFrame({"tqqq": values, "regime": mode})


def target_at(signals: pd.DataFrame, i: int, mode: str, previous: float | None) -> float:
    desired = signals[mode].iat[i]
    if mode in V2_MODES or mode == EXPLORATORY["v2"]:
        month_open = i == 0 or signals.index[i].month != signals.index[i - 1].month
        return round(cycle_target(desired, previous, bool(signals.guard.iat[i]), month_open,
                                  CycleParameters().rebalance_band), 6)
    return v3_target(desired, previous, V3Parameters().rebalance_band)


def event_study(
    data: dict[str, pd.DataFrame], signals: pd.DataFrame, start: int, end: int, model: str
) -> list[dict]:
    """Price the changed tranche from next open, with a round trip at 6 bps/side."""
    actual = targets(signals, start, end, model).tqqq
    opens = data["TQQQ"].open.to_numpy()
    events = []
    for i in range(start + 1, end):
        prior, weight = actual.iat[i - 1], actual.iat[i]
        if not np.isfinite(prior) or abs(weight - prior) < 1e-6:
            continue
        variants = V2_MODES[1:3] if model == "v2" else V3_MODES[1:4]
        differences = {"keep_prior": weight - prior}
        for omitted in variants:
            differences[omitted] = weight - target_at(signals, i, omitted, prior)
        if model == "v3":
            dip = bool(signals.dip.iat[i])
            previous_dip = bool(signals.dip.iat[i - 1])
            if dip != previous_dip:
                group = "dip_enter" if dip else "dip_exit"
                differences[group] = weight - prior
        for factor, delta in differences.items():
            if abs(delta) < 1e-6:
                continue
            for horizon in HORIZONS:
                entry, exit_ = i + 1, i + 1 + horizon
                if exit_ > end:
                    continue
                ratio = opens[exit_] / opens[entry]
                active = delta * (ratio - 1) - abs(delta) * 0.0006 * (1 + ratio)
                events.append({"signal_date": str(signals.index[i].date()), "model": model,
                               "factor": factor, "direction": "add" if delta > 0 else "cut",
                               "horizon": horizon, "weight_delta": float(delta),
                               "net_contribution": float(active)})
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/leveraged-etf")
    args = parser.parse_args()
    frames = {}
    hashes = {}
    for symbol in DATA_SYMBOLS:
        raw = (args.cache / f"{symbol.lstrip('^')}.csv").read_bytes()
        frame = pd.read_csv(BytesIO(raw))
        frames[symbol] = frame[frame.trade_date <= args.as_of]
        hashes[symbol] = hashlib.sha256(raw).hexdigest()
    data = LeveragedEtfPaper._align(frames)
    dates = pd.DatetimeIndex(data["QQQ"].trade_date)
    signals = raw_signals(data, dates)
    first_valid = signals.v2.first_valid_index()
    first = int(dates.searchsorted(first_valid))
    for mode, model in (("v2", LeveragedEtfPaper), ("v3", LeveragedEtfV3Paper)):
        np.testing.assert_allclose(targets(signals, first, len(dates) - 1, mode).tqqq,
                                   model._backtest_targets(data, first).tqqq, atol=1e-6)

    def replay(target, a, b, bps=6, *, hold=False):
        if isinstance(target, float):
            target = pd.DataFrame({"tqqq": target, "regime": "STATIC"}, index=range(len(dates)))
        rows = LeveragedEtfPaper.replay(data, target, list(range(a, b + 1)), "BASELINE",
                                       slippage=(bps - 1) / 10_000, buy_hold=hold)
        return pd.DataFrame(rows).set_index("trade_date")

    def match_weight(volatility, a, b):
        low, high = 0.0, 1.0
        for _ in range(12):
            middle = (low + high) / 2
            if metrics(replay(middle, a, b))["volatility"] < volatility:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    folds = {}
    all_events = []
    all_returns = {}
    focus = {}
    for year in range(2022, dates[-1].year + 1):
        start = int(dates.searchsorted(f"{year}-01-01"))
        end = min(len(dates) - 1, int(dates.searchsorted(f"{year + 1}-01-01")) - 1)
        if end <= start:
            break
        train = max(first, int(dates.searchsorted(f"{year - 3}-01-01")))
        modes_to_test = (*V2_MODES, *V3_MODES, *EXPLORATORY.values())
        target_by_mode = {mode: targets(signals, start, end, mode) for mode in modes_to_test}
        train_results = {mode: metrics(replay(targets(signals, train, start - 1, mode), train, start - 1))
                         for mode in (*V2_MODES, *V3_MODES)}
        chosen = {}
        for family, modes in (("v2", V2_MODES), ("v3", V3_MODES)):
            pick = max(modes, key=lambda mode: train_results[mode]["sharpe"])
            chosen[family] = pick
            target_by_mode[family + "_rolling_choice"] = target_by_mode[pick]
            target_by_mode[family + "_static"] = match_weight(
                train_results[family]["volatility"], train, start - 1
            )
            all_events.extend(event_study(data, signals, start, end, family))
        target_by_mode["hold"] = 1.0
        daily = {mode: replay(target, start, end, hold=mode == "hold") for mode, target in target_by_mode.items()}
        for mode, frame in daily.items():
            all_returns.setdefault(mode, []).append(np.log(frame.normalized_nav).diff().dropna().to_numpy())
        folds[year] = {
            "train": [str(dates[train].date()), str(dates[start - 1].date())],
            "test": [str(dates[start].date()), str(dates[end].date())],
            "chosen": chosen,
            "static_weight": {name: value for name, value in target_by_mode.items() if name.endswith("_static")},
            "metrics": {mode: metrics(frame) for mode, frame in daily.items()},
            "stress_25bps": {mode: metrics(replay(target, start, end, bps=25, hold=mode == "hold"))
                             for mode, target in target_by_mode.items()},
        }
        if year in (2023, 2025):
            model = "v3" if year == 2023 else "v2"
            for mode in (model, model + "_static", *(
                V3_MODES[1:4] if model == "v3" else V2_MODES[1:3]
            )):
                frame = daily[mode]
                nav = frame.normalized_nav
                trough = (nav / nav.cummax() - 1).idxmin()
                peak = nav.loc[:trough].idxmax()
                focus[f"{year}_{mode}"] = {
                    "peak": peak, "trough": trough,
                    "drawdown": float(nav.loc[trough] / nav.loc[peak] - 1),
                    "exposure_peak_to_trough": float(frame.loc[peak:trough].actual_tqqq.mean()),
                }
    events = pd.DataFrame(all_events)
    groups = events.groupby(["model", "factor", "direction", "horizon"]).net_contribution
    event_summary = groups.agg(n="size", mean="mean", median="median", positive_rate=lambda x: (x > 0).mean())
    comparisons = {}
    for modes in (V2_MODES, V3_MODES):
        parent = modes[0]
        for mode in (*modes[1:], parent + "_rolling_choice", EXPLORATORY[parent]):
            comparisons[mode] = {}
            for baseline in (parent, parent + "_static", "hold"):
                active = [a - b for a, b in zip(all_returns[mode], all_returns[baseline], strict=True)]
                comparisons[mode][baseline] = bootstrap(active)
    gates = {}
    for parent, mode in EXPLORATORY.items():
        checks = {
            "positive_excess_lower_bound_vs_parent": comparisons[mode][parent][0] > 0,
            "positive_excess_lower_bound_vs_static": comparisons[mode][parent + "_static"][0] > 0,
            "positive_excess_lower_bound_vs_hold": comparisons[mode]["hold"][0] > 0,
            "improves_sharpe_in_majority": sum(
                fold["metrics"][mode]["sharpe"] > fold["metrics"][parent]["sharpe"]
                for fold in folds.values()
            ) > len(folds) / 2,
            "2022_drawdown_no_worse": folds[2022]["metrics"][mode]["max_drawdown"] >=
            folds[2022]["metrics"][parent]["max_drawdown"] - 1e-8,
        }
        gates[mode] = {"checks": checks, "passed": all(checks.values())}
    print(json.dumps({
        "input_sha256": hashes, "folds": folds, "drawdown_focus": focus,
        "event_summary": event_summary.reset_index().to_dict("records"),
        "events": all_events,
        "annual_log_excess_intervals": comparisons,
        "exploratory_gates": gates,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
