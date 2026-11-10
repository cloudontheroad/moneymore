"""Test a bounded recovery overlay after v3 momentum cuts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_tqqq_execution import ROOT, bootstrap, metrics
from review_tqqq_signal_edge import raw_signals, targets

from moneymore.leveraged_etf import DATA_SYMBOLS, LeveragedEtfPaper
from moneymore.leveraged_etf_v3 import LeveragedEtfV3Paper

RULES = ("breakout_5", "breakout_10", "breakout_5_vol_easing")
SECONDARY_RULE = "breakout_10_above_sma200"


def recovery_states(qqq: pd.Series, tqqq: pd.Series, bear: pd.Series) -> pd.DataFrame:
    ema = qqq.ewm(span=10, min_periods=10).mean()
    returns = tqqq.pct_change()
    easing = returns.rolling(5).std() < returns.rolling(20).std()
    long_trend = qqq.gt(qqq.rolling(200).mean())
    result = {}
    for rule in (*RULES, SECONDARY_RULE):
        window = 10 if rule.startswith("breakout_10") else 5
        enter = qqq.gt(qqq.shift().rolling(window).max()) & qqq.gt(ema)
        if rule == "breakout_5_vol_easing":
            enter &= easing
        exit_ = qqq.lt(ema) | qqq.pct_change(3).lt(0)
        if rule == SECONDARY_RULE:
            enter &= long_trend
            exit_ |= ~long_trend
        active = False
        previous_bear = False
        flags = []
        for bearish, trigger, failure in zip(bear, enter, exit_, strict=True):
            if not bearish or not previous_bear or failure:
                active = False
            elif trigger:
                active = True
            flags.append(active)
            previous_bear = bearish
        result[rule] = flags
    return pd.DataFrame(result, index=qqq.index)


def recovery_target(
    parent: pd.DataFrame, signals: pd.DataFrame, state: pd.Series, start: int,
) -> pd.DataFrame:
    # Restore half of the gap to the same volatility-based bullish sizing.
    # Parent reductions and the independent RSI override take precedence.
    base = parent.tqqq.to_numpy()
    bull_cap = signals.v3_vol_only.to_numpy()
    may_restore = state.to_numpy().copy()
    may_restore &= ~signals.dip.to_numpy(dtype=bool)
    may_restore[1:] &= base[1:] >= base[:-1] - 1e-8
    may_restore[:start + 1] = False
    added = np.where(may_restore, np.maximum(0, bull_cap - base) / 2, 0)
    return parent.assign(tqqq=(base + added).round(6))


def episode_report(
    parent: pd.DataFrame, candidate: pd.DataFrame, base: pd.DataFrame,
    overlay: pd.DataFrame, opens: pd.Series, start: int, end: int,
) -> list[dict]:
    """Attribute actual daily log excess over affected fill/mark windows."""
    target_delta = candidate.tqqq.iloc[start:end + 1] - parent.tqqq.iloc[start:end + 1]
    active = target_delta.fillna(0).gt(1e-6).to_numpy()
    excess = np.log(overlay.normalized_nav / base.normalized_nav).diff()
    dates = base.index[1:]
    out = []
    i = 0
    while i < len(active) - 1:
        if not active[i]:
            i += 1
            continue
        begin = i
        while i < len(active) and active[i]:
            i += 1
        first_fill = begin + 1
        last_mark = min(i + 1, len(dates) - 1)
        out.append({
            "signal_date": dates[begin],
            "first_eligible_open_date": dates[first_fill],
            "last_mark_date": dates[last_mark],
            "target_add_first": float(target_delta.iloc[begin]),
            "tqqq_open_to_open_return": float(opens.iloc[start + last_mark] / opens.iloc[start + first_fill] - 1),
            "net_log_excess": float(excess.loc[dates[first_fill]:dates[last_mark]].sum()),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/leveraged-etf")
    args = parser.parse_args()
    inputs, hashes = {}, {}
    for symbol in DATA_SYMBOLS:
        raw = (args.cache / f"{symbol.lstrip('^')}.csv").read_bytes()
        hashes[symbol] = hashlib.sha256(raw).hexdigest()
        frame = pd.read_csv(BytesIO(raw))
        inputs[symbol] = frame.loc[frame.trade_date <= args.as_of]
    data = LeveragedEtfPaper._align(inputs)
    dates = pd.DatetimeIndex(data["QQQ"].trade_date)
    signals = raw_signals(data, dates)
    state = recovery_states(data["QQQ"].close.set_axis(dates), data["TQQQ"].close.set_axis(dates),
                            signals.momentum.le(0))

    def replay(target, start, end, bps=6, *, hold=False):
        rows = LeveragedEtfPaper.replay(
            data, target, list(range(start, end + 1)), "BASELINE",
            slippage=(bps - 1) / 10_000, buy_hold=hold,
        )
        return pd.DataFrame(rows).set_index("trade_date")

    def recovery_only(target, start, end):
        parent = targets(signals, start, end, "v3")
        if target == "v3":
            return parent
        return recovery_target(parent, signals, state[target], start)

    def fit_static(parent_vol, parent, start, end):
        low, high = 0.0, 1.0
        for _ in range(12):
            weight = (low + high) / 2
            static = parent.assign(tqqq=weight)
            if metrics(replay(static, start, end))["volatility"] < parent_vol:
                low = weight
            else:
                high = weight
        return (low + high) / 2

    folds, active_by_fold = {}, {}
    candidates = (*RULES, SECONDARY_RULE)
    for year in range(2022, dates[-1].year + 1):
        start = int(dates.searchsorted(f"{year}-01-01"))
        end = min(len(dates) - 1, int(dates.searchsorted(f"{year + 1}-01-01")) - 1)
        if end <= start:
            break
        train_start = int(dates.searchsorted(f"{year - 3}-01-01"))
        parent_train = recovery_only("v3", train_start, start - 1)
        train_results = {
            name: metrics(replay(recovery_only(name, train_start, start - 1), train_start, start - 1))
            for name in ("v3", *RULES)
        }
        selected = max(train_results, key=lambda name: train_results[name]["sharpe"])
        weight = fit_static(train_results["v3"]["volatility"], parent_train, train_start, start - 1)
        target_by_name = {name: recovery_only(name, start, end) for name in ("v3", *candidates)}
        parent = target_by_name["v3"]
        live_model = LeveragedEtfV3Paper._backtest_targets(data, start)
        np.testing.assert_allclose(parent.tqqq.iloc[start:end + 1], live_model.tqqq.iloc[start:end + 1])
        target_by_name.update(rolling_choice=target_by_name[selected], static=parent.assign(tqqq=weight),
                              hold=parent)
        daily = {name: replay(t, start, end, hold=name == "hold") for name, t in target_by_name.items()}
        for name in (*candidates, "rolling_choice"):
            for ref in ("v3", "static", "hold"):
                active = np.log(daily[name].normalized_nav / daily[ref].normalized_nav).diff().dropna()
                active_by_fold.setdefault((name, ref), []).append(active.to_numpy())
        folds[year] = {
            "start": str(dates[start].date()), "end": str(dates[end].date()), "selected": selected,
            "static_weight": weight,
            "metrics": {name: metrics(frame) for name, frame in daily.items()},
            "stress_25bps": {name: metrics(replay(t, start, end, 25, hold=name == "hold"))
                             for name, t in target_by_name.items()},
            "episodes": {name: episode_report(parent, target_by_name[name], daily["v3"],
                                               daily[name], data["TQQQ"].open, start, end)
                         for name in candidates},
        }
    intervals = {name: {ref: bootstrap(active_by_fold[name, ref]) for ref in ("v3", "static", "hold")}
                 for name in (*candidates, "rolling_choice")}
    gates = {}
    for name in (*candidates, "rolling_choice"):
        checks = {
            "2022_no_deeper_drawdown": folds[2022]["metrics"][name]["max_drawdown"] >=
            folds[2022]["metrics"]["v3"]["max_drawdown"] - 1e-8,
            "2023_higher_sharpe_and_return": all(
                folds[2023]["metrics"][name][field] > folds[2023]["metrics"]["v3"][field]
                for field in ("sharpe", "return")
            ),
            "positive_lower_bound_vs_all_baselines": all(
                low > 0 for low, _, _ in intervals[name].values()
            ),
            "survives_25bps_vs_parent": float(np.prod([
                (1 + fold["stress_25bps"][name]["return"]) / (1 + fold["stress_25bps"]["v3"]["return"])
                for fold in folds.values()
            ])) > 1,
        }
        gates[name] = {"checks": checks, "passed": all(checks.values())}
    print(json.dumps({"input_sha256": hashes, "folds": folds, "annual_log_excess_intervals": intervals,
                      "gates": gates}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
