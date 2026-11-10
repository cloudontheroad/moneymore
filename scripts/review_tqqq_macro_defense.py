"""Evaluate low-frequency, full cash defenses for TQQQ under macro stress."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.request import urlopen

import numpy as np
import pandas as pd

from review_tqqq_credit import adjusted_close
from review_tqqq_external import daily_prices, summary
from review_tqqq_swing import COSTS, START, WAVES, replay
from moneymore.leveraged_etf import YahooDailyClient


def fred_curve():
    with urlopen("https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M", timeout=25) as response:
        data = pd.read_csv(BytesIO(response.read()), index_col=0, parse_dates=True, na_values=".")
    return data.T10Y3M.dropna()


def load_history(end):
    def volatility(symbol):
        frame = YahooDailyClient().history(symbol, "2010-02-11", end)
        return pd.Series(frame.close.to_numpy(), index=pd.DatetimeIndex(frame.trade_date))

    with ThreadPoolExecutor(max_workers=7) as pool:
        prices = {name: pool.submit(daily_prices, name, end) for name in ("QQQ", "TQQQ")}
        bonds = {name: pool.submit(adjusted_close, name, end) for name in ("HYG", "IEI")}
        vol = {name: pool.submit(volatility, "^" + name) for name in ("VIX", "VIX3M")}
        curve = pool.submit(fred_curve)
        history = pd.concat([
            prices[name].result()[["Open", "Close"]].add_prefix(name.lower()) for name in ("QQQ", "TQQQ")
        ], axis=1, join="inner")
        for name, task in {**bonds, **vol}.items():
            history[name] = task.result()
        # Carry forward the last published daily spread across bank holidays.
        # Shift one market session to allow for FRED publication after the close.
        history["curve"] = curve.result().reindex(history.index, method="ffill").shift()
    if history.loc[START:].isna().any().any():
        raise ValueError(f"Missing inputs: {history.loc[START:].isna().sum().to_dict()}")
    return history


def signal_inputs(history):
    qqq = history.qqqClose
    slow = qqq.rolling(200).mean()
    credit = history.HYG / history.IEI
    inverted = history.curve.rolling(20).mean().lt(0)
    weekly_vol = np.log(qqq).diff().rolling(60).std().shift(5) * np.sqrt(5)
    return pd.DataFrame({
        "slow_down": qqq.lt(slow * 0.99),
        "slow_up": qqq.gt(slow * 1.01),
        "fast_down": qqq.lt(qqq.shift().rolling(63).min()),
        "sharp_shock": np.log(qqq / qqq.shift(5)).lt(-2 * weekly_vol),
        "fast_up": qqq.gt(qqq.shift().rolling(20).max()),
        "vol_stress": history.VIX.gt(history.VIX3M),
        "credit_stress": credit.lt(credit.shift().rolling(63).min()),
        "curve_watch": inverted.rolling(252).max().fillna(0).astype(bool),
    })


def cash_state(exit_, entry):
    weight = 1.0
    targets = []
    for reduce, recover in zip(exit_, entry, strict=True):
        if reduce:
            weight = 0.0
        elif recover:
            weight = 1.0
        targets.append(weight)
    return pd.Series(targets, index=exit_.index).shift(1, fill_value=1.0)


def opening_weights(history):
    flags = signal_inputs(history)
    down = flags.slow_down
    fast = flags.fast_down
    up = flags.slow_up
    vol, credit, curve = flags.vol_stress, flags.credit_stress, flags.curve_watch
    recovery = up & flags.fast_up
    shock = flags.sharp_shock
    monthly = history.qqqClose.groupby(history.index.to_period("M")).last()
    monthly_target = monthly.gt(monthly.rolling(10).mean()).astype(float).shift(1)
    return {
        "hold": pd.Series(1.0, index=history.index),
        "monthly_trend": pd.Series(monthly_target.reindex(history.index.to_period("M")).to_numpy(),
                                   index=history.index),
        "slow_trend": cash_state(down, up),
        "price_fast": cash_state(down | fast, up),
        "vol_fast": cash_state(down | (fast & vol), up),
        "credit_fast": cash_state(down | (fast & credit), up),
        "either_fast": cash_state(down | (fast & (vol | credit)), up),
        "both_fast": cash_state(down | (fast & vol & credit), up),
        "curve_watch_fast": cash_state(down | (fast & (vol | credit | curve)), up),
        "either_strict_reentry": cash_state(down | (fast & (vol | credit)), recovery),
        "shock_price_only": cash_state(down | shock, recovery),
        "shock_macro_confirmed": cash_state(down | (shock & (vol | credit)), recovery),
        "shock_and_quarterly_break": cash_state(down | fast | (shock & (vol | credit)), recovery),
    }


def risk_metrics(frame, target, fills, history, delay):
    nav = frame.nav
    result = summary([1, *nav], fills)
    trough = (nav / nav.cummax()).idxmin()
    peak = nav.loc[:trough].idxmax()
    result["max_drawdown_period"] = [str(peak.date()), str(trough.date())]
    result["worst_period_exits"] = target.diff().lt(0).loc[peak:trough].sum().item()
    for length in (21, 63):
        result[f"worst_{length}_session_return"] = float(nav.pct_change(length).min())
    result["cash_fraction"] = float(target.eq(0).mean())
    result["average_tqqq_weight"] = float(frame.weight.mean())
    result["max_actual_tqqq_weight_at_open"] = float(frame.weight.max())
    result["reentries"] = int(target.diff().gt(0).sum())
    result["exits"] = int(target.diff().lt(0).sum())
    drawdown = abs(result["max_drawdown"])
    result["calmar"] = result["cagr"] / drawdown if drawdown else None
    result["delay_sessions"] = delay
    windows = {}
    for label, (start, stop) in WAVES.items():
        window = nav.loc[start:stop]
        i = nav.index.get_loc(window.index[0])
        initial = nav.iloc[i - 1] if i else 1.0
        market = history.tqqqOpen.loc[window.index]
        off = target.diff().lt(0).loc[start:stop]
        cuts = off.index[off]
        deepest = market.idxmin()
        first_cut = cuts[0] if len(cuts) else None
        windows[label] = {
            "return": float(window.iloc[-1] / initial - 1),
            "max_drawdown": summary([initial, *window], 0)["max_drawdown"],
            "cash_fraction": float(target.loc[start:stop].eq(0).mean()),
            "first_exit": str(first_cut.date()) if first_cut is not None else None,
            "tqqq_trough_open": str(deepest.date()),
            "cash_at_trough": bool(target.loc[deepest] == 0),
            "market_drawdown_before_exit": float(market.at[first_cut] / market.loc[:first_cut].max() - 1)
            if first_cut is not None else None,
        }
    result["waves"] = windows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    history = load_history(args.end)
    weights = opening_weights(history)
    for cutoff in ("2018-12-24", "2020-03-16", "2022-10-13", "2025-04-09"):
        for name, target in opening_weights(history.loc[:cutoff]).items():
            pd.testing.assert_series_equal(target, weights[name].loc[:cutoff])
    prices = history.loc[START:]
    quarters = prices.index.to_period("Q")
    quarter_opens = prices.index[~quarters.duplicated()]
    capped = {
        "fixed_35pct": weights["hold"] * 0.35,
        "fixed_50pct": weights["hold"] * 0.5,
        "fixed_55pct": weights["hold"] * 0.55,
        "fixed_75pct": weights["hold"] * 0.75,
        "shock_50pct_or_cash": weights["shock_price_only"] * 0.5,
        "shock_75pct_or_cash": weights["shock_price_only"] * 0.75,
    }
    results = {}
    # Delay only the strategy orders, and preserve the common first-open benchmark.
    for bps, delay in [*((bps, 0) for bps in COSTS), (25, 1)]:
        group = {}
        for name, target in {**weights, **capped}.items():
            target = target.shift(delay, fill_value=target.iloc[0]).loc[prices.index]
            path, fills = replay(prices, target, bps, rebalance_on=quarter_opens if name in capped else ())
            group[name] = risk_metrics(path, target, fills, prices, delay)
        results[f"{bps}bps_delay{delay}"] = group
    print(json.dumps({
        "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
