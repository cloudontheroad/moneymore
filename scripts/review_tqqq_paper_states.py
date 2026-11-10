"""Nasdaq leverage states, conditional volatility, and pre-inception stress tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd

from review_tqqq_execution import metrics
from review_tqqq_signal_edge import raw_signals, targets
from moneymore.leveraged_etf import DATA_SYMBOLS, LeveragedEtfPaper, YahooDailyClient

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
METHODS = ("trend_3x_1x", "conditional_vol", "trend_plus_conditional", "continuous_vol")


def dated(frame):
    return frame.set_index(pd.to_datetime(frame.trade_date)).sort_index()


def monthly_scales(close):
    daily = close.pct_change()
    month = close.index.to_period("M")
    realized = daily.groupby(month).std()
    long_vol = daily.expanding(min_periods=2520).std().groupby(month).last()
    high = realized.expanding(min_periods=120).quantile(0.8)
    # First close of month uses only the previous completed month's observations.
    ratio = (long_vol / realized).clip(upper=1).shift(1)
    conditional = ratio.where(realized.shift(1).gt(high.shift(1)), 1.0)
    conditional = conditional.where(ratio.notna() & high.shift(1).notna())
    return conditional, ratio


def leverage_targets(qqq, ndx):
    above = qqq.close.gt(qqq.close.rolling(200).mean()).where(qqq.close.rolling(200).mean().notna())
    trend = (1 + 2 * above.astype(float))
    conditional, continuous = monthly_scales(ndx.close)
    months = qqq.index.to_period("M")
    cond = pd.Series(conditional.reindex(months).to_numpy(), index=qqq.index)
    cont = pd.Series(continuous.reindex(months).to_numpy(), index=qqq.index)
    return pd.DataFrame({
        "trend_3x_1x": trend,
        "conditional_vol": 3 * cond,
        "trend_plus_conditional": np.minimum(trend, 3 * cond),
        "continuous_vol": 3 * cont,
    })


def allocations(leverage):
    # Above 1x use QQQ/TQQQ. Below 1x use QQQ/cash. The shared replay's
    # second slot is remapped to QQQ only inside this research run.
    tqqq = ((leverage - 1) / 2).clip(0, 1)
    qqq = leverage.clip(0, 1).where(leverage < 1, 1 - tqqq)
    return pd.DataFrame({"tqqq": tqqq.to_numpy(), "sqqq": qqq.to_numpy(), "regime": leverage.name})


def synthetic_prices(qqq, fed_funds, spread=0.0):
    close_return = qqq.close.pct_change().fillna(0)
    overnight = (qqq.open / qqq.close.shift() - 1).fillna(0)
    # Use a one-trading-session lag, then charge two financed turns plus fees.
    rates = fed_funds.reindex(fed_funds.index.union(qqq.index)).sort_index().ffill().reindex(qqq.index).shift()
    rates.iloc[0] = rates.iloc[1]
    if rates.isna().any():
        raise ValueError("Financing series does not cover the synthetic period")
    net = 3 * close_return - (2 * (rates + spread) + 0.0095) / 252
    if (net <= -1).any() or (1 + 3 * overnight <= 0).any():
        raise ValueError("Synthetic 3x path encountered a wipeout")
    close = 100 * (1 + net).cumprod()
    opening = close.shift(fill_value=100) * (1 + 3 * overnight)
    return pd.DataFrame({"trade_date": qqq.trade_date, "open": opening, "close": close}, index=qqq.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    args = parser.parse_args()
    start = "1985-01-01"

    def fetch(symbol):
        lower = start if symbol == "^NDX" else ("1999-03-01" if symbol == "QQQ" else "2010-02-11")
        frame = YahooDailyClient(timeout=35).history(symbol, lower, args.as_of)
        return symbol, frame.loc[frame.trade_date <= args.as_of]

    with ThreadPoolExecutor(max_workers=3) as pool:
        inputs = dict(pool.map(fetch, (*DATA_SYMBOLS, "^NDX")))
    with urllib.request.urlopen(f"{FRED}?id=DFF&cosd=1999-03-01&coed={args.as_of}", timeout=35) as response:
        rate_bytes = response.read()
    rates = pd.read_csv(BytesIO(rate_bytes), index_col=0, parse_dates=True)["DFF"] / 100
    ndx, qqq = dated(inputs["^NDX"]), dated(inputs["QQQ"])
    leverage = leverage_targets(qqq, ndx)
    for day in ("2001-10-15", "2008-10-15", "2022-04-18"):
        pd.testing.assert_frame_equal(leverage.loc[:day], leverage_targets(qqq.loc[:day], ndx.loc[:day]))

    real = LeveragedEtfPaper._align({s: inputs[s] for s in DATA_SYMBOLS})
    dates = pd.DatetimeIndex(real["QQQ"].trade_date)
    real_market = {**real, "SQQQ": real["QQQ"]}
    real_leverage = leverage.reindex(dates)
    real_targets = {name: allocations(real_leverage[name]) for name in METHODS}
    start_index = dates.searchsorted("2013-01-01")
    real_targets["v3"] = targets(raw_signals(real, dates), start_index, len(dates) - 1, "v3")
    real_targets["hold"] = real_targets["trend_3x_1x"]

    def replay(market, target, first, last, cost, hold=False):
        rows = LeveragedEtfPaper.replay(
            market, target, list(range(first, last + 1)), "BASELINE",
            slippage=(cost - 1) / 10_000, buy_hold=hold,
        )
        return pd.DataFrame(rows).set_index("trade_date")

    def window(frame, start, end):
        cut = frame.iloc[1:].loc[start:end]
        prev = frame.index.get_loc(cut.index[0]) - 1
        result = pd.concat([frame.iloc[[prev]], cut]).copy()
        result["normalized_nav"] /= result.normalized_nav.iloc[0]
        return metrics(result)

    results = {}
    for cost in (6, 25):
        daily = {name: replay(real_market, target, start_index, len(dates) - 1, cost, name == "hold")
                 for name, target in real_targets.items()}
        rolling = {}
        for name, frame in daily.items():
            nav = frame.normalized_nav.iloc[1:]
            relative = nav / daily["hold"].normalized_nav.iloc[1:]
            change = (relative / relative.shift(756)).dropna()
            rolling[name] = {
                "n": len(change), "win_rate": float((change > 1).mean()),
                "median_annual_relative_return": float(np.median(change ** (1 / 3) - 1)),
            }
        results[str(cost)] = {
            "all": {name: metrics(frame) for name, frame in daily.items()},
            "2022": {name: window(frame, "2022-01-01", "2022-12-31") for name, frame in daily.items()},
            "2023_2025": {name: window(frame, "2023-01-01", "2025-12-31") for name, frame in daily.items()},
            "rolling": rolling,
        }

    synthetic, validation = {}, {}
    for spread in (0.0, 0.01):
        proxy = synthetic_prices(qqq, rates, spread)
        actual = dated(inputs["TQQQ"]).close
        compare = pd.concat({"proxy": proxy.close.pct_change(), "real": actual.pct_change()}, axis=1).dropna()
        validation[str(spread)] = {
            "days": len(compare), "daily_return_correlation": float(compare.corr().iloc[0, 1]),
            "daily_return_rmse": float(np.sqrt((compare.proxy - compare.real).pow(2).mean())),
            "annualized_growth": ((1 + compare).prod() ** (252 / len(compare)) - 1).to_dict(),
        }
        synthetic_market = {"QQQ": qqq, "TQQQ": proxy, "SQQQ": qqq}
        stress_targets = {name: allocations(leverage[name]) for name in METHODS}
        stress_targets["hold"] = stress_targets["trend_3x_1x"]
        synthetic[str(spread)] = {}
        for label, lower, upper in (("dotcom", "2000-03-10", "2002-10-31"), ("gfc", "2007-10-01", "2009-06-30")):
            first = qqq.index.searchsorted(lower)
            last = qqq.index.searchsorted(upper, side="right") - 1
            synthetic[str(spread)][label] = {
                name: metrics(replay(synthetic_market, target, first, last, 25, name == "hold"))
                for name, target in stress_targets.items()
            }
    print(json.dumps({
        "as_of": str(dates[-1].date()), "actual": results, "synthetic_validation": validation,
        "synthetic_stress": synthetic, "hashes": {
            **{name: hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() for name, frame in inputs.items()},
            "DFF": hashlib.sha256(rate_bytes).hexdigest(),
        },
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
