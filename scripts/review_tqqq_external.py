"""Run standalone TQQQ systems without using the MoneyMore strategy engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_credit import adjusted_close
from review_tqqq_execution import ROOT
from moneymore.leveraged_etf import YahooDailyClient


def summary(values, fills):
    nav = pd.Series(values, dtype=float)
    returns = nav.pct_change().dropna()
    return {
        "return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (252 / len(returns)) - 1),
        "sharpe": float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else None,
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "fills": int(fills),
    }


def daily_prices(symbol, end):
    raw = YahooDailyClient().history(symbol, "2010-02-11", end).set_index("trade_date")
    raw.index = pd.to_datetime(raw.index)
    ohlc = raw[["open", "high", "low", "close"]].dropna()
    adjusted = adjusted_close(symbol, end).reindex(ohlc.index)
    return ohlc.mul(adjusted / ohlc.close, axis=0).dropna().rename(columns=str.title)


def hold(prices, cost=0):
    return summary([1, *(prices.Close / (prices.Open.iloc[0] * (1 + cost)))], 1)


def turnaround(prices, start, end, cost, *, next_open):
    down = prices.Close.diff().lt(0)
    buy = down.shift(1, fill_value=False) & down.shift(2, fill_value=False)
    buy &= prices.index.dayofweek.isin([1, 2])
    sell = prices.Close.gt(prices.High.shift())
    prior_sell = sell.shift(1, fill_value=False)
    cash, shares, fills = 1.0, 0.0, 0
    nav = [cash]
    for day, bar in prices.loc[start:end].iterrows():
        if next_open and shares and prior_sell.at[day]:
            cash, shares = shares * bar.Open * (1 - cost), 0.0
            fills += 1
        elif not shares and buy.at[day]:
            cash, shares = 0.0, cash / (bar.Open * (1 + cost))
            fills += 1
        if not next_open and shares and sell.at[day]:
            cash, shares = shares * bar.Close * (1 - cost), 0.0
            fills += 1
        nav.append(cash + shares * bar.Close)
    return summary(nav, fills)


def wildwest_cost_path(signals, capital, band, cost):
    """Cost-aware counterpart to the upstream default long/cash ledger."""
    if not signals.sqqq_alloc.eq(0).all():
        raise ValueError("Cost replay is defined only for the default mom_vol engine")
    shares, cash, peak, cooldown, fills = 0, capital, capital, None, 0
    nav = []
    for offset, row in enumerate(signals.itertuples()):
        price = row.tqqq_open
        equity = cash + shares * price
        peak = max(peak, equity)
        target = row.tqqq_alloc
        if cooldown is not None:
            cooldown += 1
            if cooldown >= 5 and row.regime == "bull":
                peak, cooldown = equity, None
            else:
                target = 0.0
        elif offset and (equity - peak) / peak < -0.25:
            cooldown, target = 0, 0.0
        if not offset or abs(shares * price / equity - target) > band:
            desired = int(equity * target / price)
            if desired > shares:
                desired = min(desired, shares + int(cash / (price * (1 + cost))))
            delta = desired - shares
            cash -= (delta + abs(delta) * cost) * price
            shares = desired
            fills += bool(delta)
        nav.append(cash + shares * row.tqqq_close)
    return nav, fills


def daily_review(end, upstream):
    sys.path.insert(0, str(upstream))
    import config as upstream_config
    from indicators import generate_signals
    from strategy import run_strategy

    if upstream_config.REGIME_METHOD != "mom_vol":
        raise ValueError("Pin the upstream default mom_vol configuration")
    with ThreadPoolExecutor(max_workers=3) as pool:
        symbols = ("TQQQ", "SQQQ", "QQQ")
        market = dict(zip(symbols, pool.map(lambda symbol: daily_prices(symbol, end), symbols), strict=True))
    tqqq = market["TQQQ"]
    capital = upstream_config.STARTING_CAPITAL
    band = upstream_config.REBALANCE_THRESHOLD
    results = {}
    windows = (
        ("2013-01-02", end), ("2013-01-02", "2019-12-31"), ("2020-01-02", "2026-06-30"),
        ("2020-01-02", end), ("2022-01-03", "2022-12-30"), ("2023-01-03", end),
        ("2024-05-28", end),
    )
    for start, stop in dict.fromkeys((start, min(stop, end)) for start, stop in windows):
        if start >= stop:
            continue
        warmup = pd.Timestamp(start) - timedelta(days=400)
        inputs = [market[symbol].loc[warmup:stop].copy() for symbol in symbols]
        signals = generate_signals(*inputs).loc[start:stop]
        native, trades = run_strategy(signals)
        path, _ = wildwest_cost_path(signals, capital, band, 0)
        np.testing.assert_allclose(path, native.portfolio_value, rtol=1e-12)
        window = tqqq.loc[signals.index]
        if not window.index.equals(tqqq.loc[start:stop].index):
            raise ValueError("Daily sessions differ between systems")
        group = {
            "hold_no_cost": hold(window),
            "wildwest_native": summary([capital, *native.portfolio_value], len(trades)),
        }
        for bps in (10, 25):
            cost = bps / 10_000
            path, fills = wildwest_cost_path(signals, capital, band, cost)
            group[f"hold_{bps}bps"] = hold(window, cost)
            group[f"wildwest_{bps}bps"] = summary([capital, *path], fills)
            for next_open in (False, True):
                name = "next_open" if next_open else "source_close"
                group[f"turnaround_{name}_{bps}bps"] = turnaround(tqqq, start, stop, cost, next_open=next_open)
        results[f"{start}/{stop}"] = group
    return {
        "input_sha256": {symbol: hashlib.sha256(frame.to_csv().encode()).hexdigest()
                         for symbol, frame in market.items()},
        "windows": results,
    }


def vwap_day(bars, volume, equity, slip):
    opening, high, low, close = bars.T
    # Archive bars have no trade-level VWAP. Estimate it from typical price and volume.
    vwap = np.cumsum((high + low + close) / 3 * volume) / np.cumsum(volume)
    direction = np.sign(close - vwap)
    # Treat an exact tie as no change in the current signal.
    for i in np.flatnonzero(direction == 0):
        if i:
            direction[i] = direction[i - 1]
    changes = np.flatnonzero(np.diff(direction, prepend=0)[:-1]) + 1
    sizes = [*direction[changes - 1], 0]
    prices = [*opening[changes], close[-1]]
    quantity, fills = 0, 0
    commission = 0.0005
    for price, side in zip(prices, sizes, strict=True):
        if quantity:
            equity += quantity * (price - np.sign(quantity) * slip - entry) - abs(quantity) * commission
            fills += 1
        if side and equity > 0:
            entry = price + side * slip
            quantity = side * int(equity / (entry + commission))
            equity -= abs(quantity) * commission
            fills += bool(quantity)
        else:
            quantity = 0
    return max(equity, 0), fills


def vwap_review(end):
    from review_tqqq_orb import complete_sessions, load, split_factors, xcals

    start = "2018-01-02"
    frame, digest = load("TQQQ", None, None, 1, source="syncster")
    calendar = xcals.get_calendar("XNYS", start="2017-12-28", end=end)
    bars, rejected = complete_sessions(frame, calendar, start, end, 1)
    splits = split_factors("TQQQ", start, str(frame.index[-1].date()))
    sessions = {}
    minute_volume = frame.volume.to_numpy(dtype=float)
    for date, prices in bars.items():
        minutes = pd.date_range(calendar.session_open(date), periods=len(prices), freq="min")
        volume = minute_volume[frame.index.searchsorted(minutes)]
        if not np.isfinite(volume).all() or (volume < 0).any() or volume[0] <= 0:
            rejected.append(date)
            continue
        scale = np.prod([ratio for day, ratio in splits.items() if date < day])
        sessions[date] = (prices * scale, volume)
    output = {}
    for first, last in ((start, min("2023-09-28", end)), ("2023-09-29", end), ("2023-11-14", end)):
        if first > last:
            continue
        dates = [date for date in sessions if first <= date <= last]
        if not dates:
            raise ValueError(f"No valid minute data from {first} to {last}")
        comparison = [1, *[bars[d][-1, 3] / bars[dates[0]][0, 0] for d in dates]]
        results = {"hold": summary(comparison, 1), "sessions": len(dates),
                   "rejected": sum(first <= date <= last for date in rejected),
                   "first": dates[0], "last": dates[-1]}
        for slip in (0, 0.01):
            nav, fills = [25_000.0], 0
            for date in dates:
                prices, volume = sessions[date]
                equity, daily_fills = vwap_day(prices, volume, nav[-1], slip)
                nav.append(equity)
                fills += daily_fills
            results[f"vwap_{slip:.2f}_usd_slip"] = summary(nav, fills)
        output[f"{first}/{last}"] = results
    return {"input_sha256": digest, "windows": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frequency", choices=("daily", "vwap"))
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--upstream", type=Path, default=ROOT / "external_research/wildwest_tqqq")
    args = parser.parse_args()
    result = daily_review(args.end, args.upstream) if args.frequency == "daily" else vwap_review(args.end)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
