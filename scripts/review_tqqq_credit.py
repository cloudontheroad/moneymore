"""Replay one fixed credit confirmation rule with next-open execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from review_tqqq_execution import bootstrap, metrics
from review_tqqq_switch_cost import leverage_rules

from moneymore.leveraged_etf import YahooDailyClient

HOLD_SESSIONS = 20
OAS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"


def adjusted_close(symbol, as_of):
    start = int(pd.Timestamp("2010-02-11", tz="UTC").timestamp())
    stop = int((pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start}&period2={stop}&interval=1d"
    )
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 MoneyMore/1.0"}), timeout=35) as response:
        chart = json.load(response)["chart"]["result"][0]
    dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
    prices = pd.Series(chart["indicators"]["adjclose"][0]["adjclose"], index=dates.tz_localize(None).normalize())
    return prices.dropna()


def weekly_shock(price):
    prior_vol = price.pct_change().rolling(60).std().shift(5) * np.sqrt(5)
    return price.pct_change(5) / prior_vol


def signals(qqq, relative_credit, oas):
    credit_shock = weekly_shock(relative_credit)
    return pd.DataFrame({
        "price": (weekly_shock(qqq) < -1.5) & (qqq < qqq.ewm(span=30, min_periods=30).mean()),
        "credit_proxy": (credit_shock < -1).where(credit_shock.notna()),
        "credit_oas": (oas.diff(5) >= 0.20 - 1e-10).where(oas.diff(5).notna()),
    })


def events(signal, first):
    """Use the same non-overlapping price events for all confirmation rules."""
    starts = []
    next_start = first
    for index in np.flatnonzero(signal.to_numpy()):
        if index >= next_start:
            starts.append(index)
            next_start = index + HOLD_SESSIONS + 1
    return starts


def replay(prices, close_target, first, bps, *, monthly=False):
    """Fractional shares, actual next-open weights, and proportional costs per leg."""
    opens = prices[["QQQ_open", "TQQQ_open"]].to_numpy()
    closes = prices[["QQQ", "TQQQ"]].to_numpy()
    targets = (close_target.to_numpy() - 1) / 2
    months = prices.index.to_period("M")
    shares = np.zeros(2)
    cost_rate = bps / 10_000
    rows = [(1.0, np.nan, 0, 0.0, 0.0)]
    for index in range(first, len(prices)):
        current = shares * opens[index]
        initial = index == first
        if initial or targets[index - 1] != targets[index - 2] or (monthly and months[index] != months[index - 1]):
            open_nav = 1.0 if initial else current.sum()
            weight = targets[index - 1]
            weights = np.array([1 - weight, weight])
            # Solve post-fee equity = pre-fee equity - fees on the actual rebalance.
            investable = open_nav
            for _ in range(8):
                invested = weights * investable
                traded = np.abs(invested - current)
                investable = open_nav - cost_rate * traded.sum()
            np.testing.assert_allclose(invested.sum() + cost_rate * traded.sum(), open_nav, rtol=1e-12)
            shares = invested / opens[index]
        else:
            open_nav = current.sum()
            traded = np.zeros(2)
        marked = shares * closes[index]
        nav = marked.sum()
        rows.append((nav, marked[1] / nav, np.count_nonzero(traded > 1e-12),
                     traded.sum() / open_nav, 100_000 * cost_rate * traded.sum()))
    dates = prices.index[first - 1:]
    return pd.DataFrame(
        rows, index=dates, columns=["normalized_nav", "actual_tqqq", "fill_count", "turnover", "cost"]
    )


def assess(prices, flags, first, confirmations):
    starts = events(flags.price, first)
    dates = prices.index
    choices = {
        "hold": pd.Series(3.0, index=dates),
        "buffer": leverage_rules(prices.QQQ)["buffer_1pct"],
        "monthly_80pct_tqqq": pd.Series(2.6, index=dates),
    }
    positions = {"price_only": starts}
    for name in confirmations:
        confirmed = flags[name].eq(True).to_numpy()
        positions[name] = [index for index in starts if confirmed[index]]
    for name, selected in positions.items():
        target = choices["hold"].copy()
        for index in selected:
            target.iloc[index:index + HOLD_SESSIONS] = 1.0
        choices[name] = target

    folds = {}
    for bps in (6, 25):
        runs = {
            name: replay(prices, target, first, bps, monthly=name == "monthly_80pct_tqqq")
            for name, target in choices.items()
        }
        hold = runs["hold"].normalized_nav
        results = {}
        for name, run in runs.items():
            relative = run.normalized_nav / hold
            periods = {}
            for label, a, b in (
                ("2013–2019", "2013-01-01", "2019-12-31"),
                ("2020–2022", "2020-01-01", "2022-12-31"),
                ("2023–2026", "2023-01-01", "2026-12-31"),
            ):
                changes = np.log(relative).diff().loc[a:b].dropna()
                if not changes.empty:
                    periods[label] = float(np.expm1(changes.sum()))
            results[name] = {
                **metrics(run),
                "relative_return": float(relative.iloc[-1] - 1),
                "relative_return_by_period": periods,
                "annual_log_excess_interval": bootstrap([np.log(relative).diff().dropna().to_numpy()]),
            }
        folds[bps] = results

    # Decompose the relative open-to-open return during each complete round trip.
    bps = 25 / 10_000
    switch_cost = np.log1p(-bps) - np.log1p(bps)
    daily_log = np.log(prices[["QQQ_open", "TQQQ_open"]]).diff()
    ledger = []
    for index in starts:
        entry, exit_ = index + 1, index + HOLD_SESSIONS + 1
        if entry >= len(dates):
            continue
        complete = exit_ < len(dates)
        window = daily_log.iloc[entry + 1:min(exit_ + 1, len(dates))].copy()
        if not complete:
            final_move = np.log(prices[["QQQ", "TQQQ"]].iloc[-1].to_numpy() /
                                prices[["QQQ_open", "TQQQ_open"]].iloc[-1].to_numpy())
            window.loc["last_close"] = final_move
        relative = window.QQQ_open - window.TQQQ_open
        down = window.TQQQ_open < 0
        net = relative.sum() + switch_cost * (2 if complete else 1)
        ledger.append({
            "signal": str(dates[index].date()),
            "entry": str(dates[entry].date()),
            "exit": str(dates[exit_].date()) if complete else None,
            "confirmed_by": [name for name in confirmations if index in positions[name]],
            "missing_signals": [name for name in confirmations if pd.isna(flags[name].iloc[index])],
            "avoided_downside_log": float(relative[down].sum()),
            "up_day_log": float(relative[~down].sum()),
            "net_log": float(net),
            "net_relative": float(np.expm1(net)),
        })
    for name in positions:
        net = sum(row["net_log"] for row in ledger if name == "price_only" or name in row["confirmed_by"])
        np.testing.assert_allclose(np.expm1(net), folds[25][name]["relative_return"], atol=1e-11)
    return {"metrics": folds, "started": {name: len(selected) for name, selected in positions.items()}, "events": ledger}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=5) as pool:
        raw = {s: pool.submit(YahooDailyClient(timeout=35).history, s, "2010-02-11", args.as_of)
               for s in ("QQQ", "TQQQ")}
        bonds = {s: pool.submit(adjusted_close, s, args.as_of) for s in ("HYG", "IEI")}
        market = {s: future.result().set_index("trade_date") for s, future in raw.items()}
        dates = market["QQQ"].index.intersection(market["TQQQ"].index).sort_values()
        market = {s: frame.loc[dates].reset_index() for s, frame in market.items()}
        prices = pd.DataFrame({s: frame.close.to_numpy() for s, frame in market.items()}, index=pd.to_datetime(dates))
        for s, frame in market.items():
            prices[f"{s}_open"] = frame.open.to_numpy()
        credit = pd.DataFrame({s: future.result() for s, future in bonds.items()}).reindex(prices.index)
    with urlopen(OAS_URL, timeout=35) as response:
        payload = response.read()
    spread = pd.read_csv(BytesIO(payload), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    # Observe a print at least one NYSE session late. Never forward fill a missing print.
    spread = spread.reindex(prices.index).shift()
    ratio = credit.HYG / credit.IEI
    flags = signals(prices.QQQ, ratio, spread)
    for date in ("2016-02-10", "2020-03-16", "2022-06-13", "2025-04-04"):
        pd.testing.assert_frame_equal(
            flags.loc[:date], signals(prices.QQQ.loc[:date], ratio.loc[:date], spread.loc[:date])
        )
    first_oas = flags.credit_oas.first_valid_index()
    if first_oas is None:
        raise ValueError("No usable lagged OAS observations within the requested history")
    oas_start = max(pd.Timestamp("2023-11-15"), first_oas)
    print(json.dumps({
        "start": "2013-01-01",
        "end": str(prices.index[-1].date()),
        "oas_first_signal_date": str(oas_start.date()),
        "input_sha256": {
            name: hashlib.sha256(frame.to_csv().encode()).hexdigest()
            for name, frame in {"prices": prices, "bond_etfs": credit, "lagged_oas": spread}.items()
        },
        "long_proxy": assess(
            prices, flags, prices.index.searchsorted("2013-01-01"),
            ["credit_proxy"],
        ),
        "overlap_oas": assess(
            prices, flags, prices.index.searchsorted(oas_start),
            ["credit_proxy", "credit_oas"],
        ),
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
