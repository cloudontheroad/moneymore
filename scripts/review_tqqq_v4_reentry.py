"""Test earlier V4 reentries while keeping every original exit and reentry."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from moneymore.leveraged_etf import YahooDailyClient
from moneymore.tqqq_v4 import SECTORS, close_signals, v4_targets
from review_tqqq_peak_exit import episodes
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics


def load(end):
    symbols = ("QQQ", "TQQQ", "HYG", "IEI", "^VIX", "^VIX3M", "^TNX",
               "SOXX", "SPY", "RSP", "RSPD", "RSPS", *SECTORS)

    def fetch(symbol):
        frame = YahooDailyClient().history(symbol, "2010-02-11", end, adjusted=not symbol.startswith("^"))
        return frame.set_index(pd.to_datetime(frame.trade_date)).sort_index()

    with ThreadPoolExecutor(max_workers=6) as pool:
        data = dict(zip(symbols, pool.map(fetch, symbols), strict=True))
    prices = pd.concat([data[s][["open", "close"]].rename(
        columns={"open": s.lower() + "Open", "close": s.lower() + "Close"})
        for s in ("QQQ", "TQQQ")], axis=1, join="inner").loc[:end]
    if prices.index[-1] != pd.Timestamp(end):
        raise ValueError("The requested final session is unavailable")
    close = pd.DataFrame({s: data[s].close.reindex(prices.index) for s in symbols})
    prices = prices.join(close[["HYG", "IEI", "^VIX", "^VIX3M"]].rename(columns=lambda s: s.lstrip("^")))
    return prices, close


def early_weight(base, ready, stop, size):
    """Open at most one trial during each original cash interval."""
    weights = base.copy()
    cash = base.eq(0)
    for _, block in cash.loc[cash].groupby(cash.ne(cash.shift()).cumsum()):
        # The exit remains cash at its original open.
        dates = block.index[1:]
        entries = dates[ready.reindex(dates, fill_value=False) & ~stop.reindex(dates, fill_value=False)]
        if not len(entries):
            continue
        trial = dates[dates.get_loc(entries[0]):][:10]
        breaks = np.flatnonzero(stop.reindex(trial, fill_value=False))
        weights.loc[trial[:breaks[0]] if len(breaks) else trial] = size
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history, close = load(args.end)
    next_session = xcals.get_calendar("XNYS").next_session(pd.Timestamp(args.end))
    base = v4_targets(history, close, close, close, next_session=next_session)
    price = history.qqqClose
    credit = (history.HYG / history.IEI).pct_change(5).gt(0)
    volatility = (history.VIX / history.VIX3M).diff(5).lt(0)
    breakout = price.gt(price.shift().rolling(5).max())
    trend = price.gt(price.rolling(175).mean())
    price_floor = price.lt(price.shift().rolling(5).min())
    exit_signal = close_signals(history, close, close).exit
    confirmations = {
        "trend": trend & breakout,
        "trend_cross": trend & breakout & credit & volatility,
        "fast_cross": breakout & credit & volatility,
    }
    variants = {"v4": base}
    for label, ready in confirmations.items():
        for size in (.25, .50):
            variants[f"{label}_{int(size*100)}"] = early_weight(
                base, ready.reindex(base.index).shift().eq(True),
                (price_floor | exit_signal).reindex(base.index).shift().eq(True), size)
    # Exploratory follow-up: require the recovery signal to persist on a second close.
    confirmed = confirmations["trend_cross"]
    stop = (price_floor | exit_signal).reindex(base.index).shift().eq(True)
    variants["two_closes_25"] = early_weight(
        base, (confirmed & confirmed.shift(fill_value=False)).reindex(base.index).shift().eq(True),
        stop, .25,
    )
    result = {"metrics": {}, "latest": {label: float(w.iloc[-1]) for label, w in variants.items()}}
    for cost in (25, 50):
        for period, (start, end) in {
            "full": ("2014-10-31", args.end), "2014_2019": ("2014-10-31", "2019-12-31"),
            "2020_2022": ("2020-01-01", "2022-12-31"), "2023_onward": ("2023-01-01", args.end),
            "2026": ("2026-01-01", args.end),
        }.items():
            prices = history.loc[start:end]
            reference = base.loc[prices.index]
            baseline, _ = replay(prices, reference, cost)
            metrics = result["metrics"].setdefault(str(cost), {}).setdefault(period, {})
            for label, weights in variants.items():
                target = weights.loc[prices.index]
                frame, fills = replay(prices, target, cost)
                stats = path_metrics(frame, fills)
                if label != "v4":
                    stats["pairs"] = episodes(prices, reference, target, baseline, frame, added_exposure=True)
                    for trade in stats["pairs"]["trades"]:
                        index = base.index.get_loc(trade["date"])
                        resumes = np.flatnonzero(base.iloc[index:].eq(1))
                        trade["lead_sessions"] = int(resumes[0]) if len(resumes) else None
                stats["max_drawdown_date"] = str((frame.nav / frame.nav.cummax()).idxmin().date())
                metrics[label] = stats
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
