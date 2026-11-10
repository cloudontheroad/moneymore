"""Compare recovery signals while fixing every TQQQ reduction date."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from review_tqqq_external import daily_prices
from review_tqqq_swing import COSTS, START, assess, replay
from review_tqqq_swing_reentry import opening_weights, relative_effect
from review_tqqq_switch_cost import leverage_rules
from moneymore.leveraged_etf import YahooDailyClient

MAX_WAIT = 10
FORWARD = 20


def recovery_signals(history):
    qqq = history.qqqClose
    equal = history.QQQE
    relative = equal / qqq
    term = history.VIX / history.VIX3M
    # Signals at the prior completed close for trading at the current open.
    price = qqq.gt(qqq.shift().rolling(10).max())
    breadth = equal.gt(equal.rolling(20).mean()) & relative.gt(relative.shift(5))
    easing = term.lt(term.shift(5)) & history.VIX.lt(history.VIX.shift(5))
    return pd.DataFrame({"price": price, "breadth": breadth, "easing": easing}).shift(1, fill_value=False)


def defense_windows(parent):
    target = parent.to_numpy()
    reductions = np.flatnonzero((target == 0) & parent.shift(fill_value=0).eq(1).to_numpy())
    windows = []
    for sell, end in zip(reductions, [*reductions[1:], len(parent)]):
        risk_on = np.flatnonzero(target[sell:end] == 1)
        baseline_buy = sell + risk_on[0] if len(risk_on) else end
        windows.append((sell, baseline_buy, end))
    return windows


def recovery_paths(parent, signals):
    windows = defense_windows(parent)
    variants = {
        "early_price": (signals.price, True),
        "early_breadth": (signals.price & signals.breadth, True),
        "early_easing": (signals.price & signals.easing, True),
        "early_both": (signals.price & signals.breadth & signals.easing, True),
        "wait_breadth": (signals.breadth, False),
        "wait_easing": (signals.easing, False),
        "wait_both": (signals.breadth & signals.easing, False),
    }
    for name, (confirmed, early) in variants.items():
        path = parent.copy()
        buys = []
        for sell, baseline_buy, end in windows:
            if early:
                first, last, fallback = sell + 1, baseline_buy, baseline_buy
            else:
                first, last = baseline_buy, min(baseline_buy + MAX_WAIT, end)
                fallback = last
            eligible = np.flatnonzero(confirmed.iloc[first:last].to_numpy())
            buy = first + eligible[0] if len(eligible) else fallback
            path.iloc[sell:end] = 0.0
            path.iloc[buy:end] = 1.0
            if buy != baseline_buy:
                buys.append({
                    "sell": str(parent.index[sell].date()),
                    "baseline_buy": str(parent.index[baseline_buy].date()) if baseline_buy < end else None,
                    "candidate_buy": str(parent.index[buy].date()) if buy < end else None,
                    "session_difference": int(buy - baseline_buy),
                })
        # An early fill may only happen after the original sell. A failed wait can
        # omit a round trip, but may never add or move a sale.
        actual_sales = np.flatnonzero(path.eq(0).to_numpy() & path.shift(fill_value=0).eq(1).to_numpy())
        if not set(actual_sales).issubset(sell for sell, _, _ in windows):
            raise AssertionError("A recovery rule changed the reduction schedule")
        yield name, path, buys


def price_event_study(parent, signals, prices):
    """Condition on the same first price breakout per parent defense window."""
    events = []
    for sell, baseline_buy, _ in defense_windows(parent):
        breakouts = np.flatnonzero(signals.price.iloc[sell + 1:baseline_buy].to_numpy())
        if not len(breakouts):
            continue
        first = sell + 1 + breakouts[0]
        end = first + FORWARD
        if end >= len(prices):
            continue
        tqqq = prices.tqqqOpen.iloc[end] / prices.tqqqOpen.iloc[first]
        qqq = prices.qqqOpen.iloc[end] / prices.qqqOpen.iloc[first]
        gross = tqqq / qqq
        events.append({
            "date": str(prices.index[first].date()),
            "breadth": bool(signals.breadth.iloc[first]),
            "easing": bool(signals.easing.iloc[first]),
            "gross_relative": float(gross - 1),
            "net_relative_at_25bps": float(gross * ((1 - 0.0025) / (1 + 0.0025)) ** 2 - 1),
        })
    groups = {}
    for name, condition in (
        ("all", lambda row: True),
        ("breadth", lambda row: row["breadth"]),
        ("no_breadth", lambda row: not row["breadth"]),
        ("easing", lambda row: row["easing"]),
        ("no_easing", lambda row: not row["easing"]),
        ("both", lambda row: row["breadth"] and row["easing"]),
    ):
        values = [row["net_relative_at_25bps"] for row in events if condition(row)]
        groups[name] = {"count": len(values), "wins": sum(value > 0 for value in values),
                        "mean": float(np.mean(values)) if values else None}
    return {"events": events, "groups": groups}


def load_history(end):
    def volatility(symbol):
        frame = YahooDailyClient().history(symbol, "2010-02-11", end)
        return pd.Series(frame.close.to_numpy(), index=pd.DatetimeIndex(frame.trade_date))

    with ThreadPoolExecutor(max_workers=5) as pool:
        etfs = {symbol: pool.submit(daily_prices, symbol, end) for symbol in ("QQQ", "TQQQ", "QQQE")}
        indices = {symbol: pool.submit(volatility, "^" + symbol) for symbol in ("VIX", "VIX3M")}
        history = pd.concat([
            etfs[symbol].result()[["Open", "Close"]].add_prefix(symbol.lower()) for symbol in ("QQQ", "TQQQ")
        ], axis=1, join="inner")
        history["QQQE"] = etfs["QQQE"].result().Close
        for symbol, future in indices.items():
            history[symbol] = future.result()
    if history.loc[START:].isna().any().any():
        raise ValueError(f"Incomplete common history: {history.loc[START:].isna().sum().to_dict()}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    history = load_history(args.end)
    signals = recovery_signals(history)
    parent = opening_weights(history.qqqClose)["early_exit"]
    for cutoff in ("2018-12-24", "2020-03-16", "2022-10-13", "2025-04-09"):
        pd.testing.assert_frame_equal(recovery_signals(history.loc[:cutoff]), signals.loc[:cutoff])
    prices = history.loc[START:]
    parent = parent.loc[prices.index]
    signals = signals.loc[prices.index]
    variants = dict((name, (target, changes)) for name, target, changes in recovery_paths(parent, signals))
    # Truncating the last episode must not change any earlier order.
    for cutoff in ("2018-12-24", "2020-03-16", "2022-10-13", "2025-04-09"):
        for name, path, _ in recovery_paths(parent.loc[:cutoff], signals.loc[:cutoff]):
            pd.testing.assert_series_equal(path, variants[name][0].loc[:cutoff])

    control = ((leverage_rules(history.qqqClose).buffer_1pct - 1) / 2).shift().loc[prices.index]
    results = {}
    for cost in COSTS:
        hold, _ = replay(prices, pd.Series(1.0, index=prices.index), cost)
        base, base_fills = replay(prices, parent, cost, alternative="qqq")
        buffer, buffer_fills = replay(prices, control, cost, alternative="qqq")
        results[cost] = {
            "hold": assess(hold, 1, hold, prices.tqqqOpen),
            "buffer": assess(buffer, buffer_fills, hold, prices.tqqqOpen),
            "parent": assess(base, base_fills, hold, prices.tqqqOpen),
        }
        for name, (target, changes) in variants.items():
            run, fills = replay(prices, target, cost, alternative="qqq")
            results[cost][name] = {
                **assess(run, fills, hold, prices.tqqqOpen),
                "vs_parent": relative_effect(run, base),
                "changes": changes,
            }
    print(json.dumps({
        "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "defense_episodes": len(defense_windows(parent)),
        "price_event_study": price_event_study(parent, signals, prices),
        "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
