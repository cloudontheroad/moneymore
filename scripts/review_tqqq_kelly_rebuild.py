"""Replay the clean-sheet Kelly allocator without importing any legacy signals into it."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from review_tqqq_kelly import baseline_weights
from review_tqqq_macro_defense import load_history, risk_metrics
from review_tqqq_swing import START, replay
from moneymore.tqqq_kelly import choose_allocation, scenario_book

MODES = {
    "pooled_weekly": ("pooled", "W-FRI", False, False),
    "conditional_weekly": ("conditional", "W-FRI", False, False),
    "constrained_weekly": ("constrained", "W-FRI", False, False),
    "robust_weekly": ("robust", "W-FRI", False, False),
    "robust_cash_override": ("robust", "W-FRI", True, False),
    "binary_robust_weekly": ("binary_robust", "W-FRI", False, False),
    "volatility_scaled_weekly": ("robust", "W-FRI", False, True),
    "volatility_scaled_monthly": ("robust", "M", False, True),
}


def trade_to_target(cash, shares, price, target, fee):
    equity = cash + shares * price
    difference = equity * target - shares * price
    trade = difference / (1 + np.sign(difference) * target * fee)
    charges = abs(trade) * fee
    cash -= trade + charges
    shares += trade / price
    np.testing.assert_allclose(shares * price + cash, equity - charges, atol=1e-12)
    np.testing.assert_allclose(shares * price / (shares * price + cash), target, atol=1e-12)
    return cash, shares, abs(trade) / equity


def standalone_replay(prices, scenarios, bps, mode, cadence, cash_override, volatility_scaled):
    periods = prices.index.to_period(cadence)
    opens, closes = prices.tqqqOpen.to_numpy(), prices.tqqqClose.to_numpy()
    cash, shares = 1.0, 0.0
    target = 0.0
    fills = 0
    rows = []
    log = []
    fee = bps / 10_000
    for i, date in enumerate(prices.index):
        prior_price = closes[max(i - 1, 0)]
        current = np.clip(shares * prior_price / (cash + shares * prior_price), 0.0, 1.0)
        scheduled = i == 0 or periods[i] != periods[i - 1]
        turnover = 0.0
        open_weight = shares * opens[i] / (cash + shares * opens[i])
        if scheduled or (cash_override and current > 0):
            decision = choose_allocation(scenarios[date], current, bps, mode, volatility_scaled=volatility_scaled)
            if scheduled or (decision.weight == 0 and not decision.keep):
                if not decision.keep and abs(decision.weight - current) > 1e-9:
                    cash, shares, turnover = trade_to_target(cash, shares, opens[i], decision.weight, fee)
                    target = decision.weight
                    fills += turnover > 1e-9
                    log.append({
                        "date": str(date.date()), "prior_weight": float(current), "new_target": target,
                        "expected_log_growth": decision.expected_log_growth, "risk_moment": decision.risk_moment,
                        "scenarios": scenarios[date].count, "effective_neighbors": scenarios[date].effective_count,
                        "intraweek_cash_override": not scheduled,
                    })
        marked_open = cash + shares * opens[i]
        invested = shares * opens[i] / marked_open
        rows.append((cash + shares * closes[i], invested, invested - open_weight, turnover, target))
    frame = pd.DataFrame(rows, index=prices.index, columns=["nav", "weight", "weight_change", "turnover", "target"])
    return frame, int(fills), log


def verify_causality(market, scenarios):
    for cutoff in ("2015-08-24", "2016-06-27", "2020-03-16", "2022-10-13", "2025-04-09"):
        prefix = scenario_book(market.loc[:cutoff])
        for date in list(prefix)[-5:]:
            a, b = prefix[date], scenarios[date]
            np.testing.assert_array_equal(a.gross, b.gross)
            np.testing.assert_allclose(a.volatility_scaled, b.volatility_scaled, atol=1e-12)
            np.testing.assert_allclose(a.reference, b.reference, atol=1e-12)
            np.testing.assert_allclose(a.adverse, b.adverse, atol=1e-12)


def enrich(frame, target, fills, prices):
    metrics = risk_metrics(frame, target, fills, prices, 0)
    metrics["turnover"] = float(frame.turnover.sum())
    metrics["target_full_fraction"] = float(target.eq(1).mean())
    metrics["subperiods"] = {}
    for label, a, b in (("2013–2019", "2013", "2019"), ("2020–2022", "2020", "2022"), ("2023+", "2023", None)):
        nav = frame.nav.loc[a:b]
        if nav.empty:
            continue
        start = frame.index.get_loc(nav.index[0])
        initial = frame.nav.iloc[start - 1] if start else 1.0
        path = np.r_[initial, nav.to_numpy()]
        metrics["subperiods"][label] = {
            "cagr": float((path[-1] / path[0]) ** (252 / len(nav)) - 1),
            "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
        }
    return metrics


def matched_quarterly_weight(prices, average_weight):
    """Match actual exposure after drift, for historical attribution only."""
    relative_price = prices.tqqqOpen / prices.tqqqOpen.groupby(prices.index.to_period("Q")).transform("first")
    lower, upper = 0.0, 1.0
    for _ in range(40):
        target = (lower + upper) / 2
        invested = target * relative_price
        if (invested / (1 - target + invested)).mean() < average_weight:
            lower = target
        else:
            upper = target
    return (lower + upper) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    history = load_history(args.end)
    market = history[["qqqClose", "tqqqOpen", "tqqqClose", "VIX", "VIX3M"]].rename(columns={
        "qqqClose": "qqq_close", "tqqqOpen": "tqqq_open", "tqqqClose": "tqqq_close",
        "VIX": "vix", "VIX3M": "vix3m",
    })
    scenarios = scenario_book(market)
    verify_causality(market, scenarios)
    prior = baseline_weights(history)
    prices = history.loc[START:]
    hold_target = pd.Series(1.0, index=prices.index)
    quarter_opens = prices.index[~prices.index.to_period("Q").duplicated()]
    results = {}
    for bps in (10, 25, 50):
        group = {}
        for name, target in (("hold", hold_target), ("prior_defense", prior.loc[prices.index])):
            frame, fills = replay(prices, target, bps)
            group[name] = enrich(frame, target, fills, prices)
        for name, settings in MODES.items():
            frame, fills, trades = standalone_replay(prices, scenarios, bps, *settings)
            group[name] = {**enrich(frame, frame.target, fills, prices), "trades": trades}
        # Descriptive comparator only. Full-sample average exposure is deliberately
        # used for matching, never to decide an order in any Kelly candidate.
        for name in ("robust_weekly", "volatility_scaled_monthly"):
            weight = group[name]["average_tqqq_weight"]
            target = pd.Series(matched_quarterly_weight(prices, weight), index=prices.index)
            static, fills = replay(prices, target, bps, rebalance_on=quarter_opens)
            np.testing.assert_allclose(static.weight.mean(), weight, atol=1e-10)
            group[f"matched_{name}_static"] = enrich(static, target, fills, prices)
        results[bps] = group
    print(json.dumps({
        "start": START, "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
