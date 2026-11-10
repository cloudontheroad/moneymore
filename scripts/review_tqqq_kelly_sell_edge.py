"""Pair each Kelly cut with holding the same opening position and measure net edge."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from moneymore.tqqq_kelly import choose_allocation, scenario_book
from review_tqqq_kelly_forecast import FEATURES, MODE, features, forecast_books, standardize
from review_tqqq_kelly_rebuild import standalone_replay
from review_tqqq_macro_defense import load_history
from review_tqqq_swing import START

HORIZONS = ("first_week", "month")


def paired_returns(start_weight, target, gross, bps):
    """Compare open-to-open wealth, then restore the hold branch's end weight."""
    fee = bps / 10_000
    hold = 1 + start_weight * (gross - 1)
    trim = 1 + target * (gross - 1)
    exit_fee = fee * (start_weight - target) / (1 - fee * target)
    restore_weight = start_weight * gross / hold
    pre_restore = target * gross / trim
    reentry_fee = fee * (restore_weight - pre_restore) / (1 + fee * restore_weight)
    gross_log = np.log(trim / hold)
    cost_log = np.log1p(-exit_fee) + np.log1p(-reentry_fee)
    return gross_log, cost_log


def ridge_forecast(past, current, net_outcomes):
    x, query = standardize(past, current)
    design = np.column_stack([np.ones(len(past)), x])
    penalty = np.diag([0.0, *([12.0] * x.shape[1])])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ net_outcomes)
    return float(np.r_[1, query] @ coefficients)


def realized_outcomes(dates, opens, i, j, current, target, bps):
    outcomes = {}
    for horizon, end in (("first_week", min(i + 5, j)), ("month", j)):
        gross, cost = paired_returns(current, target, opens[end] / opens[i], bps)
        outcomes[horizon] = {
            "end": str(dates[end].date()), "gross_log": float(gross),
            "cost_log": float(cost), "net_log": float(gross + cost),
        }
    return outcomes


def opportunities(market, original, price_book, reference, bps):
    dates = market.index
    boundaries = np.flatnonzero(~dates.to_period("M").duplicated())
    opens = market.tqqq_open.to_numpy()
    closes = market.tqqq_close.to_numpy()
    feature_table = features(market)
    features_by_group = {key: feature_table[list(FEATURES[key])].to_numpy()
                         for key in ("price", "price_vol_credit")}
    starts, ends = boundaries[:-1], boundaries[1:]
    historical_gross = {
        "first_week": opens[np.minimum(starts + 5, ends)] / opens[starts],
        "month": opens[ends] / opens[starts],
    }
    output = []
    placebo = []
    for i, j in zip(starts, ends, strict=True):
        date = dates[i]
        if date not in reference.index or i == 0:
            continue
        # The reference account's shares/cash are identical in the two hypothetical
        # paths before execution. Forecasts only see the preceding closing weight.
        opening_weight = reference.at[date, "weight"] - reference.at[date, "weight_change"]
        opening_weight = float(np.clip(opening_weight, 0, 1))
        if opening_weight <= 1e-9:
            continue
        relative_open = opens[i] / closes[i - 1]
        close_weight = opening_weight / (opening_weight + relative_open * (1 - opening_weight))
        fixed = realized_outcomes(dates, opens, i, j, opening_weight, opening_weight / 2, bps)
        placebo.append({"date": str(date.date()), "outcomes": fixed})
        for key, book in (("old", original), ("price", price_book)):
            decision = choose_allocation(book[date], close_weight, bps, "robust", volatility_scaled=True)
            if decision.keep or decision.weight >= min(close_weight, opening_weight) - 1e-9:
                continue
            outcomes = realized_outcomes(dates, opens, i, j, opening_weight, decision.weight, bps)
            forecasts = {"first_week": {}, "month": {}}
            complete = ends < i
            # Cost and comparable starting allocation were known before the order.
            # Do not feed the still-unknown opening gap into historical forecasts.
            for horizon, gross in historical_gross.items():
                gross_log, cost_log = paired_returns(close_weight, decision.weight, gross[complete], bps)
                historical_net = gross_log + cost_log
                if len(historical_net):
                    forecasts[horizon]["past_average"] = float(historical_net.mean())
                for group, values in features_by_group.items():
                    historical = values[np.maximum(starts[complete] - 1, 0)]
                    query = values[i - 1]
                    valid = (starts[complete] > 0) & np.isfinite(historical).all(axis=1)
                    if valid.sum() >= 18 and np.isfinite(query).all():
                        forecasts[horizon][group] = ridge_forecast(historical[valid], query, historical_net[valid])
            prior_peak = opens[max(0, i - 63):i + 1].max()
            output.append({
                "date": str(date.date()), "model": key, "starting_weight": opening_weight,
                "target": decision.weight, "past_samples": int(complete.sum()), "forecasts": forecasts,
                "market_drawdown_before_cut": float(opens[i] / prior_peak - 1),
                "market_low_after_cut": float(opens[i:j + 1].min() / opens[i] - 1),
                "outcomes": outcomes, "fixed_size_outcomes": fixed,
            })
    return output, placebo


def blocked_interval(values):
    rng = np.random.default_rng(20261109)
    count = len(values)
    if count < 8:
        return None
    starts = rng.integers(0, count, size=(2000, (count + 3) // 4))
    blocks = (starts[:, :, None] + np.arange(4)) % count
    draws = values[blocks.reshape(2000, -1)[:, :count]].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def aggregate(events, horizon):
    values = np.array([event["outcomes"][horizon]["net_log"] for event in events])
    if not len(values):
        return {"events": 0}
    gross = np.array([row["outcomes"][horizon]["gross_log"] for row in events])
    return {
        "events": len(values), "wins": int((values > 0).sum()),
        "mean_net_log": float(values.mean()),
        "combined_net": float(np.expm1(values.sum())),
        "combined_net_without_best": float(np.expm1(values.sum() - max(0, values.max()))),
        "mean_net_log_interval": blocked_interval(values),
        "mean_avoided_declines_log": float(gross.clip(min=0).mean()),
        "mean_missed_gains_log": float(gross.clip(max=0).mean()),
        "mean_cost_log": float(np.mean([row["outcomes"][horizon]["cost_log"] for row in events])),
        "periods": {
            name: float(np.expm1(sum(row["outcomes"][horizon]["net_log"] for row in events
                                    if start <= row["date"] < end)))
            for name, start, end in (("2013–2019", "2013", "2020"), ("2020–2022", "2020", "2023"),
                                     ("2023+", "2023", "9999"))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    history = load_history(args.end)
    market = history.rename(columns={
        "qqqClose": "qqq_close", "tqqqOpen": "tqqq_open", "tqqqClose": "tqqq_close",
        "VIX": "vix", "VIX3M": "vix3m", "HYG": "hyg", "IEI": "iei",
    })
    original = scenario_book(market)
    books, _ = forecast_books(market, original)
    results = {}
    for bps in (10, 25, 50):
        reference, _, _ = standalone_replay(history.loc[START:], original, bps, *MODE)
        rows, placebo = opportunities(market, original, books["price"], reference, bps)
        if bps == 25:
            # Compare forecasts on a truncated input, keeping only settled events.
            cutoff = "2023-02-15"
            past_market, past_ref = market.loc[:cutoff], reference.loc[:cutoff]
            earlier, _ = opportunities(past_market, original, books["price"], past_ref, bps)
            assert earlier == [row for row in rows if row["outcomes"]["month"]["end"] <= cutoff]
        grouped = {}
        for model in ("old", "price"):
            events = [row for row in rows if row["model"] == model]
            grouped[model] = {
                "all": {horizon: aggregate(events, horizon) for horizon in HORIZONS},
                **{
                    name: {horizon: aggregate(
                        [row for row in events if row["forecasts"][horizon].get(name, 0) > 0], horizon
                    ) for horizon in HORIZONS}
                    for name in ("past_average", "price", "price_vol_credit")
                },
            }
        fixed_size_groups = {
            "all_invested_months": placebo,
            **{model: [{"date": row["date"], "outcomes": row["fixed_size_outcomes"]}
                       for row in rows if row["model"] == model] for model in ("old", "price")},
        }
        results[bps] = {
            "aggregates": grouped,
            "same_cut_size": {
                name: {horizon: aggregate(group, horizon) for horizon in HORIZONS}
                for name, group in fixed_size_groups.items()
            },
            "events": rows,
        }
    print(json.dumps({
        "end": str(market.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
