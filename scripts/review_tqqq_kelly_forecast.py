"""Walk-forward forecast ablations with the existing monthly Kelly optimizer frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from moneymore.tqqq_kelly import Scenarios, market_features, scenario_book
from review_tqqq_kelly import baseline_weights
from review_tqqq_kelly_rebuild import enrich, matched_quarterly_weight, standalone_replay
from review_tqqq_macro_defense import load_history
from review_tqqq_swing import START, WAVES, replay

MODE = ("robust", "M", False, True)
FEATURES = {
    "price": ("medium_trend", "near_peak", "short_shock"),
    "price_vol": ("medium_trend", "near_peak", "short_shock", "vol_term", "vol_acceleration"),
    "price_credit": ("medium_trend", "near_peak", "short_shock", "credit_momentum"),
    "price_vol_credit": ("medium_trend", "near_peak", "short_shock", "vol_term",
                         "vol_acceleration", "credit_momentum"),
}


def features(market):
    log = np.log(market.qqq_close)
    returns = log.diff()
    slow_vol = returns.rolling(63).std()
    credit = np.log(market.hyg / market.iei)
    return market_features(market).assign(
        short_shock=log.diff(5) / (slow_vol * np.sqrt(5)),
        vol_acceleration=np.log(returns.rolling(20).std() / slow_vol),
        credit_momentum=credit.diff(21) / (credit.diff().rolling(63).std() * np.sqrt(21)),
    )


def standardize(past, current):
    center = np.median(past, axis=0)
    scale = np.maximum(np.subtract(*np.quantile(past, (0.75, 0.25), axis=0)), 1e-8)
    x = np.clip((past - center) / scale, -4, 4)
    query = np.clip((current - center) / scale, -4, 4)
    return x, query


def learned_probabilities(past, current, classes):
    x, query = standardize(past, current)
    # Fixed regularization, and another year-equivalent shrink toward historical frequencies.
    model = LogisticRegression(C=1.0, max_iter=500)
    model.fit(x, classes)
    probabilities = model.predict_proba(query[None])[0]
    reference = np.zeros(len(classes))
    count = len(classes)
    for label, prob in zip(model.classes_, probabilities, strict=True):
        same = classes == label
        reference[same] = (count * prob / same.sum() + 12 / count) / (count + 12)
    np.testing.assert_allclose(reference.sum(), 1)
    return reference


def return_distribution(past, current, normalized_returns, vol, current_vol, days):
    """Predict the return magnitude, then reuse leave-one-out forecast residuals."""
    x, query = standardize(past, current)
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.diag([0.0, *([12.0] * x.shape[1])])
    inverse = np.linalg.solve(design.T @ design + penalty, design.T)
    coefficients = inverse @ normalized_returns
    leverage = np.einsum("ij,ji->i", design, inverse)
    loo_residual = (normalized_returns - design @ coefficients) / (1 - leverage)
    projection = np.r_[1, query] @ coefficients + loo_residual
    raw = np.exp(projection * vol * np.sqrt(days))
    scaled = np.exp(np.log(raw) * np.clip(current_vol / vol, 0.5, 2))
    probability = np.full(len(raw), 1 / len(raw))
    tail = raw <= np.quantile(raw, 0.10)
    adverse = 0.95 * probability + 0.05 * tail / tail.sum()
    return Scenarios(raw, scaled, probability, adverse, len(raw), float(1 / (probability @ probability)))


def forecast_books(market, original):
    signals = features(market)
    boundaries = np.flatnonzero(~market.index.to_period("M").duplicated())
    start, end = boundaries[:-1], boundaries[1:]
    valid = (start > 0) & np.isfinite(market_features(market).iloc[np.maximum(start - 1, 0)]).all(axis=1)
    start, end = start[valid], end[valid]
    opens, closes = market.tqqq_open.to_numpy(), market.tqqq_close.to_numpy()
    returns = np.log(market.tqqq_close).diff()
    volatility = pd.concat([returns.rolling(n).std() for n in (20, 60)], axis=1).max(axis=1).to_numpy()
    # Scale paths by what was known at each historical decision, including the eventual label.
    path_low = np.array([
        np.log(min(opens[i:j + 1].min(), closes[i:j].min()) / opens[i]) /
        (volatility[i - 1] * np.sqrt(j - i))
        for i, j in zip(start, end, strict=True)
    ])
    week_low = np.array([
        np.log(min(opens[i:i + 6].min(), closes[i:i + 5].min()) / opens[i]) /
        (volatility[i - 1] * np.sqrt(5))
        for i in start
    ])
    gross = opens[end] / opens[start]
    complete_months = dict(zip(start, end, strict=True))
    books = {name: original.copy() for name in (*FEATURES, "price_ridge")}
    price_features = signals[list(FEATURES["price"])].to_numpy()
    forecasts = []
    for i in boundaries:
        date = market.index[i]
        if date not in original:
            continue
        count = np.searchsorted(end, i)
        np.testing.assert_array_equal(gross[:count], original[date].gross)
        if not np.isfinite(signals.iloc[i - 1].to_numpy()).all():
            continue
        cutoff = np.quantile(path_low[:count], 0.20)
        tail = path_low[:count] <= cutoff
        week_cutoff = np.quantile(week_low[:count], 0.20)
        weekly_tail = week_low[:count] <= week_cutoff
        labels = np.where(tail, 0, np.where(gross[:count] < 1, 1, 2))
        score = {"date": str(date.date()), "observed_tail": None, "probabilities": {
            "old": float(original[date].reference @ tail),
            "historical_frequency": float(tail.mean()),
        }, "observed_week_tail": None, "week_probabilities": {
            "old": float(original[date].reference @ weekly_tail),
            "historical_frequency": float(weekly_tail.mean()),
        }}
        if i in complete_months:
            j = complete_months[i]
            adverse = np.log(min(opens[i:j + 1].min(), closes[i:j].min()) / opens[i])
            score["observed_tail"] = bool(adverse / (volatility[i - 1] * np.sqrt(j - i)) <= cutoff)
            weekly_adverse = np.log(min(opens[i:i + 6].min(), closes[i:i + 5].min()) / opens[i])
            score["observed_week_tail"] = bool(weekly_adverse / (volatility[i - 1] * np.sqrt(5)) <= week_cutoff)
        for name, cols in FEATURES.items():
            values = signals[list(cols)].to_numpy()
            past = values[start[:count] - 1]
            known = np.isfinite(past).all(axis=1)
            if known.sum() < 18 or len(np.unique(labels[known])) < 2:
                continue
            probability = np.zeros(count)
            probability[known] = learned_probabilities(past[known], values[i - 1], labels[known])
            score["probabilities"][name] = float(probability @ tail)
            score["week_probabilities"][name] = float(probability @ weekly_tail)
            books[name][date] = replace(
                original[date], reference=probability, effective_count=float(1 / (probability @ probability)),
            )
        known_starts = start[:count]
        vol = volatility[known_starts - 1]
        days = end[:count] - known_starts
        books["price_ridge"][date] = return_distribution(
            price_features[known_starts - 1], price_features[i - 1],
            np.log(gross[:count]) / (vol * np.sqrt(days)), vol, volatility[i - 1], days,
        )
        forecasts.append(score)
    return books, forecasts


def attribute(prices, candidate, reference, bps):
    move = prices.tqqqOpen.shift(-1) / prices.tqqqOpen - 1
    move.iloc[-1] = prices.tqqqClose.iloc[-1] / prices.tqqqOpen.iloc[-1] - 1
    difference = candidate.weight - reference.weight
    allocation = np.log1p(candidate.weight * move) - np.log1p(reference.weight * move)
    fee = bps / 10_000
    costs = np.log1p(-fee * candidate.turnover) - np.log1p(-fee * reference.turnover)
    active = allocation + costs
    np.testing.assert_allclose(active.sum(), np.log(candidate.nav.iloc[-1] / reference.nav.iloc[-1]), atol=1e-10)
    reduced = difference.lt(-1e-10)
    intervals = reduced.ne(reduced.shift(fill_value=False)).cumsum()
    defensive = active[reduced].groupby(intervals[reduced]).sum()
    return {
        "less_exposure_avoided_declines_log": float(allocation.loc[(difference < 0) & (move < 0)].sum()),
        "less_exposure_missed_gains_log": float(allocation.loc[(difference < 0) & (move > 0)].sum()),
        "more_exposure_net_log": float(allocation.loc[difference > 0].sum()),
        "cost_difference_log": float(costs.sum()),
        "net_log": float(active.sum()),
        "net_relative_without_best_defense": float(np.expm1(active.sum() - max([0.0, *defensive.to_numpy()]))),
        "waves": {name: float(active.loc[a:b].sum()) for name, (a, b) in {
            **WAVES, "2014–2016": ("2014-09-01", "2016-12-31"),
        }.items()},
    }


def exits(prices, frame):
    result = []
    for i in np.flatnonzero(frame.weight_change.lt(-0.20).to_numpy()):
        if i + 21 >= len(prices):
            continue
        buy = np.flatnonzero(frame.weight_change.iloc[i + 1:].gt(0.20).to_numpy())
        entry = i + 1 + buy[0] if len(buy) else None
        history = prices.tqqqOpen.iloc[max(0, i - 63):i + 1]
        forward = prices.tqqqOpen.iloc[i:min(i + 22, entry + 1 if entry is not None else len(prices))]
        result.append({
            "date": str(prices.index[i].date()), "target": float(frame.target.iloc[i]),
            "drawdown_before_sell": float(history.iloc[-1] / history.max() - 1),
            "drop_after_sell": float(forward.min() / forward.iloc[0] - 1),
            "first_reentry": str(prices.index[entry].date()) if entry is not None else None,
            "first_reentry_vs_sell": float(prices.tqqqOpen.iloc[entry] / forward.iloc[0] - 1)
            if entry is not None else None,
        })
    return result


def score_forecasts(rows):
    common = [r for r in rows if r["date"] >= START and r["observed_tail"] is not None
              and all(name in r["probabilities"] for name in FEATURES)]
    return {
        "months": len(common),
        **{
            horizon: {name: float(np.mean([(row[prob][name] - row[actual]) ** 2 for row in common]))
                      for name in ("historical_frequency", "old", *FEATURES)}
            for horizon, prob, actual in (
                ("month", "probabilities", "observed_tail"), ("first_week", "week_probabilities", "observed_week_tail"),
            )
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
    books, forecasts = forecast_books(market, original)
    for cutoff in ("2016-06-27", "2020-03-16", "2022-10-13", "2025-04-09"):
        prefix = market.loc[:cutoff]
        prior, _ = forecast_books(prefix, scenario_book(prefix))
        date = prefix.index[prefix.index.to_period("M") == prefix.index[-1].to_period("M")][0]
        for name in books:
            np.testing.assert_allclose(prior[name][date].reference, books[name][date].reference, atol=1e-12)
            np.testing.assert_allclose(prior[name][date].volatility_scaled,
                                       books[name][date].volatility_scaled, atol=1e-12)

    prices = history.loc[START:]
    quarters = prices.index[~prices.index.to_period("Q").duplicated()]
    results = {}
    for bps in (10, 25, 50):
        group = {}
        for name, targets in (("hold", pd.Series(1.0, index=prices.index)),
                              ("prior_defense", baseline_weights(history).loc[prices.index])):
            frame, fills = replay(prices, targets, bps)
            group[name] = enrich(frame, targets, fills, prices)
        old, _, _ = standalone_replay(prices, original, bps, *MODE)
        for name, book in {"old": original, **books}.items():
            frame, fills, trades = standalone_replay(prices, book, bps, *MODE)
            average = frame.weight.mean()
            target = pd.Series(matched_quarterly_weight(prices, average), index=prices.index)
            fixed, fixed_fills = replay(prices, target, bps, rebalance_on=quarters)
            np.testing.assert_allclose(fixed.weight.mean(), average, atol=1e-10)
            group[name] = {
                **enrich(frame, frame.target, fills, prices),
                "matched": enrich(fixed, target, fixed_fills, prices),
                "vs_old": attribute(prices, frame, old, bps),
                "vs_matched": attribute(prices, frame, fixed, bps),
                "exit_ledger": exits(prices, frame), "trades": trades,
            }
        results[bps] = group
    print(json.dumps({
        "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "forecast_brier": score_forecasts(forecasts), "monthly_forecasts": forecasts, "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
