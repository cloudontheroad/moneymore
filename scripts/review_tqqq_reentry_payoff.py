"""Forecast executable TQQQ recovery payoffs and replay fixed-exit entry policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from review_tqqq_kelly_forecast import standardize
from review_tqqq_kelly_rebuild import matched_quarterly_weight
from review_tqqq_kelly_sell_edge import blocked_interval
from review_tqqq_sharpe import daily_returns, paired_interval, sharpe
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics
from review_tqqq_turning_points import price_signals
from review_tqqq_weekly_cycle import CROSS, PRICE, allocation, execution_state, labels, train, weekly_features

REBOUND = ("one_week", "recovery_from_low", "distance10")


def inputs(history, context, alternate):
    data, state = weekly_features(history, context, alternate)
    price = np.log(state.close)
    data = data.assign(
        one_week=price.diff(),
        recovery_from_low=price - price.rolling(4).min(),
        distance10=np.log(state.close / state.sma10),
    )
    return data, state


def forward_payoffs(history, weeks, horizon):
    """Settle from the post-signal open to the open four weekly decisions later."""
    offsets = history.index.get_indexer(weeks) + 1
    opens = history.tqqqOpen.to_numpy()
    closes = history.tqqqClose.to_numpy()
    rows = []
    for i, j in zip(offsets[:-horizon], offsets[horizon:], strict=True):
        if j >= len(history):
            break
        floor = min(opens[i:j + 1].min(), closes[i:j].min())
        rows.append((np.log(opens[j] / opens[i]), -np.log(floor / opens[i]), history.index[j]))
    result = pd.DataFrame(rows, index=weeks[:len(rows)], columns=["terminal", "loss", "settled"])
    return result.reindex(weeks)


def forecasts(data, outcome):
    # Use the same completed post-pullback weeks for both feature groups.
    valid = data[[*PRICE, *REBOUND, *CROSS]].notna().all(axis=1)
    context = data.below10_share.gt(0)
    result = pd.DataFrame(index=data.index)
    columns = ("mean_terminal", "mean_loss", "median_loss", "tail_loss",
               "price_terminal", "price_loss", "cross_terminal", "cross_loss")
    for name in columns:
        result[name] = np.nan
    for date in data.index[valid & context]:
        past = valid & context & outcome.settled.le(date)
        observed = outcome.loc[past, ["terminal", "loss"]].to_numpy()
        if len(observed) < 52:
            continue
        mean, loss = observed.mean(axis=0)
        median_loss, tail_loss = np.quantile(observed[:, 1], (0.50, 0.80))
        result.loc[date, ["mean_terminal", "mean_loss", "median_loss", "tail_loss"]] = (
            mean, loss, median_loss, tail_loss,
        )
        for name, names in (("price", [*PRICE, *REBOUND]), ("cross", [*PRICE, *REBOUND, *CROSS])):
            x, query = standardize(data.loc[past, names].to_numpy(), data.loc[date, names].to_numpy())
            design = np.column_stack((np.ones(len(x)), x))
            penalty = np.diag([0.0, *([26.0] * x.shape[1])])
            coeff = np.linalg.solve(design.T @ design + penalty, design.T @ observed)
            expected, risk = np.r_[1, query] @ coeff
            result.loc[date, [f"{name}_terminal", f"{name}_loss"]] = expected, max(0, risk)
    return result


def elevated_loss(loss):
    return loss.gt(loss.shift().rolling(104, min_periods=26).quantile(0.80))


def diagnostics(projected, realized, bps, start):
    cost = np.log((1 + bps / 10_000) / (1 - bps / 10_000))
    valid = projected.notna().all(axis=1) & realized.settled.notna() & (projected.index >= start)
    predicted, actual = projected.loc[valid], realized.loc[valid]
    tail = actual.loss.ge(predicted.tail_loss)
    output = {}
    for name in ("mean", "price", "cross"):
        terminal, loss = predicted[f"{name}_terminal"], predicted[f"{name}_loss"]
        selected = terminal.gt(cost)
        low_risk = selected & loss.le(predicted.median_loss)
        ranked_high = elevated_loss(projected[f"{name}_loss"]).loc[valid]
        output[name] = {
            "weeks": len(actual), "terminal_mae": float((terminal - actual.terminal).abs().mean()),
            "downside_mae": float((loss - actual.loss).abs().mean()),
            "tail_auc": float(roc_auc_score(tail, loss)),
            "net_positive": int(actual.terminal.gt(cost).sum()), "tail_weeks": int(tail.sum()),
            "positive_forecast_weeks": int(selected.sum()),
            "positive_forecast_actual_mean": float((actual.terminal[selected] - cost).mean()) if selected.any() else None,
            "low_risk_weeks": int(low_risk.sum()), "low_risk_tail_weeks": int(tail[low_risk].sum()),
            "low_risk_actual_mean": float((actual.terminal[low_risk] - cost).mean()) if low_risk.any() else None,
            "ranked_high_weeks": int(ranked_high.sum()),
            "ranked_high_tail_weeks": int(tail[ranked_high].sum()),
        }
    return output


def weekly_recommendation(prediction, model, variant, bps):
    cost = np.log((1 + bps / 10_000) / (1 - bps / 10_000))
    terminal, loss = prediction[f"{model}_terminal"], prediction[f"{model}_loss"]
    profitable = terminal.gt(cost)
    safe = loss.le(prediction.median_loss)
    if variant == "return":
        target = profitable.astype(float)
    elif variant == "both":
        target = (profitable & safe).astype(float)
    else:
        target = profitable.astype(float) * (0.5 + 0.5 * safe)
    # Outside the fitted pullback context, defer to the original buy signal.
    return target.where(terminal.notna(), 1.0)


def only_reentry(base, recommendation, *, clock=None):
    target = []
    level = 1.0
    waiting_since = None
    weeks = clock.to_list() if clock is not None else [None] * len(base)
    for active, desired, week in zip(base.to_numpy(), recommendation.to_numpy(), weeks, strict=True):
        if not active:
            level = 0.0
            waiting_since = None
        else:
            if clock is not None and level == 0 and pd.notna(week):
                if waiting_since is None:
                    waiting_since = week
                elif week != waiting_since:
                    desired = 1.0
            level = max(level, desired if np.isfinite(desired) else 1.0)
        target.append(level)
    return pd.Series(target, index=base.index)


def pairs(prices, base, target, reference, frame):
    """Reconcile every intervention to the full net NAV relative to immediate buying."""
    difference = np.log(frame.nav / reference.nav)
    base_days = base.eq(1)
    starts = np.flatnonzero((base_days & ~base_days.shift(fill_value=False)).to_numpy())
    ends = np.flatnonzero((~base_days & base_days.shift(fill_value=False)).to_numpy())
    records = []
    for i in starts:
        remaining = ends[ends > i]
        j = remaining[0] if len(remaining) else len(base) - 1
        tranche = target.iloc[i:j + int(not len(remaining))]
        if tranche.eq(1).all():
            continue
        first = tranche[tranche.gt(0)].index
        full = tranche[tranche.eq(1)].index
        previous = difference.iloc[i - 1] if i else 0
        records.append({
            "date": str(prices.index[i].date()), "end": str(prices.index[j].date()),
            "first_buy": str(first[0].date()) if len(first) else None,
            "full_buy": str(full[0].date()) if len(full) else None,
            "initial_target": float(tranche.iloc[0]),
            "delta": float(difference.iloc[j] - previous),
        })
    np.testing.assert_allclose(sum(row["delta"] for row in records), difference.iloc[-1], atol=1e-9)
    delta = np.array([row["delta"] for row in records])
    return {
        "changed": len(delta), "better": int((delta > 1e-9).sum()), "worse": int((delta < -1e-9).sum()),
        "mean_delta": float(delta.mean()) if len(delta) else None,
        "mean_delta_interval": blocked_interval(delta),
        "terminal_edge_without_best": float(np.expm1(delta.sum() - max(0, delta.max()))) if len(delta) else 0,
        "events": records,
    }


def replay_policies(history, predictions, prior, bps, start):
    base = price_signals(history).trend
    prices = history.loc[start:]
    rules = {"sma175": base, "prior_weekly": prior}
    weekly_clock = pd.Series(predictions.index, index=predictions.index)
    clock = execution_state(history, weekly_clock)
    one_week_wait = pd.Series(0.0, index=predictions.index)
    rules["fixed_one_week"] = only_reentry(base, execution_state(history, one_week_wait), clock=clock)
    for model in ("mean", "price", "cross"):
        for variant in ("return", "both", "staged"):
            weekly = weekly_recommendation(predictions, model, variant, bps)
            rules[f"{model}_{variant}"] = only_reentry(base, execution_state(history, weekly))
        if model != "mean":
            loss = predictions[f"{model}_loss"]
            # Exploration added after inspecting the median-loss gate. Rank
            # predictions against prior predictions to account for shrinkage.
            tail_rank = elevated_loss(loss)
            ranked = execution_state(history, (~tail_rank).astype(float))
            rules[f"{model}_tail"] = only_reentry(base, ranked)
            rules[f"{model}_tail_one_week"] = only_reentry(base, ranked, clock=clock)
    reference, _ = replay(prices, base.loc[start:], bps)
    reference_returns = daily_returns(reference.nav)
    quarter = prices.index[~prices.index.to_period("Q").duplicated()]
    output = {}
    for name, target in rules.items():
        target = target.loc[start:]
        assert target[base.loc[start:].eq(0)].eq(0).all()
        frame, fills = replay(prices, target, bps)
        returns = daily_returns(frame.nav)
        weight = matched_quarterly_weight(prices, frame.weight.mean())
        matched, matched_fills = replay(prices, pd.Series(weight, index=prices.index), bps, rebalance_on=quarter)
        output[name] = {
            **path_metrics(frame, fills),
            "same_exposure": path_metrics(matched, matched_fills),
            "sharpe_without_2020_2022": float(sharpe(returns[~prices.index.year.isin([2020, 2022])])),
            "sharpe_interval": paired_interval(returns, reference_returns),
            "pairing": pairs(prices, base.loc[start:], target, reference, frame),
            "latest_target": float(target.iloc[-1]),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    for name in ("history-cache", "context-cache", "alternate-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).loc[:args.end]

    history, context, alternate = map(read, (args.history_cache, args.context_cache, args.alternate_cache))
    data, state = inputs(history, context, alternate)
    old, _ = weekly_features(history, context, alternate)
    previous = train(old, labels(state, 13, 8))
    cutoff = previous[["price", "cross"]].shift().rolling(104, min_periods=52).quantile(0.80)
    ready = cutoff.notna().all(axis=1) & previous[["price", "cross"]].notna().all(axis=1)
    weekly = state[["fast_off", "repair"]].assign(
        ready=ready, price_high=previous.price.gt(cutoff.price), cross_high=previous.cross.gt(cutoff.cross)
    )
    prior = allocation(price_signals(history).trend, execution_state(history, weekly), "price_reentry")
    datasets = {h: forward_payoffs(history, data.index, h) for h in (2, 4, 8)}
    forecast_by_horizon = {h: forecasts(data, outcome) for h, outcome in datasets.items()}
    for date in ("2018-12-21", "2020-03-20", "2022-10-14", "2025-04-04"):
        h, c, a = (frame.loc[:date] for frame in (history, context, alternate))
        prefix, _ = inputs(h, c, a)
        pd.testing.assert_frame_equal(prefix, data.loc[:date])
        pd.testing.assert_frame_equal(forecasts(prefix, forward_payoffs(h, prefix.index, 4)),
                                      forecast_by_horizon[4].loc[:date])
        target = weekly_recommendation(forecast_by_horizon[4], "price", "staged", 25)
        left = only_reentry(price_signals(h).trend, execution_state(h, target.loc[:date]))
        right = only_reentry(price_signals(history).trend, execution_state(history, target))
        pd.testing.assert_series_equal(left, right.loc[:date])
    # Align all reentry variants to the earlier report's common comparison window.
    start = ready[ready].index[0]
    output = {
        h: {
            "forecast": diagnostics(prediction, datasets[h], 25, start),
            "portfolio": {bps: replay_policies(history, prediction, prior, bps, start) for bps in (25, 50)},
            "latest": json.loads(prediction.tail(6).reset_index().to_json(orient="records", date_format="iso")),
        }
        for h, prediction in forecast_by_horizon.items()
    }
    hashes = {name: hashlib.sha256(frame.to_csv().encode()).hexdigest() for name, frame in
              (("market", history), ("context", context), ("alternate", alternate))}
    print(json.dumps({"start": str(start.date()), "input_sha256": hashes, "results": output}, indent=2))


if __name__ == "__main__":
    main()
