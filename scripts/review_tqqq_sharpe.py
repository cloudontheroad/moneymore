"""Attribute TQQQ Sharpe and replay conditional volatility sizing at the next open."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_kelly_rebuild import matched_quarterly_weight
from review_tqqq_macro_defense import load_history
from review_tqqq_swing import START, replay
from review_tqqq_swing_round_trips import open_signal, path_metrics, state_targets


def strategies(prices):
    close = prices.qqqClose
    average = close.rolling(175).mean()
    trend = state_targets(open_signal(close.gt(average)), open_signal(close.lt(0.99 * average)))
    hold = pd.Series(1.0, index=prices.index)
    targets = {"hold": hold, "sma175": trend}
    returns = np.log(close).diff()
    measures = {
        "vol": returns.rolling(21).std(),
        "downside": returns.clip(upper=0).pow(2).rolling(21).mean().pow(0.5),
    }
    for name, realized in measures.items():
        past = realized.shift().rolling(756, min_periods=252)
        normal = past.median()
        extreme = past.quantile(0.80)
        ratio = (normal / realized).clip(upper=1).fillna(1)
        conditional = ratio.where(realized.gt(extreme), 1)
        for frequency in ("M", "W-FRI"):
            boundary = ~prices.index.to_period(frequency).duplicated()
            for variant, multiplier in (("conditional", conditional), ("continuous", ratio)):
                applied = multiplier.shift().where(boundary).ffill().fillna(1)
                targets[f"{name}_{frequency}_{variant}"] = trend * applied
    return pd.DataFrame(targets)


def daily_returns(nav):
    return np.r_[nav.iloc[0] - 1, nav.pct_change().dropna()]


def sharpe(returns):
    return np.sqrt(252) * returns.mean(axis=-1) / returns.std(axis=-1, ddof=1)


def paired_interval(candidate, reference):
    rng = np.random.default_rng(20261109)
    length = len(candidate)
    starts = rng.integers(0, length, (2000, (length + 19) // 20))
    blocks = (starts[..., None] + np.arange(20)) % length
    indices = blocks.reshape(len(starts), -1)[:, :length]
    differences = sharpe(candidate[indices]) - sharpe(reference[indices])
    return np.quantile(differences, [0.025, 0.975]).tolist()


def attribution(frame, reference, prices):
    returns = daily_returns(frame.nav)
    market = prices.tqqqClose.pct_change().to_numpy()
    squared = np.square(returns - returns.mean())
    result = {}
    for label, mask in (("bottom_5pct", market <= np.nanquantile(market, 0.05)),
                        ("top_5pct", market >= np.nanquantile(market, 0.95))):
        result[label] = {
            "days": int(mask.sum()),
            "share_of_variance": float(squared[mask].sum() / squared.sum()),
            "log_return": float(np.log1p(returns[mask]).sum()),
            "avg_open_weight": float(frame.weight.to_numpy()[mask].mean()),
        }
    # Opening exposure precedes the current session. The overnight leg uses
    # the previous close's holdings, reconstructed from the last open weight.
    open_to_close = prices.tqqqClose / prices.tqqqOpen
    close_weight = frame.weight * open_to_close / (1 + frame.weight * (open_to_close - 1))
    overnight = close_weight.shift().fillna(0) * (prices.tqqqOpen / prices.tqqqClose.shift() - 1).fillna(0)
    session = frame.weight * (open_to_close - 1)
    daily_log = np.log1p(returns)
    overnight_log, session_log = np.log1p(overnight), np.log1p(session)
    costs = daily_log - overnight_log - session_log
    np.testing.assert_array_less(costs, 1e-10)
    result.update({
        "annual_mean": float(returns.mean() * 252),
        "annual_volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "overnight_log": float(overnight_log.sum()),
        "session_log": float(session_log.sum()),
        "cost_log": float(costs.sum()),
        "sharpe_without_2022": float(sharpe(returns[prices.index.year != 2022])),
        "sharpe_delta_interval": paired_interval(returns, daily_returns(reference.nav)),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--history-cache", type=Path)
    args = parser.parse_args()
    history = (pd.read_csv(args.history_cache, index_col=0, parse_dates=True)
               if args.history_cache else load_history(args.end)).loc[:args.end]
    targets = strategies(history)
    for cutoff in ("2018-12-24", "2020-03-24", "2022-10-14", "2025-04-09"):
        pd.testing.assert_frame_equal(strategies(history.loc[:cutoff]), targets.loc[:cutoff])
    prices = history.loc[START:]
    results = {}
    for bps in (25, 50):
        baseline, _ = replay(prices, targets.sma175.loc[START:], bps)
        results[bps] = {}
        for name, target in targets.loc[START:].items():
            frame, fills = replay(prices, target, bps)
            weight = matched_quarterly_weight(prices, frame.weight.mean())
            quarter = prices.index[~prices.index.to_period("Q").duplicated()]
            matched, matched_fills = replay(prices, pd.Series(weight, index=prices.index), bps,
                                            rebalance_on=quarter)
            results[bps][name] = {
                **path_metrics(frame, fills),
                **attribution(frame, baseline, prices),
                "matched": path_metrics(matched, matched_fills),
            }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
