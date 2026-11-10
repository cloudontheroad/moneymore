"""Keep the defensive-rotation exits fixed and compare causal recovery rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_breadth import early_overlay
from review_tqqq_kelly_rebuild import matched_quarterly_weight
from review_tqqq_kelly_sell_edge import blocked_interval
from review_tqqq_peak_styles import FRED_CHECK, audit_episodes, signals
from review_tqqq_sharpe import daily_returns, paired_interval, sharpe
from review_tqqq_swing import START, replay
from review_tqqq_swing_round_trips import path_metrics
from review_tqqq_turning_points import price_signals

SELL_RULES = {"market": "defensives_equal_rates_up", "fred": FRED_CHECK}
RECOVERY = ("fixed", "week_one", "week_two", "qqq", "rates", "style", "either", "both")
ROBUSTNESS = ("both_delayed", "both_other_rate")


def observable(history, context, alternate, weeks, source):
    rate = context["^TNX" if source == "market" else "DGS10"].reindex(history.index)
    if source == "fred":
        rate = rate.ffill(limit=3).shift()
    style = np.log(alternate.RSPD / alternate.RSPS).reindex(history.index)
    return pd.DataFrame({
        "rates": rate,
        "rate_change": rate.diff(5),
        "style": style,
        "style_change": style.diff(5),
        "qqq_change": history.qqqClose.pct_change(5),
    }).shift().reindex(weeks)


def return_dates(trend, episodes, observed, rule, *, delay_sessions=0):
    target = trend.copy()
    choices = []
    for event in episodes:
        start = pd.Timestamp(event["date"])
        end = pd.Timestamp(event["end"])
        # On the original cash path, the endpoint is the scheduled four-week
        # cap or the day the 175-day trend turns off. An active final window
        # remains eligible through the last observed session.
        possible = observed.loc[observed.index > start]
        possible = possible.loc[possible.index < end if event["closed"] else possible.index <= end]
        start_values = observed.loc[start]
        rate = possible.rates.lt(start_values.rates) & possible.rate_change.lt(0)
        style = possible["style"].gt(start_values["style"]) & possible.style_change.gt(0)
        choices_by_rule = {
            "fixed": possible.index[:0],
            "week_one": possible.index[:1],
            "week_two": possible.index[1:2],
            "qqq": possible.index[possible.qqq_change.gt(0)],
            "rates": possible.index[rate],
            "style": possible.index[style],
            "either": possible.index[rate | style],
            "both": possible.index[rate & style],
        }
        hits = choices_by_rule[rule]
        stop = end
        advanced = False
        if len(hits):
            delayed = target.index.get_loc(hits[0]) + delay_sessions
            if delayed < target.index.get_loc(end) + int(not event["closed"]):
                stop = target.index[delayed]
                advanced = True
        i, j = target.index.get_indexer([start, stop])
        target.iloc[i:j + int(not advanced and not event["closed"])] = 0
        choices.append({
            "sell": event["date"], "fixed_end": event["end"], "decision_end": str(stop.date()),
            "advanced": advanced, "settled": event["closed"], "days_earlier": (end - stop).days,
            "action_at_fixed_end": "buy" if event["closed"] and trend.at[end] else (
                "trend_off" if event["closed"] else "open"),
        })
    return target, choices


def episode_comparison(frame, fixed, base, prices, choices):
    against_fixed = np.log(frame.nav / fixed.nav)
    against_base = np.log(frame.nav / base.nav)
    for row in choices:
        start = prices.index.get_loc(row["sell"])
        end = prices.index.get_loc(row["fixed_end"])
        delta = against_fixed.iloc[end] - (against_fixed.iloc[start - 1] if start else 0)
        edge = against_base.iloc[end] - (against_base.iloc[start - 1] if start else 0)
        buy = prices.index.get_loc(row["decision_end"])
        tqqq = prices.tqqqOpen.iloc[buy:end + 1] / prices.tqqqOpen.iloc[buy]
        row.update({
            "delta_vs_fixed": float(delta), "edge_vs_175": float(edge),
            "tqqq_low_after_recovery": float(tqqq.min() - 1) if row["advanced"] else None,
            "tqqq_return_until_fixed_end": float(tqqq.iloc[-1] - 1) if row["advanced"] else None,
        })
    np.testing.assert_allclose(sum(row["delta_vs_fixed"] for row in choices), against_fixed.iloc[-1], atol=1e-9)
    np.testing.assert_allclose(sum(row["edge_vs_175"] for row in choices), against_base.iloc[-1], atol=1e-9)
    settled = [row for row in choices if row["settled"]]
    delta = np.array([row["delta_vs_fixed"] for row in settled])
    return {
        "episodes": choices,
        "advanced": sum(row["advanced"] for row in settled),
        "better": int((delta > 1e-9).sum()), "worse": int((delta < -1e-9).sum()),
        "mean_delta": float(delta.mean()), "mean_delta_interval": blocked_interval(delta),
        "cumulative_delta_without_best": float(np.expm1(delta.sum() - max(0, delta.max()))),
        "mean_delta_excluding_2022": float(np.mean([
            row["delta_vs_fixed"] for row in settled if not row["sell"].startswith("2022")
        ])),
    }


def build_targets(history, panel, context, alternate, flags, source):
    trend = price_signals(history).trend
    reference, _ = replay(history.loc[START:], trend.loc[START:], 25)
    sell_rule = SELL_RULES[source]
    events = audit_episodes(history, flags, sell_rule, 25, trend, reference)["episodes"]
    features = observable(history, context, alternate, panel.index, source)
    targets = {rule: return_dates(trend.loc[START:], events, features, rule) for rule in RECOVERY}
    other = "fred" if source == "market" else "market"
    other_features = observable(history, context, alternate, panel.index, other)
    targets["both_delayed"] = return_dates(trend.loc[START:], events, features, "both", delay_sessions=1)
    targets["both_other_rate"] = return_dates(trend.loc[START:], events, other_features, "both")
    pd.testing.assert_series_equal(targets["fixed"][0], early_overlay(trend, flags, sell_rule).loc[START:],
                                   check_names=False)
    for target, choices in targets.values():
        assert all(target.at[row["sell"]] == 0 for row in choices)
    return trend.loc[START:], targets


def evaluate(prices, trend, targets, bps):
    base, base_fills = replay(prices, trend, bps)
    fixed, _ = replay(prices, targets["fixed"][0], bps)
    output = {"sma175": path_metrics(base, base_fills)}
    fixed_returns = daily_returns(fixed.nav)
    quarter = prices.index[~prices.index.to_period("Q").duplicated()]
    for name, (target, template) in targets.items():
        frame, fills = replay(prices, target, bps)
        returns = daily_returns(frame.nav)
        weight = matched_quarterly_weight(prices, frame.weight.mean())
        matched, matched_fills = replay(prices, pd.Series(weight, index=prices.index), bps,
                                        rebalance_on=quarter)
        output[name] = {
            **path_metrics(frame, fills),
            "sharpe_without_2022": float(sharpe(returns[prices.index.year != 2022])),
            "sharpe_delta_interval": paired_interval(returns, fixed_returns),
            "matched": path_metrics(matched, matched_fills),
            "pairing": episode_comparison(frame, fixed, base, prices, [dict(row) for row in template]),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    for name in ("history-cache", "breadth", "context-cache", "alternate-cache", "fred-yield-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: pd.read_csv(path, index_col=0, parse_dates=True, na_values=["."]).loc[:args.end]
    history, panel, context, alternate = map(read, (args.history_cache, args.breadth,
                                                    args.context_cache, args.alternate_cache))
    context["DGS10"] = read(args.fred_yield_cache).DGS10.reindex(context.index)
    flags, _, _ = signals(history, context, panel, alternate)
    output = {}
    for source in SELL_RULES:
        trend, targets = build_targets(history, panel, context, alternate, flags, source)
        for cutoff in ("2015-12-22", "2022-01-10", "2024-05-20", "2026-02-10"):
            h, p, c, a = (frame.loc[:cutoff] for frame in (history, panel, context, alternate))
            prefix_flags, _, _ = signals(h, c, p, a)
            _, truncated = build_targets(h, p, c, a, prefix_flags, source)
            for rule in (*RECOVERY, *ROBUSTNESS):
                pd.testing.assert_series_equal(truncated[rule][0], targets[rule][0].loc[:cutoff])
        output[source] = {bps: evaluate(history.loc[START:], trend, targets, bps) for bps in (25, 50)}
    inputs = {name: hashlib.sha256(frame.to_csv().encode()).hexdigest() for name, frame in
              (("market", history), ("context", context), ("breadth", panel), ("alternate", alternate))}
    print(json.dumps({"input_sha256": inputs, "results": output}, indent=2))


if __name__ == "__main__":
    main()
