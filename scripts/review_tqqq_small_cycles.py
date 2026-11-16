"""Separate short bull-market pullbacks from larger reversals and price both trading legs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_crisis_mechanisms import SECTORS, mechanisms
from review_tqqq_peak_exit import episodes, frozen_reentry, overlay, signal_table
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics

START = "2019-01-02"
STOP_AFTER = 30
GROUPS = ("all", "internals", "credit", "both")
REPAIRS = ("five_days", "price_repair", "breadth_repair")


def features(history, context, sectors):
    q = np.log(history.qqqClose)
    ret = q.diff()
    vol = ret.rolling(63).std().shift() * np.sqrt(5)
    high = q.ge(q.rolling(20).max())
    positions = pd.Series(np.arange(len(q)), index=q.index)
    peak = q.where(high).ffill()
    peak_age = positions - positions.where(high).ffill()
    peak_vol = vol.where(high).ffill()
    slow = q.rolling(175).mean()
    market_up = q.gt(slow) & slow.gt(slow.shift(20))
    pressure = np.log(history.HYG / history.IEI).diff(5)
    pressure_cut = pressure.shift().rolling(504, min_periods=252).quantile(.25)
    credit = pressure.lt(pressure_cut) & np.log(history.VIX / history.VIX3M).diff(5).gt(0)
    sectors = np.log(sectors[list(SECTORS)].reindex(q.index))
    sector5 = sectors.diff(5)
    sector2 = sectors.diff(2)
    weak = sector5.lt(0).where(sector5.notna()).mean(axis=1).where(sector5.count(axis=1).eq(len(SECTORS)))
    repair = sector2.gt(0).where(sector2.notna()).mean(axis=1).where(sector2.count(axis=1).eq(len(SECTORS)))
    internals = weak.gt(.5) & (np.log(context.SOXX.reindex(q.index)) - q).diff(5).lt(0)
    alert = market_up & peak_age.between(1, 10) & (peak - q).ge(peak_vol) & q.diff(3).lt(0)
    reclaim = q.gt(q.shift().rolling(3).max())
    return pd.DataFrame({
        "q": q, "slow": slow, "peak": peak, "peak_age": peak_age, "peak_vol": peak_vol,
        "drawdown_sigma": (peak - q) / peak_vol, "alert": alert,
        "internals": internals, "credit": credit, "both": internals & credit,
        "price_repair": reclaim, "breadth_repair": reclaim | (ret.gt(0) & repair.ge(2 / 3)),
        "trend_up": market_up, "credit_healing": pressure.gt(0),
        "breaks_repair": q.lt(q.shift().rolling(3).min()),
    })


def first_alerts(data):
    # Do not let repeated signals from one high restart the holding clock.
    anchor = pd.Series(np.arange(len(data)), index=data.index) - data.peak_age
    return data.alert & ~anchor.where(data.alert).ffill().shift().eq(anchor)


def classify(data, start, end):
    peak = data.peak.iloc[start]
    window = data.iloc[start + 1:end + 1]
    regained = window.q.ge(peak).to_numpy().nonzero()[0]
    broken = window.q.lt(window.slow).to_numpy().nonzero()[0]
    if len(broken) and (not len(regained) or broken[0] < regained[0]):
        return "trend_break", start + 1 + broken[0]
    if len(regained):
        duration = regained[0] + 1
        return ("quick_repair" if duration <= 10 else "slow_repair"), start + duration
    return "unresolved_30d", end


def reentry_at(data, first_open, repair):
    if repair == "five_days":
        return min(first_open + 5, len(data))
    limit = min(first_open + 10, len(data))
    checks = np.flatnonzero(data[repair].iloc[first_open:limit - 1].to_numpy())
    return first_open + 1 + checks[0] if len(checks) else limit


def study(prices, data, alerts):
    opens = prices.tqqqOpen.to_numpy()
    close = prices.tqqqClose.to_numpy()
    rows = []
    for i in np.flatnonzero(alerts):
        if prices.index[i] < pd.Timestamp(START) or i + STOP_AFTER >= len(data):
            continue
        label, end = classify(data, i, i + STOP_AFTER)
        horizon = slice(i + 1, min(i + 11, len(opens)))
        row = {
            "date": str(prices.index[i].date()),
            "anchor": str(prices.index[i - int(data.peak_age.iloc[i])].date()),
            "class": label, "resolved_on": str(prices.index[end].date()), "days": int(end - i),
            "qqq_to_alert": float(np.expm1(data.q.iloc[i] - data.peak.iloc[i])),
            "qqq_to_low": float(np.expm1(data.q.iloc[i:end + 1].min() - data.peak.iloc[i])),
            "tqqq_next_10d_low": float(close[horizon].min() / opens[i + 1] - 1),
            **{name: bool(data[name].iloc[i]) for name in GROUPS if name != "all"},
        }
        for rule in REPAIRS:
            buy = reentry_at(data, i + 1, rule)
            gross = np.log(opens[i + 1] / opens[buy])
            row[rule] = {
                "buy": str(prices.index[buy].date()),
                "gross": float(gross),
                "net_25": float(gross + np.log((1 - .0025) / (1 + .0025))),
            }
        rows.append(row)
    return rows


def allocation(data, baseline, alerts, group, repair):
    weights = baseline.copy()
    events = alerts if group == "all" else alerts & data[group]
    prior_cash = np.flatnonzero(baseline.eq(0))
    next_available = 0
    for signal in np.flatnonzero(events):
        sell = signal + 1
        if sell >= len(data) or sell < next_available or baseline.iloc[sell] == 0:
            continue
        buy = reentry_at(data, sell, repair)
        cash = prior_cash.searchsorted(sell)
        if cash < len(prior_cash):
            buy = min(buy, prior_cash[cash])
        weights.iloc[sell:buy] = .75
        next_available = buy
    return weights


def buy_back(data, baseline, core, repair):
    """Try a single earlier full reentry per prior cash interval."""
    valid = data.trend_up & data[repair]
    if repair == "breadth_repair":
        valid &= data.credit_healing
    ready = valid.shift(fill_value=False)
    failed = data.breaks_repair.shift(fill_value=False)
    cash = baseline.eq(0)
    chunks = cash.ne(cash.shift()).cumsum()
    target = baseline.copy()
    for _, block in cash.loc[cash].groupby(chunks):
        # A new defense starts on an opening fill. Its own close is the first eligible repair observation.
        first_ten = block.index[1:11]
        candidates = first_ten[ready.loc[first_ten] & core.loc[first_ten].eq(1)]
        if not len(candidates):
            continue
        active = block.loc[candidates[0]:].index
        stop = np.flatnonzero((failed.loc[active] | core.loc[active].eq(0)).to_numpy())
        target.loc[active[:stop[0]] if len(stop) else active] = 1.
    return target


def evaluate(history, targets):
    result = {}
    for cost in (25, 50):
        by_era = {}
        for era, (start, end) in {"full": (START, "2026-11-06"), "2019_2022": (START, "2022-12-31"),
                                  "2023_2026": ("2023-01-01", "2026-11-06")}.items():
            prices = history.loc[start:end]
            weights = targets.loc[prices.index]
            base, _ = replay(prices, weights.prior, cost)
            models = {}
            for name, target in weights.items():
                frame, fills = replay(prices, target, cost)
                models[name] = path_metrics(frame, fills)
                if name not in ("hold", "prior"):
                    models[name]["extra"] = episodes(prices, weights.prior, target, base, frame,
                                                     added_exposure=name.startswith("buy_back_"))
            by_era[era] = models
        result[str(cost)] = by_era
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "context", "alternate", "sectors", "fred", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    history, context, alternate, sectors = map(read, (args.history, args.context, args.alternate, args.sectors))
    sectors = sectors.reindex(history.index)
    rates = pd.concat([read(args.fred / f"{name}.csv").iloc[:, 0].rename(name)
                       for name in ("BAA10Y", "DGS10", "DFII10")], axis=1).sort_index()
    data = features(history, context, sectors)
    alerts = first_alerts(data)
    core, safe, clock = frozen_reentry(history, context, alternate)
    context_risk = mechanisms(history.qqqClose, context.SPY, sectors[list(SECTORS)], sectors.XLF,
                              sectors.IWF / sectors.IWD, rates, real_yields=True)
    prior = signal_table(history, context, alternate)
    prior["baseline"] = prior.shock_credit_vol | context_risk.narrowing_exit
    baseline, _ = overlay(core, prior, safe, clock, "baseline")
    targets = pd.DataFrame({"hold": 1., "prior": baseline})
    for group in GROUPS:
        for repair in REPAIRS:
            targets[f"{group}_{repair}"] = allocation(data, baseline, alerts, group, repair)
    for repair in ("price_repair", "breadth_repair"):
        targets[f"buy_back_{repair}"] = buy_back(data, baseline, core, repair)
    for date in ("2020-03-20", "2023-10-27", "2025-04-04"):
        prefix = features(history.loc[:date], context, sectors)
        pd.testing.assert_frame_equal(data.loc[:date], prefix)
        pd.testing.assert_series_equal(targets.loc[:date, "all_breadth_repair"], allocation(
            prefix, baseline.loc[:date], first_alerts(prefix), "all", "breadth_repair"), check_names=False)
        pd.testing.assert_series_equal(targets.loc[:date, "buy_back_breadth_repair"], buy_back(
            prefix, baseline.loc[:date], core.loc[:date], "breadth_repair"), check_names=False)
    report = {"events": study(history, data, alerts), "portfolios": evaluate(history, targets)}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
