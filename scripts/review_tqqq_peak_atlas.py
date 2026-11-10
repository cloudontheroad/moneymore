"""Align major TQQQ selloffs with preceding Nasdaq highs and audit early warnings."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from review_tqqq_breadth import fixed_window_events, replay_overlays
from review_tqqq_breadth_diffusion import SECTORS, weekly_sector_breadth
from review_tqqq_credit import adjusted_close
from review_tqqq_kelly_forecast import standardize
from review_tqqq_macro_defense import load_history
from review_tqqq_swing import START

SYMBOLS = (*SECTORS, "QQXT", "SPY", "RSP", "SOXX", "^VXN", "^TNX")
PRICE = ("price_gap", "price_20", "price_63", "price_vol")
CROSS = {
    "ndx_extech": "QQXT vs QQQ",
    "chips": "SOXX vs QQQ",
    "us_breadth": "RSP vs SPY",
    "defensives": "XLY vs XLP",
    "credit": "HYG vs IEI",
    "nasdaq_vol": "VXN jump",
    "vol_curve": "VIX vs VIX3M",
    "rates": "10-year yield rise",
    "industry": "sector weakness",
}


def features(history, context, panel):
    qqq = np.log(history.qqqClose)
    prices = np.log(context.reindex(history.index))
    data = pd.DataFrame({
        "price_gap": qqq - qqq.rolling(63).max(),
        "price_20": qqq.diff(20),
        "price_63": qqq.diff(63),
        "price_vol": qqq.diff().rolling(21).std(),
        "ndx_extech": -(prices.QQXT - qqq).diff(20),
        "chips": -(prices.SOXX - qqq).diff(20),
        "us_breadth": -(prices.RSP - prices.SPY).diff(20),
        "defensives": -(prices.XLY - prices.XLP).diff(20),
        "credit": -np.log(history.HYG / history.IEI).diff(20),
        "nasdaq_vol": prices["^VXN"].diff(5),
        "vol_curve": np.log(history.VIX / history.VIX3M),
        "rates": context["^TNX"].reindex(history.index).diff(20),
    }).shift().reindex(panel.index)
    data["industry"] = weekly_sector_breadth(history, panel.index, context[list(SECTORS)])
    data["ndx_participation"] = -panel.breadth.where(panel.coverage.ge(0.90)).diff(4)
    return data


def alarms(data):
    monitored = data[[*CROSS, "ndx_participation"]]
    threshold = monitored.shift().rolling(104, min_periods=26).quantile(0.80)
    return monitored.gt(threshold).where(monitored.notna() & threshold.notna())


def peak_atlas(history, warning, drawdown):
    """Retrospective QQQ highs with a complete 63-session TQQQ outcome."""
    qqq = history.qqqClose
    tqqq = history.tqqqClose
    local_high = qqq.eq(qqq.rolling(41, center=True).max())
    index = history.index
    candidates = []
    for i in np.flatnonzero(local_high.to_numpy()):
        if index[i] < pd.Timestamp(START) or i + 63 >= len(index):
            continue
        trough = tqqq.iloc[i + 1:i + 64].idxmin()
        loss = tqqq.at[trough] / tqqq.iloc[i] - 1
        if loss <= -drawdown:
            candidates.append((index[i], trough, loss))
    waves = []
    for peak, trough, loss in candidates:
        # Keep distinct waves. A second local high during the first decline
        # belongs to that decline unless it exceeds the original QQQ high.
        if waves and peak <= pd.Timestamp(waves[-1]["trough"]):
            if qqq.at[peak] <= qqq.at[pd.Timestamp(waves[-1]["peak"])]:
                continue
            waves.pop()
        prior = warning.loc[:peak].tail(4)
        following = warning.loc[peak + pd.offsets.Day(1):].head(4)
        last_week = prior.iloc[-1]
        first_warning = {}
        for name in prior:
            hits = prior.index[prior[name].eq(True)]
            if len(hits):
                first_warning[name] = (peak - hits[0]).days
        waves.append({
            "peak": str(peak.date()), "trough": str(trough.date()), "loss": float(loss),
            "last_decision": str(prior.index[-1].date()),
            "available": [name for name, value in last_week.items() if pd.notna(value)],
            "last_week_flags": [name for name, value in last_week.items() if pd.notna(value) and value],
            "first_warning_days": first_warning,
            "first_after_peak_days": {
                name: (following.index[following[name].eq(True)][0] - peak).days
                for name in following if following[name].eq(True).any()
            },
        })
    return waves


def outcomes(history, weeks, horizon=63):
    loc = history.index.get_indexer(weeks)
    opens = history.tqqqOpen
    result = pd.DataFrame(index=weeks, columns=["low", "end_return"], dtype=float)
    for date, i in zip(weeks, loc, strict=True):
        j = i + horizon
        if j < len(opens):
            gross = opens.iloc[i:j + 1] / opens.iloc[i]
            result.loc[date] = gross.min() - 1, gross.iloc[-1] - 1
    result["settled_at"] = history.index.take(np.minimum(loc + horizon, len(history) - 1))
    result.loc[result.low.isna(), "settled_at"] = pd.NaT
    return result


def walk_forward(data, labels, eligible, threshold):
    good = eligible & data[[*PRICE, *CROSS]].notna().all(axis=1)
    output = pd.DataFrame(index=data.index, columns=["price_model", "cross_model"], dtype=float)
    for date in data.index[good]:
        train = good & labels.settled_at.lt(date)
        target = labels.loc[train, "low"].le(-threshold)
        if len(target) < 52 or min(target.sum(), (~target).sum()) < 5:
            continue
        for name, inputs in (("price_model", list(PRICE)), ("cross_model", [*PRICE, *CROSS])):
            x, query = standardize(data.loc[train, inputs].to_numpy(), data.loc[date, inputs].to_numpy())
            model = LogisticRegression(C=0.30, max_iter=500).fit(x, target)
            output.at[date, name] = model.predict_proba(query[None])[0, 1]
    return output


def alarm_summary(warning, mask):
    return {
        name: {"weeks": int(values.notna().sum()), "alerts": int(values.eq(True).sum())}
        for name, values in warning.loc[mask].items()
    }


def matched_highs(data, warning, benign, waves):
    """Retrospective price-state controls, selected without using other features."""
    trailing_alerts = warning.astype(float).rolling(4, min_periods=2).max()
    benign_prices = data.loc[benign, list(PRICE)].dropna()
    scale = (benign_prices.quantile(0.75) - benign_prices.quantile(0.25)).clip(lower=1e-8)
    records = []
    for wave in waves:
        date = pd.Timestamp(wave["last_decision"])
        point = data.loc[date, list(PRICE)]
        distance = ((benign_prices - point) / scale).pow(2).sum(axis=1, min_count=len(PRICE))
        separation = pd.offsets.Day(28)
        pool = distance.loc[(distance.index < date - separation) | (distance.index > date + separation)]
        matches = []
        for match in pool.dropna().sort_values(kind="stable").index:
            if all(abs((match - prior).days) >= 28 for prior in matches):
                matches.append(match)
            if len(matches) == 3:
                break
        if not matches:
            continue
        records.append({
            "peak": wave["peak"], "controls": [str(day.date()) for day in matches],
            "case_alerts": trailing_alerts.loc[date].dropna().to_dict(),
            "control_alert_rates": trailing_alerts.loc[matches].mean().dropna().to_dict(),
        })
    result = {}
    for name in warning:
        pairs = [(row["case_alerts"][name], row["control_alert_rates"][name])
                 for row in records if name in row["case_alerts"] and name in row["control_alert_rates"]]
        if pairs:
            case, controls = np.mean(pairs, axis=0)
            result[name] = {"paired_waves": len(pairs), "case_rate": float(case), "control_rate": float(controls)}
    return {"matches": records, "alert_rates": result}


def evaluate(history, data, warning, panel, threshold):
    eligible = data.price_gap.ge(np.log(0.97)) & (data.index >= pd.Timestamp(START))
    labels = outcomes(history, data.index)
    severe = eligible & labels.low.le(-threshold)
    benign = eligible & labels.low.gt(-threshold) & labels.end_return.gt(0)
    probability = walk_forward(data, labels, eligible, threshold)
    common = probability.notna().all(axis=1) & labels.low.notna()
    true = labels.low.le(-threshold)
    flags = probability.gt(probability.shift().rolling(104, min_periods=16).quantile(0.80))
    flags["all_near_high"] = eligible
    metrics = {}
    for name in probability:
        pred, observed = probability.loc[common, name], true[common]
        selected = common & flags[name]
        metrics[name] = {
            "weeks": int(common.sum()), "severe_weeks": int(observed.sum()),
            "brier": float(np.square(pred - observed).mean()),
            "auc": float(roc_auc_score(observed, pred)),
            "selected": int(selected.sum()), "selected_severe": int((selected & true).sum()),
            "dates": [str(day.date()) for day in flags.index[selected]],
        }
    waves = peak_atlas(history, warning, threshold)
    windows = [(wave["peak"], wave["trough"]) for wave in waves]
    return {
        "waves": waves,
        "matched_highs": matched_highs(data, warning, benign, waves),
        "cohorts": {
            "severe_high_weeks": int(severe.sum()), "benign_high_weeks": int(benign.sum()),
            "severe": alarm_summary(warning, severe), "benign": alarm_summary(warning, benign),
        },
        "forecast": metrics,
        "events": fixed_window_events(history, panel, flags, 25, {name: flags[name] for name in flags}),
        "systems": {
            bps: replay_overlays(history, flags, bps, list(probability.columns), wave_windows=windows)
            for bps in (25, 50)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--breadth", required=True, type=Path)
    parser.add_argument("--history-cache", type=Path)
    parser.add_argument("--context-cache", type=Path)
    args = parser.parse_args()
    history = (pd.read_csv(args.history_cache, index_col=0, parse_dates=True)
               if args.history_cache else load_history(args.end)).loc[:args.end]
    panel = pd.read_csv(args.breadth, index_col=0, parse_dates=True).loc[:args.end]
    if args.context_cache:
        context = pd.read_csv(args.context_cache, index_col=0, parse_dates=True).loc[:args.end, list(SYMBOLS)]
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            context = pd.concat(list(pool.map(lambda name: adjusted_close(name, args.end).rename(name), SYMBOLS)), axis=1)
    data = features(history, context, panel)
    warning = alarms(data)
    eligible = data.price_gap.ge(np.log(0.97)) & (data.index >= pd.Timestamp(START))
    full = walk_forward(data, outcomes(history, panel.index), eligible, 0.30)
    for cutoff in ("2018-12-24", "2020-03-24", "2022-10-14", "2025-04-09"):
        prefix = features(history.loc[:cutoff], context.loc[:cutoff], panel.loc[:cutoff])
        pd.testing.assert_frame_equal(prefix, data.loc[:cutoff])
        pd.testing.assert_frame_equal(alarms(prefix), warning.loc[:cutoff])
        earlier = walk_forward(prefix, outcomes(history.loc[:cutoff], prefix.index),
                               eligible.loc[:cutoff], 0.30)
        pd.testing.assert_frame_equal(earlier, full.loc[:cutoff])
    print(json.dumps({
        "input_sha256": {name: hashlib.sha256(frame.to_csv().encode()).hexdigest()
                         for name, frame in (("market", history), ("breadth", panel), ("context", context))},
        "results": {f"{threshold:.0%}": evaluate(history, data, warning, panel, threshold)
                    for threshold in (0.25, 0.30, 0.35)},
    }, indent=2))


if __name__ == "__main__":
    main()
