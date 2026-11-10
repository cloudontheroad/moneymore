"""Isolate semiconductor leadership breaks and defensive rotation at Nasdaq highs."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_breadth import early_overlay, fixed_window_events, replay_overlays
from review_tqqq_credit import adjusted_close
from review_tqqq_macro_defense import load_history
from review_tqqq_peak_atlas import alarms, features, peak_atlas
from review_tqqq_swing import START, replay
from review_tqqq_turning_points import price_signals

RULES = ("chips", "defensives", "either", "same_week", "paired_recent", "former_leader", "staples_positive",
         "chips_equal", "defensives_equal", "chips_confirmed", "defensives_confirmed")
EXPLORATORY = ("soft_market", "chips_soft_market", "chips_equal_soft_market", "rates_up",
               "defensives_rates_up", "defensives_equal_rates_up")
ALT_SYMBOLS = ("XSD", "RSPD", "RSPS")
FRED_CHECK = "defensives_equal_fred_rates_up"


def signals(history, context, panel, alternate):
    data = features(history, context, panel)
    warning = alarms(data)
    trend = price_signals(history).trend.reindex(panel.index).eq(1)
    eligible = data.price_gap.ge(np.log(0.97)) & trend
    chips = warning.chips.eq(True)
    defensives = warning.defensives.eq(True)
    recent_chips = chips.shift(fill_value=False).rolling(3, min_periods=1).max().eq(1)
    recent_defensives = defensives.shift(fill_value=False).rolling(3, min_periods=1).max().eq(1)
    relative_chips = np.log(context.SOXX.reindex(history.index) / history.qqqClose)
    previous_lead = relative_chips.shift(20).diff(43).shift().reindex(panel.index).gt(0)
    staples_positive = context.XLP.reindex(history.index).pct_change(20).shift().reindex(panel.index).gt(0)
    alt = alternate.reindex(history.index)
    equal_weight = pd.DataFrame({
        "chips_equal": -np.log(alt.XSD / history.qqqClose).diff(20),
        "defensives_equal": -np.log(alt.RSPD / alt.RSPS).diff(20),
    }).shift().reindex(panel.index)
    equal_alarms = equal_weight.gt(equal_weight.shift().rolling(104, min_periods=26).quantile(0.80))
    flags = pd.DataFrame({
        "near_high": eligible,
        "chips": eligible & chips,
        "defensives": eligible & defensives,
        "either": eligible & (chips | defensives),
        "same_week": eligible & chips & defensives,
        "paired_recent": eligible & ((chips & recent_defensives) | (defensives & recent_chips) | (chips & defensives)),
        "former_leader": eligible & chips & previous_lead,
        "staples_positive": eligible & defensives & staples_positive,
        "chips_equal": eligible & equal_alarms.chips_equal,
        "defensives_equal": eligible & equal_alarms.defensives_equal,
        "chips_confirmed": eligible & chips & equal_alarms.chips_equal,
        "defensives_confirmed": eligible & defensives & equal_alarms.defensives_equal,
        "chips_only": eligible & chips & ~defensives,
        "defensives_only": eligible & defensives & ~chips,
        "neither": eligible & ~chips & ~defensives,
        "repair": pd.Series(False, index=panel.index),
    })
    prior_strength = data.price_63.shift().rolling(104, min_periods=26).median()
    context_flags = pd.DataFrame({
        "strong_63d": data.price_63.gt(prior_strength).where(prior_strength.notna()),
        "rising_yields": data.rates.gt(0).where(data.rates.notna()),
    })
    # These checks were added after seeing the main subgroup table. The same
    # state is applied to the alternate ETF so it tests proxy transfer.
    soft_market = context_flags.strong_63d.eq(False)
    rates_up = context_flags.rising_yields.eq(True)
    flags = flags.assign(
        soft_market=eligible & soft_market,
        chips_soft_market=flags.chips & soft_market,
        chips_equal_soft_market=flags.chips_equal & soft_market,
        rates_up=eligible & rates_up,
        defensives_rates_up=flags.defensives & rates_up,
        defensives_equal_rates_up=flags.defensives_equal & rates_up,
    )
    if "DGS10" in context:
        # FRED H.15 releases can follow the closing bell. Delay one session
        # before evaluating the same 20-session change as the market yield.
        published = context.DGS10.reindex(history.index).ffill(limit=3).shift()
        fred_up = published.diff(20).shift().reindex(panel.index).gt(0)
        flags[FRED_CHECK] = flags.defensives_equal & fred_up
    return flags, context_flags, warning


def audit_episodes(history, flags, rule, bps, trend, base):
    prices = history.loc[START:]
    overlay = early_overlay(trend, flags, rule).loc[START:]
    adjusted, _ = replay(prices, overlay, bps)
    active = np.log(adjusted.nav / base.nav)
    cash = (trend.loc[START:] > overlay).to_numpy()
    starts = np.flatnonzero(cash & ~np.r_[False, cash[:-1]])
    ends = np.flatnonzero(~cash & np.r_[False, cash[:-1]])
    episodes = []
    for i in starts:
        following = ends[ends > i]
        j = following[0] if len(following) else len(prices) - 1
        previous = active.iloc[i - 1] if i else 0.0
        edge = float(active.iloc[j] - previous)
        opens = prices.tqqqOpen.iloc[i:j + 1] / prices.tqqqOpen.iloc[i] - 1
        episodes.append({
            "date": str(prices.index[i].date()), "end": str(prices.index[j].date()),
            "closed": bool(len(following)), "net_log_edge": edge,
            "lowest_open": float(opens.min()), "tqqq_move": float(opens.iloc[-1]),
            "opening_action": "sell" if i and trend.loc[prices.index[i - 1]] else "skip_reentry",
        })
    np.testing.assert_allclose(sum(event["net_log_edge"] for event in episodes), active.iloc[-1], atol=1e-9)
    edge = np.array([event["net_log_edge"] for event in episodes])
    best = max(episodes, key=lambda row: row["net_log_edge"], default=None)
    return {
        "episodes": episodes, "count": len(episodes), "wins": int((edge > 0).sum()),
        "avoided_log": float(edge.clip(min=0).sum()), "missed_log": float(edge.clip(max=0).sum()),
        "best": best, "cumulative_excess_without_best": float(np.expm1(edge.sum() - max(0, edge.max())))
        if len(edge) else 0.0,
    }


def evaluate(history, panel, flags, context_flags, waves, bps):
    rules = [name for name in (*RULES, *EXPLORATORY, FRED_CHECK) if name in flags]
    groups = {name: flags[name] for name in (*rules, "near_high", "chips_only", "defensives_only", "neither")}
    for context in context_flags:
        for name in ("chips", "defensives", "chips_equal", "defensives_equal"):
            known = context_flags[context].notna()
            condition = context_flags[context].eq(True)
            groups[f"{name}_{context}"] = flags[name] & condition
            groups[f"{name}_without_{context}"] = flags[name] & known & ~condition
    events = {weeks: fixed_window_events(history, panel, flags, bps, groups, weeks=weeks)
              for weeks in (2, 4, 8)}
    windows = [(wave["peak"], wave["trough"]) for wave in waves]
    systems = replay_overlays(history, flags, bps, rules, wave_windows=windows)
    trend = price_signals(history).trend
    base, _ = replay(history.loc[START:], trend.loc[START:], bps)
    return {
        "windows": events, "systems": systems,
        "episodes": {name: audit_episodes(history, flags, name, bps, trend, base) for name in rules},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--breadth", required=True, type=Path)
    parser.add_argument("--history-cache", type=Path)
    parser.add_argument("--context-cache", required=True, type=Path)
    parser.add_argument("--alternate-cache", type=Path)
    parser.add_argument("--fred-yield-cache", type=Path)
    args = parser.parse_args()
    history = (pd.read_csv(args.history_cache, index_col=0, parse_dates=True)
               if args.history_cache else load_history(args.end)).loc[:args.end]
    panel = pd.read_csv(args.breadth, index_col=0, parse_dates=True).loc[:args.end]
    context = pd.read_csv(args.context_cache, index_col=0, parse_dates=True).loc[:args.end]
    if args.fred_yield_cache:
        fred = pd.read_csv(args.fred_yield_cache, index_col=0, parse_dates=True, na_values=["."])
        context["DGS10"] = fred["DGS10"].reindex(context.index)
    if args.alternate_cache:
        alternate = pd.read_csv(args.alternate_cache, index_col=0, parse_dates=True).loc[:args.end]
    else:
        with ThreadPoolExecutor(max_workers=len(ALT_SYMBOLS)) as pool:
            alternate = pd.concat(list(pool.map(lambda s: adjusted_close(s, args.end).rename(s), ALT_SYMBOLS)), axis=1)
    if alternate.reindex(history.index).loc[START:, list(ALT_SYMBOLS)].isna().any().any():
        raise ValueError("Missing equal-weight ETF closes")
    flags, contexts, raw = signals(history, context, panel, alternate)
    for date in ("2018-12-24", "2020-03-24", "2022-10-14", "2025-04-09"):
        earlier = signals(history.loc[:date], context.loc[:date], panel.loc[:date], alternate.loc[:date])
        for actual, full in zip(earlier, (flags, contexts, raw), strict=True):
            pd.testing.assert_frame_equal(actual, full.loc[:date])
    waves = peak_atlas(history, raw, 0.30)
    for wave in waves:
        peak = pd.Timestamp(wave["peak"])
        past = flags.loc[:peak].tail(4)
        wave["eligible_warnings"] = {
            name: [str(date.date()) for date in past.index[past[name]]]
            for name in (*RULES, *EXPLORATORY, FRED_CHECK) if name in past
        }
    inputs = {name: hashlib.sha256(frame.to_csv().encode()).hexdigest() for name, frame in
              (("market", history), ("context", context), ("breadth", panel), ("alternate", alternate))}
    print(json.dumps({"input_sha256": inputs, "waves": waves, "results": {
        bps: evaluate(history, panel, flags, contexts, waves, bps) for bps in (25, 50)
    }}, indent=2))


if __name__ == "__main__":
    main()
