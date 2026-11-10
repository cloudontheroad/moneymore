"""Pair v3 reductions with a bounded delay and test selective deferral."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_execution import ROOT, bootstrap, metrics
from review_tqqq_position_edge import aggregate
from review_tqqq_signal_edge import raw_signals, targets

from moneymore.leveraged_etf import DATA_SYMBOLS, LeveragedEtfPaper, YahooDailyClient

RULES = ("defer_vol_trim", "defer_vol_and_momentum")
DEFER_SESSIONS = 5


def reduction_context(data, signals):
    qqq = pd.Series(data["QQQ"].close.to_numpy(), index=signals.index)
    vix = pd.Series(data["^VIX"].close.to_numpy(), index=signals.index)
    vix3m = pd.Series(data["^VIX3M"].close.to_numpy(), index=signals.index)
    long_trend = qqq.ewm(span=100, min_periods=100).mean()
    healthy = (qqq > long_trend) & (long_trend > long_trend.shift(20)) & (vix < vix3m)
    return pd.DataFrame({
        "healthy": healthy,
        "positive_momentum": signals.momentum.gt(0),
        "dip": signals.dip.astype(bool),
        "dip_exit": signals.dip.shift(fill_value=False) & ~signals.dip,
    }).reset_index(drop=True)


def category(row):
    if row.dip or row.dip_exit:
        return "dip_transition"
    if not row.healthy:
        return "weak_or_stressed"
    return "healthy_vol_trim" if row.positive_momentum else "healthy_momentum_cut"


def selective_targets(parent, context, start, end, rule):
    """Delay one eligible cut. Further cuts, stress and expiry restore parent sizing."""
    values = parent.tqqq.to_numpy(copy=True)
    original = parent.tqqq.to_numpy()
    retained = None
    expiry = 0
    for i in range(start + 1, end + 1):
        weight, prior = original[i], original[i - 1]
        if not np.isfinite(weight) or not np.isfinite(prior):
            retained = None
            continue
        row = context.iloc[i]
        group = category(row)
        eligible = group == "healthy_vol_trim" or (
            rule == "defer_vol_and_momentum" and group == "healthy_momentum_cut"
        )
        if retained is not None:
            if not eligible or i >= expiry or weight != prior:
                retained = None
            else:
                values[i] = max(weight, retained)
        elif eligible and weight < prior:
            retained = prior
            expiry = i + DEFER_SESSIONS
            values[i] = prior
    return parent.assign(tqqq=values)


def pair_events(parent, context, base, replay, data, start, end):
    """Change one close target at a time, keeping pre-cut ledger history identical."""
    rows = []
    original = parent.tqqq.to_numpy()
    market_up = data["TQQQ"].close.pct_change().iloc[start:end + 1].to_numpy() > 0
    for i in range(start + 1, end):
        prior, target = original[i - 1], original[i]
        if not (np.isfinite(prior) and np.isfinite(target) and target < prior):
            continue
        # Restore the original policy at the next changed parent target, or
        # after five signal days. Include the next open's catch-up trade.
        stop = i + 1
        while stop < min(i + DEFER_SESSIONS, end) and original[stop] == target:
            stop += 1
        until = min(stop + 1, end)
        variant = parent.copy()
        variant.loc[i:stop - 1, "tqqq"] = prior
        delayed = replay(variant, start, until)
        original_path = base.iloc[:until - start + 2]
        excess = np.log(delayed.normalized_nav / original_path.normalized_nav).diff().iloc[1:].to_numpy()
        window = slice(i - start + 1, until - start + 1)
        net = excess[window]
        up = market_up[window]
        rows.append({
            "signal_date": str(data["QQQ"].trade_date.iat[i]),
            "through_date": str(data["QQQ"].trade_date.iat[until]),
            "category": category(context.iloc[i]),
            "cut_weight": float(prior - target),
            "delay_log_excess": float(net.sum()),
            "delay_log_excess_up_days": float(net[up].sum()),
            "delay_log_excess_other_days": float(net[~up].sum()),
            "delay_cost_delta_usd": float(delayed.cost.sum() - original_path.cost.sum()),
        })
    return rows


def load_data(cache: Path, source: str, as_of: str):
    def load(symbol):
        if source == "yahoo":
            frame = YahooDailyClient(timeout=35).history(symbol, "2010-02-11", as_of)
            payload = frame.to_csv(index=False).encode()
        else:
            payload = (cache / f"{symbol.lstrip('^')}.csv").read_bytes()
            frame = pd.read_csv(BytesIO(payload))
        return symbol, frame.loc[frame.trade_date <= as_of], hashlib.sha256(payload).hexdigest()

    with ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(load, DATA_SYMBOLS))
    data = LeveragedEtfPaper._align({symbol: frame for symbol, frame, _ in downloaded})
    return data, {symbol: digest for symbol, _, digest in downloaded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--source", choices=("cache", "yahoo"), default="cache")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/leveraged-etf")
    args = parser.parse_args()
    data, hashes = load_data(args.cache, args.source, args.as_of)
    dates = pd.DatetimeIndex(data["QQQ"].trade_date)
    signals = raw_signals(data, dates)
    context = reduction_context(data, signals)
    first_year = 2013 if args.source == "yahoo" else 2022

    def replay(target, start, end, bps=6, hold=False):
        rows = LeveragedEtfPaper.replay(
            data, target, list(range(start, end + 1)), "BASELINE",
            slippage=(bps - 1) / 10_000, buy_hold=hold,
        )
        return pd.DataFrame(rows).set_index("trade_date")

    events, folds, daily_by_year = [], {}, {}
    for year in range(first_year, dates[-1].year + 1):
        start = int(dates.searchsorted(f"{year}-01-01"))
        end = int(dates.searchsorted(f"{year + 1}-01-01")) - 1
        if end <= start:
            break
        parent = targets(signals, start, end, "v3")
        candidates = {name: selective_targets(parent, context, start, end, name) for name in RULES}
        variants = {"v3": parent, "hold": parent, **candidates}
        daily = {
            bps: {name: replay(target, start, end, bps, name == "hold") for name, target in variants.items()}
            for bps in (6, 25)
        }
        events.extend(pair_events(parent, context, daily[6]["v3"], replay, data, start, end))
        daily_by_year[year] = daily
        folds[year] = {
            "metrics": {name: metrics(frame) for name, frame in daily[6].items()},
            "stress_25bps": {name: metrics(frame) for name, frame in daily[25].items()},
        }
    summaries = {}
    for label, lower, upper in (
        ("earlier_transfer", 2013, 2021),
        ("recent", 2022, 2026),
        ("bull_2023_2025", 2023, 2025),
        ("all", first_year, 2026),
    ):
        years = [year for year in daily_by_year if lower <= year <= upper]
        if not years:
            continue
        stats = {}
        for bps in (6, 25):
            stats[str(bps)] = {
                name: aggregate([daily_by_year[year][bps][name] for year in years])
                for name in ("v3", "hold", *RULES)
            }
        stats["intervals_vs_v3_25bps"] = {
            name: bootstrap([
                np.log(daily_by_year[y][25][name].normalized_nav / daily_by_year[y][25]["v3"].normalized_nav)
                .diff().dropna().to_numpy() for y in years
            ]) for name in RULES
        }
        summaries[label] = stats
    event_table = pd.DataFrame(events)
    event_table["sample"] = np.where(event_table.signal_date.lt("2022-01-01"), "earlier_transfer", "recent")
    event_summary = event_table.groupby(["sample", "category"]).agg(
        count=("delay_log_excess", "size"), mean_delay_log_excess=("delay_log_excess", "mean"),
        win_rate=("delay_log_excess", lambda values: (values > 0).mean()),
        up_day_contribution=("delay_log_excess_up_days", "mean"),
        other_day_contribution=("delay_log_excess_other_days", "mean"),
        cost_delta_usd=("delay_cost_delta_usd", "mean"),
    ).reset_index().to_dict("records")
    gates = {}
    if "earlier_transfer" in summaries:
        for name in RULES:
            bull = summaries["bull_2023_2025"]["25"]
            checks = {
                "no_deeper_drawdown_2022": folds[2022]["stress_25bps"][name]["max_drawdown"] >=
                folds[2022]["stress_25bps"]["v3"]["max_drawdown"] - 1e-8,
                "closes_at_least_one_third_bull_gap": bull[name]["annualized"] - bull["v3"]["annualized"] >=
                max(0, bull["hold"]["annualized"] - bull["v3"]["annualized"]) / 3,
                "positive_25bps_excess_in_both_samples": all(
                    summaries[sample]["intervals_vs_v3_25bps"][name][0] > 0
                    for sample in ("earlier_transfer", "recent")
                ),
            }
            gates[name] = {"checks": checks, "passed": all(checks.values())}
    print(json.dumps({
        "source": args.source, "first_date": str(dates[0].date()), "as_of": str(dates[-1].date()),
        "input_sha256": hashes, "folds": folds, "summaries": summaries, "event_summary": event_summary,
        "research_gates": gates, "events": events,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
