"""Test partial rather than all-or-nothing healthy-regime v3 reductions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_cut_edge import load_data, reduction_context
from review_tqqq_execution import ROOT, bootstrap, metrics
from review_tqqq_position_edge import aggregate
from review_tqqq_signal_edge import raw_signals, targets

from moneymore.leveraged_etf import LeveragedEtfPaper

FRACTIONS = {"sell_half": 0.50, "sell_three_quarters": 0.75}
MAX_SIGNAL_DAYS = 5


def partial_targets(
    parent: pd.DataFrame, context: pd.DataFrame, start: int, end: int, fraction: float,
) -> pd.DataFrame:
    """Execute a first tranche, then finish if risk changes or the window expires."""
    original = parent.tqqq.to_numpy()
    values = original.copy()
    eligible = (context.healthy & ~context.positive_momentum & ~context.dip & ~context.dip_exit).to_numpy()
    retained = None
    expires = 0
    for i in range(start + 1, end + 1):
        prior, weight = original[i - 1], original[i]
        if not np.isfinite(prior) or not np.isfinite(weight):
            retained = None
            continue
        if retained is not None:
            if not eligible[i] or i >= expires or weight != prior:
                retained = None
            else:
                values[i] = retained
        elif eligible[i] and weight < prior:
            retained = round(weight + (1 - fraction) * (prior - weight), 6)
            expires = i + MAX_SIGNAL_DAYS
            values[i] = retained
    return parent.assign(tqqq=values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--source", choices=("yahoo", "cache"), default="yahoo")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/leveraged-etf")
    args = parser.parse_args()
    data, hashes = load_data(args.cache, args.source, args.as_of)
    dates = pd.DatetimeIndex(data["QQQ"].trade_date)
    if dates[-1] < pd.Timestamp("2025-12-31"):
        parser.error("This fixed evaluation requires completed data through 2025")
    signals = raw_signals(data, dates)
    context = reduction_context(data, signals)

    def replay(target, start, end, bps=6, hold=False):
        rows = LeveragedEtfPaper.replay(data, target, list(range(start, end + 1)), "BASELINE",
                                       slippage=(bps - 1) / 10_000, buy_hold=hold)
        return pd.DataFrame(rows).set_index("trade_date")

    def risk_match(target, start, end):
        vol = metrics(replay(target, start, end))["volatility"]
        low, high = 0.0, 1.0
        for _ in range(12):
            weight = (low + high) / 2
            realized = metrics(replay(target.assign(tqqq=weight), start, end))["volatility"]
            if realized < vol:
                low = weight
            else:
                high = weight
        return (low + high) / 2

    first_year = 2013 if args.source == "yahoo" else 2022
    daily, folds = {}, {}
    for year in range(first_year, dates[-1].year + 1):
        start = int(dates.searchsorted(f"{year}-01-01"))
        end = int(dates.searchsorted(f"{year + 1}-01-01")) - 1
        if end <= start:
            break
        train_start = int(dates.searchsorted(f"{year - 2}-01-01"))
        parent = targets(signals, start, end, "v3")
        parent_train = targets(signals, train_start, start - 1, "v3")
        variants = {"v3": parent, "hold": parent}
        matched = {}
        for name, fraction in FRACTIONS.items():
            candidate = partial_targets(parent, context, start, end, fraction)
            train = partial_targets(parent_train, context, train_start, start - 1, fraction)
            weight = risk_match(train, train_start, start - 1)
            variants[name] = candidate
            variants[f"{name}_static"] = parent.assign(tqqq=weight)
            matched[name] = weight
            # Overrides must be strictly positive and confined to the eligible state.
            diff = (candidate.tqqq - parent.tqqq).fillna(0)
            assert (diff >= 0).all()
            assert not diff[~context.healthy | context.positive_momentum | context.dip | context.dip_exit].any()
            mid = (start + end) // 2
            prefix = partial_targets(parent, context, start, mid, fraction)
            pd.testing.assert_series_equal(prefix.tqqq.iloc[:mid + 1], candidate.tqqq.iloc[:mid + 1])
        daily[year] = {bps: {name: replay(target, start, end, bps, name == "hold")
                             for name, target in variants.items()} for bps in (6, 25)}
        folds[year] = {
            "matched_static_weight": matched,
            "retained_position_days": {name: int(variants[name].tqqq.gt(parent.tqqq).sum()) for name in FRACTIONS},
            "metrics": {str(bps): {name: metrics(frame) for name, frame in frames.items()}
                        for bps, frames in daily[year].items()},
        }

    summaries = {}
    for label, lower, upper in (("earlier", 2013, 2021), ("recent", 2022, 2026),
                                 ("bull_2023_2025", 2023, 2025), ("all", first_year, 2026)):
        years = [year for year in daily if lower <= year <= upper]
        if not years:
            continue
        stats = {str(bps): {name: aggregate([daily[y][bps][name] for y in years])
                            for name in daily[years[0]][bps]} for bps in (6, 25)}
        stats["intervals_25bps"] = {
            name: {
                ref: bootstrap([
                    np.log(daily[y][25][name].normalized_nav / daily[y][25][ref].normalized_nav)
                    .diff().dropna().to_numpy() for y in years
                ]) for ref in ("v3", f"{name}_static")
            } for name in FRACTIONS
        }
        summaries[label] = stats

    gates = {}
    bull = summaries["bull_2023_2025"]["25"]
    for name in FRACTIONS:
        checks = {
            "preserves_2022_drawdown_at_25bps": (
                folds[2022]["metrics"]["25"][name]["max_drawdown"]
                >= folds[2022]["metrics"]["25"]["v3"]["max_drawdown"] - 1e-8
            ),
            "closes_one_third_of_bull_return_gap_at_25bps": (
                bull[name]["annualized"] - bull["v3"]["annualized"]
                >= max(0.0, bull["hold"]["annualized"] - bull["v3"]["annualized"]) / 3
            ),
            "higher_25bps_sharpe_vs_static_in_each_sample": all(
                summaries[sample]["25"][name]["sharpe"] > summaries[sample]["25"][f"{name}_static"]["sharpe"]
                for sample in ("earlier", "recent") if sample in summaries
            ),
            "positive_25bps_excess_lower_bound_vs_v3_in_both_samples": (
                "earlier" in summaries and all(
                    summaries[sample]["intervals_25bps"][name]["v3"][0] > 0
                    for sample in ("earlier", "recent")
                )
            ),
        }
        gates[name] = {"checks": checks, "passed": all(checks.values())}
    print(json.dumps({
        "source": args.source, "first_date": str(dates[0].date()), "as_of": str(dates[-1].date()),
        "input_sha256": hashes, "folds": folds, "summaries": summaries, "gates": gates,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
