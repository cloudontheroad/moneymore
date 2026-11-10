"""Fixed literature-inspired Nasdaq exposure benchmarks, research only."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from review_tqqq_cut_edge import load_data
from review_tqqq_execution import ROOT, metrics
from review_tqqq_signal_edge import raw_signals, targets
from moneymore.leveraged_etf import LeveragedEtfPaper


def rules(data, dates):
    qqq = pd.Series(data["QQQ"].close.to_numpy(dtype=float), index=dates)
    above = qqq > qqq.rolling(200).mean()
    votes = pd.concat([qqq.pct_change(window).gt(0) for window in (21, 63, 252)], axis=1)
    realized = qqq.pct_change().rolling(21).std()
    high_vol = realized > realized.shift().rolling(756, min_periods=504).quantile(0.8)
    # Month-end rules use the last observed close, updating on the first
    # subsequent session close so no session calendar or future bar is needed.
    months = pd.Series(dates.to_period("M"), index=dates)
    new_month = months.ne(months.shift())
    trend_weight = votes.mean(axis=1).shift().where(new_month).ffill()
    vol_weight = (1 - 0.5 * high_vol.astype(float)).shift().where(new_month).ffill()
    return pd.DataFrame({
        "sma200_cash": above.astype(float),
        "sma200_qqq": above.astype(float),
        "multi_horizon": trend_weight,
        "tail_vol_only": vol_weight,
    }, index=dates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--source", choices=("yahoo", "cache"), default="yahoo")
    args = parser.parse_args()
    data, hashes = load_data(ROOT / "data/leveraged-etf", args.source, args.as_of)
    dates = pd.DatetimeIndex(data["QQQ"].trade_date)
    signal = rules(data, dates)
    for date in ("2016-06-27", "2020-03-16", "2022-01-24", "2025-04-04"):
        length = dates.searchsorted(date, side="right")
        if 0 < length < len(dates):
            truncated = {symbol: frame.iloc[:length] for symbol, frame in data.items()}
            pd.testing.assert_frame_equal(rules(truncated, dates[:length]), signal.iloc[:length])
    parent_signals = raw_signals(data, dates)
    start = int(dates.searchsorted("2013-01-01" if args.source == "yahoo" else "2022-01-01"))
    end = len(dates) - 1
    variants = {name: targets(parent_signals, start, end, name) for name in ("v2", "v3")}
    for name in signal:
        allocation = signal[name].to_numpy()
        variants[name] = pd.DataFrame({
            "tqqq": allocation, "sqqq": 1 - allocation if name == "sma200_qqq" else 0.0,
            "regime": name,
        })
    variants["hold"] = variants["v3"]
    variants["static_75"] = variants["v3"].assign(tqqq=0.75)

    result = {}
    for bps in (6, 25):
        daily = {}
        for name, target in variants.items():
            # Reuse the second asset slot of the research replay for QQQ.
            # The production SQQQ price feed and account are separate.
            market = {**data, "SQQQ": data["QQQ"]} if name == "sma200_qqq" else data
            rows = LeveragedEtfPaper.replay(
                market, target, list(range(start, end + 1)), "BASELINE",
                slippage=(bps - 1) / 10_000, buy_hold=name == "hold",
            )
            daily[name] = pd.DataFrame(rows).set_index("trade_date")
        periods = {
            "continuous": {name: metrics(frame) for name, frame in daily.items()}
        }
        for label, first, last in (
            ("2013_2021", "2013-01-01", "2021-12-31"),
            ("2022", "2022-01-01", "2022-12-31"),
            ("2023_2025", "2023-01-01", "2025-12-31"),
            ("2026_ytd", "2026-01-01", args.as_of),
        ):
            periods[label] = {}
            for name, frame in daily.items():
                cut = frame.iloc[1:].loc[first:last].copy()
                if len(cut) < 2:
                    continue
                prior = frame.index.get_loc(cut.index[0]) - 1
                cut = pd.concat([frame.iloc[[prior]], cut])
                cut["normalized_nav"] /= cut.normalized_nav.iloc[0]
                periods[label][name] = metrics(cut)
        hold = daily["hold"].normalized_nav.iloc[1:]
        rolling = {}
        for name, frame in daily.items():
            relative_nav = frame.normalized_nav.iloc[1:] / hold
            excess = (relative_nav / relative_nav.shift(756)).dropna()
            if not excess.empty:
                rolling[name] = {
                    "windows": len(excess), "win_rate": float((excess > 1).mean()),
                    "median_relative_annualized": float(np.median(excess ** (1 / 3) - 1)),
                }
        result[str(bps)] = {"periods": periods, "rolling_three_year_vs_hold": rolling}
    print(json.dumps({"as_of": str(dates[-1].date()), "input_sha256": hashes, "results": result},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
