"""Replay public TQQQ swing rules and test their recovery leg on fixed Kelly cuts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_external import daily_prices, summary
from review_tqqq_kelly_forecast import MODE, forecast_books
from review_tqqq_kelly_rebuild import matched_quarterly_weight, standalone_replay
from review_tqqq_kelly_sell_edge import aggregate, blocked_interval, opportunities, paired_returns
from review_tqqq_macro_defense import load_history
from review_tqqq_swing import START, replay
from moneymore.tqqq_kelly import scenario_book

WINDOWS = (("2013", "2019"), ("2020", "2022"), ("2023", None))


def open_signal(signal):
    return signal.shift(fill_value=False)


def state_targets(buy, sell):
    # The last mutually exclusive signal determines the position at today's open.
    if (buy & sell).any():
        raise ValueError("A session cannot both enter and exit")
    return pd.Series(np.select([buy, sell], [1.0, 0.0], default=np.nan), index=buy.index).ffill().fillna(0)


def signals(prices, bars):
    span = (bars.High - bars.Low).replace(0, np.nan)
    ibs = ((bars.Close - bars.Low) / span).reindex(prices.index)
    low, middle, high = (open_signal(mask) for mask in (ibs.lt(0.13), ibs.gt(0.50), ibs.gt(0.87)))
    qqq = prices.qqqClose
    qqq_returns = np.log(qqq).diff()
    stress = open_signal(qqq_returns.rolling(20).std().gt(qqq_returns.rolling(63).std()))
    targets = {
        "hold": pd.Series(1.0, index=prices.index),
        "ibs_upstream": state_targets(low, middle),
        "ibs_sell_first": state_targets(open_signal(ibs.lt(0.50)), high),
        "ibs_sell_first_stress": state_targets(open_signal(ibs.lt(0.50)), high & stress),
    }
    for length, buy_buffer, sell_buffer in ((200, 1, 1), (175, 1, 1), (175, 0, 1)):
        average = qqq.rolling(length).mean()
        buy = open_signal(qqq.gt(average * (1 + buy_buffer / 100)))
        sell = open_signal(qqq.lt(average * (1 - sell_buffer / 100)))
        targets[f"sma_{length}_{buy_buffer}_{sell_buffer}"] = state_targets(buy, sell)
    above = qqq.gt(qqq.rolling(175).mean())
    reclaim = open_signal(above & ~above.shift(fill_value=False))
    return pd.DataFrame(targets), pd.DataFrame({"ibs_low": low, "ibs_mid": middle, "sma_reclaim": reclaim})


def recovery_day(signal, i, j, rule):
    armed = False
    for k in range(i + 1, j):
        low, middle, reclaim = signal[k]
        recover = {"ibs_low": low, "ibs_rebound": armed and middle, "sma_reclaim": reclaim}
        if recover[rule]:
            return k
        armed |= low
    return j


def test_recovery(events, prices, recovery, bps):
    dates = prices.index
    opens = prices.tqqqOpen.to_numpy()
    signal = recovery.to_numpy()
    results = {}
    for model in ("old", "price"):
        cuts = [event for event in events if event["model"] == model]
        results[model] = {}
        for rule in ("month", "ibs_low", "ibs_rebound", "sma_reclaim"):
            pairs = []
            for event in cuts:
                i = dates.get_loc(event["date"])
                j = dates.get_loc(event["outcomes"]["month"]["end"])
                k = j if rule == "month" else recovery_day(signal, i, j, rule)
                gross, cost = paired_returns(event["starting_weight"], event["target"], opens[k] / opens[i], bps)
                pairs.append({
                    "date": event["date"], "buy_date": str(dates[k].date()), "sessions": k - i,
                    "early": k < j, "month_end": str(dates[j].date()),
                    "outcomes": {"paired": {"net_log": float(gross + cost),
                                            "gross_log": float(gross), "cost_log": float(cost)}},
                })
            baseline = np.array([event["outcomes"]["month"]["net_log"] for event in cuts])
            new = np.array([event["outcomes"]["paired"]["net_log"] for event in pairs])
            delta = new - baseline
            if rule == "month":
                np.testing.assert_allclose(new, baseline)
            results[model][rule] = {
                **aggregate(pairs, "paired"),
                "early": sum(pair["early"] for pair in pairs),
                "median_days": float(np.median([pair["sessions"] for pair in pairs])),
                "better_than_month": int((delta > 1e-10).sum()),
                "worse_than_month": int((delta < -1e-10).sum()),
                "improvement_interval": blocked_interval(delta),
                "pairs": pairs,
            }
    return results


def path_metrics(frame, fills):
    result = summary(np.r_[1, frame.nav], fills)
    result["exposure"] = float(frame.weight.mean())
    result["annual_turnover"] = float(frame.turnover.sum() * 252 / len(frame))
    result["periods"] = {}
    for start, end in WINDOWS:
        block = frame.loc[start:end]
        if len(block) < 2:
            continue
        i = frame.index.get_loc(block.index[0])
        anchor = frame.nav.iloc[i - 1] if i else 1
        result["periods"][start] = summary(np.r_[anchor, block.nav], block.turnover.gt(1e-9).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--history-cache", type=Path, help="Optional CSV history produced with --save-inputs")
    parser.add_argument("--tqqq-cache", type=Path, help="Optional CSV of adjusted TQQQ OHLC")
    parser.add_argument("--save-inputs", type=Path, help="Optional existing directory for the two input snapshots")
    parser.add_argument("--ibs-python", type=Path, help="Optional interpreter with the upstream IBS package installed")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as executor:
        history_job = executor.submit(load_history, args.end) if args.history_cache is None else None
        bars = (daily_prices("TQQQ", args.end) if args.tqqq_cache is None
                else pd.read_csv(args.tqqq_cache, index_col=0, parse_dates=True)).loc[:args.end]
        history = (history_job.result() if history_job else
                   pd.read_csv(args.history_cache, index_col=0, parse_dates=True)).loc[:args.end]
    if args.save_inputs:
        history.to_csv(args.save_inputs / "tqqq_swing_history.csv")
        bars.to_csv(args.save_inputs / "tqqq_swing_ohlc.csv")
    if bars.reindex(history.index).loc[START:, ["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("Missing adjusted TQQQ bars in the replay window")
    np.testing.assert_allclose(bars.loc[history.index, "Open"], history.tqqqOpen, rtol=1e-8)
    np.testing.assert_allclose(bars.loc[history.index, "Close"], history.tqqqClose, rtol=1e-8)
    targets, recovery = signals(history, bars)
    for cutoff in ("2018-12-24", "2020-03-24", "2022-10-14", "2025-04-09"):
        prefix_targets, prefix_recovery = signals(history.loc[:cutoff], bars.loc[:cutoff])
        pd.testing.assert_frame_equal(prefix_targets, targets.loc[:cutoff])
        pd.testing.assert_frame_equal(prefix_recovery, recovery.loc[:cutoff])
    if args.ibs_python:
        upstream = bars.assign(IBS=(bars.Close - bars.Low) / (bars.High - bars.Low)).loc[START:]
        source = (
            "import sys, pandas as pd; from ibs_strategy.backtest import run_backtest; "
            "bars = pd.read_csv(sys.stdin, index_col=0, parse_dates=True); "
            "print(run_backtest(bars).data['Position'].astype(int).to_json(orient='values'))"
        )
        actual = subprocess.run([str(args.ibs_python), "-c", source], input=upstream.to_csv(),
                                text=True, capture_output=True, check=True)
        # The upstream starts flat at START and cannot read the previous day's signal.
        expected, _ = signals(history.loc[START:], bars)
        np.testing.assert_array_equal(json.loads(actual.stdout), expected.ibs_upstream)
    market = history.rename(columns={
        "qqqClose": "qqq_close", "tqqqOpen": "tqqq_open", "tqqqClose": "tqqq_close",
        "VIX": "vix", "VIX3M": "vix3m", "HYG": "hyg", "IEI": "iei",
    })
    original = scenario_book(market)
    books, _ = forecast_books(market, original)
    prices = history.loc[START:]
    results = {}
    for bps in (0, 10, 25, 50):
        systems = {}
        hold_nav, _ = replay(prices, targets.hold.loc[START:], bps)
        for name, target in targets.loc[START:].items():
            frame, fills = replay(prices, target, bps)
            weight = matched_quarterly_weight(prices, frame.weight.mean())
            first_days = prices.index[~prices.index.to_period("Q").duplicated()]
            matched, matched_fills = replay(prices, pd.Series(weight, index=prices.index), bps,
                                            rebalance_on=first_days)
            relative_nav = np.log(frame.nav / hold_nav.nav)
            active = relative_nav.diff().fillna(relative_nav.iloc[0])
            systems[name] = {
                **path_metrics(frame, fills),
                "matched": path_metrics(matched, matched_fills),
                "log_excess_2022": float(active.loc["2022"].sum()),
                "log_excess_without_2022": float(active[active.index.year != 2022].sum()),
            }
        baseline, _, _ = standalone_replay(prices, original, bps, *MODE)
        events, _ = opportunities(market, original, books["price"], baseline, bps)
        results[bps] = {"systems": systems, "recovery": test_recovery(events, prices, recovery, bps)}
    fingerprints = {name: hashlib.sha256(frame.to_csv().encode()).hexdigest()
                    for name, frame in (("market", history), ("tqqq_ohlc", bars))}
    print(json.dumps({"start": START, "end": str(prices.index[-1].date()),
                      "input_sha256": fingerprints, "costs": results}, indent=2))


if __name__ == "__main__":
    main()
