"""Attribute 3x/1x switching costs and compare fixed execution alternatives."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from moneymore.leveraged_etf import LeveragedEtfPaper
from review_tqqq_cut_edge import load_data
from review_tqqq_execution import ROOT, bootstrap, metrics
from review_tqqq_paper_states import allocations


def leverage_rules(qqq: pd.Series) -> pd.DataFrame:
    distance = qqq / qqq.rolling(200).mean() - 1
    reference = pd.Series(np.where(distance > 0, 3.0, 1.0), index=qqq.index)
    buffered = pd.Series(np.where(distance > .01, 3.0, np.where(distance < -.01, 1.0, np.nan)),
                         index=qqq.index).ffill().fillna(reference)
    staged = pd.Series(np.where(distance > .01, 3.0, np.where(distance < -.01, 1.0, 2.0)),
                       index=qqq.index)
    months = pd.Series(qqq.index.to_period("M"), index=qqq.index)
    monthly = reference.shift().where(months.ne(months.shift())).ffill()
    return pd.DataFrame({"daily": reference, "buffer_1pct": buffered, "middle_2x": staged,
                         "monthly": monthly})


def switch_episodes(states, low_cost, high_cost, hold):
    """Group daily NAV differences from each first executable open until the next."""
    states = states.iloc[:-1]
    signals = states.index[states.ne(states.shift())]
    # The first day's target is placed from cash. Later segments start
    # immediately after the close signal, on the next session's open.
    dates = list(low_cost.index[1:])
    next_dates = {day: dates[i + 1] for i, day in enumerate(dates[:-1])}
    starts = [dates[0], *(next_dates[day] for day in signals if day in next_dates)]
    drag = np.log(low_cost.normalized_nav / high_cost.normalized_nav).diff()
    active = np.log(low_cost.normalized_nav / hold.normalized_nav).diff()
    out = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else None
        days = [day for day in dates if start <= day and (stop is None or day < stop)]
        if not days:
            continue
        position = dates.index(start)
        state = float(states.loc[dates[max(0, position - 1)]])
        out.append({
            "first_mark": start, "last_mark": days[-1], "sessions": len(days), "target_leverage": state,
            "incremental_cost_log_drag": float(drag.loc[days].sum()),
            "low_cost_active_log_return": float(active.loc[days].sum()),
            "low_cost_turnover": float(low_cost.loc[days, "turnover"].sum()),
        })
    np.testing.assert_allclose(sum(x["incremental_cost_log_drag"] for x in out), drag.sum())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--source", choices=("yahoo", "cache"), default="yahoo")
    args = parser.parse_args()
    data, hashes = load_data(ROOT / "data/leveraged-etf", args.source, args.as_of)
    dates = pd.DatetimeIndex(data["QQQ"].trade_date)
    qqq = pd.Series(data["QQQ"].close.to_numpy(), index=dates)
    leverage = leverage_rules(qqq)
    for day in ("2016-06-27", "2020-03-16", "2022-01-24", "2025-04-04"):
        pd.testing.assert_frame_equal(leverage_rules(qqq.loc[:day]), leverage.loc[:day])
    market = {**data, "SQQQ": data["QQQ"]}
    start = int(dates.searchsorted("2013-01-01" if args.source == "yahoo" else "2022-01-01"))
    indices = list(range(start, len(dates)))

    def run(name, bps):
        column = "daily" if name == "hold" else name
        rows = LeveragedEtfPaper.replay(
            market, allocations(leverage[column]), indices, "BASELINE",
            slippage=(bps - 1) / 10_000, buy_hold=name == "hold",
        )
        return pd.DataFrame(rows).set_index("trade_date")

    results, daily = {}, {}
    for bps in (6, 15, 25):
        daily[bps] = {name: run(name, bps) for name in (*leverage.columns, "hold")}
        results[bps] = {}
        for name, frame in daily[bps].items():
            nav = frame.normalized_nav.iloc[1:]
            relative = nav / daily[bps]["hold"].normalized_nav.iloc[1:]
            rolling = (relative / relative.shift(756)).dropna()
            periods = {}
            for label, a, b in (("earlier", "2013", "2021"), ("2022", "2022", "2022"),
                                ("bull", "2023", "2025"), ("ytd", "2026", "2026")):
                sample = nav.loc[f"{a}-01-01":f"{b}-12-31"]
                if not len(sample):
                    continue
                prior = frame.index.get_loc(sample.index[0]) - 1
                marks = pd.concat([frame.iloc[[prior]], frame.loc[sample.index]]).copy()
                marks["normalized_nav"] /= marks.normalized_nav.iloc[0]
                periods[label] = metrics(marks)
            results[bps][name] = {
                "overall": metrics(frame), "periods": periods,
                "switch_count": int(leverage[name].iloc[start:-1].diff().abs().gt(0).sum()) if name != "hold" else 0,
                "three_year_win_rate": float((rolling > 1).mean()) if len(rolling) else None,
                "annual_log_excess_interval_vs_hold": bootstrap([
                    sample.to_numpy() for _, sample in np.log(relative).diff().dropna().groupby(
                        lambda day: day[:4]
                    )
                ]) if len(relative) > 1 and name != "hold" else None,
            }

    state = leverage["daily"].iloc[start:].copy()
    state.index = state.index.strftime("%Y-%m-%d")
    episodes = switch_episodes(state, daily[6]["daily"], daily[25]["daily"], daily[6]["hold"])
    summary = {}
    for name, selected in (
        ("up_to_5_sessions", [e for e in episodes if e["sessions"] <= 5]),
        ("up_to_20_sessions", [e for e in episodes if e["sessions"] <= 20]),
        ("all", episodes),
    ):
        summary[name] = {
            "count": len(selected),
            "cost_drag": sum(e["incremental_cost_log_drag"] for e in selected),
            "active_log_return": sum(e["low_cost_active_log_return"] for e in selected),
        }
    print(json.dumps({
        "as_of": str(dates[-1].date()), "input_sha256": hashes, "results": results,
        "episode_summary": summary, "episodes": episodes,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
