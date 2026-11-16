"""Validate frozen top rules on official NDXE and on comparable TQQQ decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd

from review_tqqq_crisis_mechanisms import CASES, SECTORS, case_profiles
from review_tqqq_peak_exit import START, episodes
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics
from review_tqqq_top_followthrough import target_table, weekly_features
from review_tqqq_weekly_cycle import execution_state

ERAS = {"pre_tqqq": ("2005", "2009"), "early": ("2010", "2013"), "2014_2019": ("2014", "2019"),
        "2020_2026": ("2020", "2026")}
FEE = .0025
ROUND_TRIP = (1 - FEE) / (1 + FEE)


def official_index(directory):
    pairs = []
    for path in sorted(directory.glob("ndxe-*.xlsx")):
        with ZipFile(path) as archive:
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            headers = ["".join(item.itertext()) for item in strings]
            if headers[:2] != ["Trade Date", "Index Value"]:
                raise ValueError(f"Unexpected Nasdaq export columns in {path.name}")
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            for row in sheet.findall(".//{*}row")[1:]:
                cells = [cell.findtext("{*}v") for cell in row[:2]]
                if cells and all(value is not None for value in cells):
                    pairs.append((int(cells[0]), float(cells[1])))
    if not pairs:
        raise ValueError("No Nasdaq NDXE data")
    dates, values = zip(*pairs, strict=True)
    index = pd.to_datetime(list(dates), unit="D", origin="1899-12-30")
    result = pd.Series(values, index=index, name="NDXE").sort_index()
    if not result.index.is_unique or result.lt(0).any():
        raise ValueError("NDXE export contains duplicate dates or negative values")
    return result


def exit_windows(week, index_close, traded_close, traded_open, *, tail):
    close_week = index_close.reindex(week.index)
    near_high = close_week.ge(.97 * close_week.rolling(13).max())
    eligible = near_high & week[["leadership", "leadership_cut", "equal_trend", "peak_age"]].notna().all(axis=1)
    eligible &= week.equal_trend.shift().notna()
    equal_weak = week.equal_trend.lt(0) & week.equal_trend.shift().lt(0)
    dates = index_close.index
    locs = dates.get_indexer(week.index)
    # The historical index leg uses next close. The fund leg uses next open.
    mark = traded_close.reindex(dates)
    execution = traded_open.reindex(dates)
    rows = []
    for i, signal_date in enumerate(week.index):
        start = locs[i] + 1
        end = start + 20
        if not eligible.iloc[i] or end >= len(dates):
            continue
        entry, one_week, four_weeks = execution.iloc[[start, start + 5, end]]
        held_path = mark.iloc[start:end + 1]
        if not (np.isfinite([entry, one_week, four_weeks]).all() and held_path.notna().all()):
            continue
        anchor = locs[i] - int(week.peak_age.iloc[i])
        rows.append({
            "date": signal_date, "phase": i % 5, "anchor": dates[anchor],
            "equal_weak": bool(equal_weak.iloc[i]), "trigger": bool(week.equal_weight.iloc[i]),
            "tail": bool(held_path.min() / entry - 1 <= -tail),
            "skip_net_5": float(np.log(entry / one_week * ROUND_TRIP)),
            "skip_net": float(np.log(entry / four_weeks * ROUND_TRIP)),
        })
    return pd.DataFrame(rows).set_index("date")


def describe_windows(windows):
    def summary(sample):
        return {"n": len(sample), "tails": int(sample["tail"].sum()), "wins": int(sample.skip_net.gt(0).sum()),
                "mean": float(sample.skip_net.mean()) if len(sample) else None,
                "mean_5": float(sample.skip_net_5.mean()) if len(sample) else None}

    report = {}
    for name, (start, end) in ERAS.items():
        era = windows.loc[start:end]
        alerts = era.loc[era.trigger].drop_duplicates("anchor")
        peers = era.loc[era.equal_weak & ~era.trigger & ~era.anchor.isin(alerts.anchor)]
        report[name] = {
            "all_near": summary(era), "matched_state": summary(peers), "alerts_one_per_high": summary(alerts),
            "alerts": json.loads(alerts.reset_index().to_json(orient="records", date_format="iso")),
            "phase_controls": {str(phase): summary(peers.loc[peers.phase.eq(phase)]) for phase in range(5)},
        }
    return report


def reentry_windows(history, prior, week):
    risk = execution_state(history, week.slow_confirmed).eq(True)
    opens = history.tqqqOpen
    rows = []
    starts = np.flatnonzero(prior.eq(1) & prior.shift(fill_value=0).eq(0))
    stops = np.flatnonzero(prior.eq(0) & prior.shift(fill_value=0).eq(1))
    for start in starts:
        if opens.index[start] < pd.Timestamp(START) or start + 20 >= len(opens):
            continue
        next_stop = stops[stops > start]
        stop = next_stop[0] if len(next_stop) else len(opens) - 1
        row = {"date": str(opens.index[start].date()), "blocked": bool(risk.iloc[start])}
        for horizon in (5, 20):
            end = min(start + horizon, stop)
            row[f"hold_{horizon}"] = float(opens.iloc[end] / opens.iloc[start] * ROUND_TRIP - 1)
        rows.append(row)
    return rows


def portfolio_check(history, targets):
    result = {}
    prices = history.loc[START:]
    weights = targets.loc[START:]
    for cost in (25, 50):
        baseline, fills = replay(prices, weights.prior, cost)
        models = {"prior": path_metrics(baseline, fills)}
        for name in ("prior_equal_weight", "prior_other_etf"):
            target = weights[name]
            frame, fills = replay(prices, target, cost)
            models[name] = {**path_metrics(frame, fills), "incremental": episodes(
                prices, weights.prior, target, baseline, frame)}
        result[str(cost)] = models
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "context", "alternate", "sectors", "equal", "ndxe", "fred", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    history, context, alternate = map(read, (args.history, args.context, args.alternate))
    sectors = read(args.sectors)
    equal = read(args.equal).iloc[:, 0]
    ndxe = official_index(args.ndxe)
    ndx = read(args.fred / "NASDAQ100.csv").iloc[:, 0].dropna().loc[ndxe.index[0]:history.index[-1]]
    placeholders = ndxe.index[ndxe.eq(0)]
    if len(placeholders.intersection(ndx.index)):
        raise ValueError("NDXE export contains a zero on a trading day")
    ndxe = ndxe.drop(placeholders)
    # Use the traded market calendar where it is available. FRED can carry a holiday value.
    extra = ndx.loc[history.index[0]:].index.difference(history.index)
    ndx = ndx.drop(extra)
    if not ndxe.index.difference(ndx.index).empty:
        raise ValueError("NDXE export contains unexpected trading dates")
    missing = ndx.index.difference(ndxe.index)
    rates = pd.concat([read(args.fred / f"{name}.csv").iloc[:, 0].rename(name)
                       for name in ("BAA10Y", "DGS10", "DFII10")], axis=1).sort_index()
    credit = -np.log(history.HYG / history.IEI)
    official = weekly_features(ndx, ndxe, pd.DataFrame(index=ndx.index), rates.BAA10Y.reindex(ndx.index).shift())
    modern = weekly_features(history.qqqClose, equal, sectors[list(SECTORS)], credit)
    cut = "2014-12-26"
    prior_ndx = ndx.loc[:cut]
    pd.testing.assert_frame_equal(official.loc[:cut], weekly_features(
        prior_ndx, ndxe.loc[:cut], pd.DataFrame(index=prior_ndx.index), rates.BAA10Y.reindex(prior_ndx.index).shift()))
    cross = official[["equal_weight"]].reindex(history.index)
    positions = pd.Series(np.arange(len(history)), index=history.index)
    anchor_dates = pd.Series(ndx.index, index=np.arange(len(ndx))).reindex(
        ndx.index.get_indexer(official.index) - official.peak_age.to_numpy()
    ).set_axis(official.index)
    cross["peak_age"] = positions - anchor_dates.reindex(history.index).map(positions)
    targets = target_table(history, context, alternate, sectors, rates, modern, cross)
    cases = {name: CASES[name] for name in ("2007", "2011", "2015", "2018", "2020", "2021", "2024", "2025")}
    official_daily = official[["equal_weight"]].reindex(ndx.index).eq(True)
    shared = modern.equal_weight.to_frame("etf").join(official.equal_weight.rename("ndxe")).loc[START:].dropna()
    report = {
        "coverage": {"start": str(ndxe.index[0].date()), "end": str(ndxe.index[-1].date()), "sessions": len(ndxe),
                     "closed_day_placeholders": len(placeholders), "filtered_fred_dates": len(extra),
                     "missing_ndxe_sessions": [str(date.date()) for date in missing]},
        "ndx": describe_windows(exit_windows(official, ndx, ndx, ndx, tail=.10)),
        "tqqq": describe_windows(exit_windows(modern, history.qqqClose, history.tqqqClose, history.tqqqOpen, tail=.20)),
        "historical_cases": case_profiles(ndx, official_daily, cases, rules={"equal_weight": "equal_weight"}),
        "signal_overlap": {"etf": int(shared.etf.sum()), "ndxe": int(shared.ndxe.sum()),
                           "both": int((shared.etf & shared.ndxe).sum()),
                           "same_targets": targets.prior_equal_weight.loc[START:].equals(
                               targets.prior_other_etf.loc[START:])},
        "portfolio_check": portfolio_check(history, targets),
        "reentries": reentry_windows(history, targets.prior, modern),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
