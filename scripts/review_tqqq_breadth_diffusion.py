"""Test persistence and cross-industry confirmation of Nasdaq-100 breadth warnings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from review_tqqq_breadth import PEAK_WINDOWS, fixed_window_events, replay_overlays, warning_signals
from review_tqqq_credit import adjusted_close
from review_tqqq_macro_defense import load_history

SECTORS = ("XLK", "XLY", "XLP", "XLI", "XLV", "IYZ")
NDX_SPLIT = ("QTEC", "QQXT")
WARNINGS = ("price", "breadth", "bounded", "persistent", "sector", "breadth_sector",
            "persistent_sector", "ndx_tech_only", "ndx_extech_only", "ndx_both")


def weekly_sector_breadth(history, weeks, prices):
    # Read the last completed close before the week starts, including short weeks.
    dates = history.index[history.index.get_indexer(weeks) - 1]
    prices = prices.reindex(history.index)
    valid = prices.rolling(50).count().eq(50)
    weak = prices.lt(prices.rolling(50).mean()).where(valid)
    return weak.mean(axis=1).where(valid.all(axis=1)).loc[dates].set_axis(weeks)


def diffusion_flags(history, panel, sectors, min_coverage):
    flags = warning_signals(history, panel, min_coverage)
    breadth = panel.breadth.where(panel.coverage.ge(min_coverage))
    days = history.index.get_indexer(panel.index) - 1
    close = history.qqqClose
    near_high = close.div(close.rolling(63).max()).ge(0.97).iloc[days].set_axis(panel.index)
    sector_breadth = weekly_sector_breadth(history, panel.index, sectors[list(SECTORS)])
    tech_weak, extech_weak = (
        weekly_sector_breadth(history, panel.index, sectors[[symbol]]).eq(1) for symbol in NDX_SPLIT
    )
    spreading = sector_breadth.gt(0.50) & sector_breadth.diff(4).gt(0)
    follow_up = flags.breadth.shift(fill_value=False) & near_high & breadth.diff().notna()
    persistent = follow_up & breadth.diff().le(0)
    return flags.assign(
        follow_up=follow_up,
        persistent=persistent,
        sector=near_high & spreading,
        breadth_sector=flags.breadth & spreading,
        persistent_sector=persistent & spreading,
        ndx_tech_only=near_high & tech_weak & ~extech_weak,
        ndx_extech_only=near_high & extech_weak & ~tech_weak,
        ndx_both=near_high & tech_weak & extech_weak,
        near_high=near_high,
        sector_weak_share=sector_breadth,
        sector_spreading=spreading,
        ndx_both_weak=tech_weak & extech_weak,
    )


def read_french_archive(path):
    with ZipFile(path) as archive:
        lines = archive.read(archive.namelist()[0]).decode("utf-8-sig").splitlines()
    title = next(i for i, row in enumerate(lines) if "Value Weighted Returns -- Daily" in row)
    end = next(i for i in range(title + 2, len(lines)) if not re.match(r"^\d{8},", lines[i]))
    returns = pd.read_csv(StringIO("\n".join(lines[title + 1:end])), index_col=0, na_values=[-99.99, -999])
    returns.index = pd.to_datetime(returns.index.astype(str), format="%Y%m%d")
    return (1 + returns.loc["2010":].div(100)).cumprod()


def evaluate(history, panel, sectors, french, coverage, bps):
    flags = diffusion_flags(history, panel, sectors, coverage)
    groups = {
        "common_near_high": flags["compare"],
        **{name: flags[name] for name in WARNINGS},
        "all_follow_ups": flags.follow_up,
        "follow_up_repair": flags.follow_up & ~flags.persistent,
        "sector_on_common_dates": flags.sector & flags["compare"],
        "all_near_high": flags.near_high,
    }
    events = fixed_window_events(history, panel, flags, bps, groups)
    waves = {}
    for label, (start, end) in PEAK_WINDOWS.items():
        peak = history.qqqClose.loc[start:end].idxmax()
        lead = flags.loc[peak - pd.offsets.Day(60):peak]
        after = flags.loc[peak + pd.offsets.Day(1):peak + pd.offsets.Day(60)]
        late = {}
        for name in ("sector_spreading", "ndx_both_weak"):
            hits = after.index[after[name]]
            if len(hits):
                day = hits[0]
                late[name] = {
                    "trade_date": str(day.date()),
                    "tqqq_open_vs_peak_close": float(history.at[day, "tqqqOpen"] / history.at[peak, "tqqqClose"] - 1),
                }
        waves[label] = {
            "peak": str(peak.date()),
            **{name: [str(date.date()) for date in lead.index[lead[name]]] for name in WARNINGS},
            "first_confirmation_after_peak_without_near_high_gate": late,
        }
    french_audit = {}
    if french is not None:
        # The research release is delayed and can be revised. Use it only
        # for independent retrospective cross-source agreement.
        share = weekly_sector_breadth(history, panel.index, french)
        valid = share.notna() & share.diff(4).notna()
        french_spread = share.gt(0.50) & share.diff(4).gt(0)
        proxy = flags.sector_weak_share.gt(0.50) & flags.sector_weak_share.diff(4).gt(0)
        french_audit = {
            "through": str(french.index[-1].date()),
            "weeks": int(valid.sum()),
            "correlation": float(share[valid].corr(flags.sector_weak_share[valid])),
            "etf_spread_weeks": int((proxy & valid).sum()),
            "both_spread_weeks": int((proxy & french_spread & valid).sum()),
        }
    return {
        "events": events,
        "systems": replay_overlays(history, flags, bps, WARNINGS),
        "waves": waves,
        "french_audit": french_audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--breadth", required=True, type=Path)
    parser.add_argument("--history-cache", type=Path)
    parser.add_argument("--sector-cache", type=Path)
    parser.add_argument("--french-archive", type=Path)
    args = parser.parse_args()
    history = (pd.read_csv(args.history_cache, index_col=0, parse_dates=True)
               if args.history_cache else load_history(args.end)).loc[:args.end]
    panel = pd.read_csv(args.breadth, index_col=0, parse_dates=True).loc[:args.end]
    symbols = (*SECTORS, *NDX_SPLIT)
    if args.sector_cache:
        sectors = pd.read_csv(args.sector_cache, index_col=0, parse_dates=True).loc[:args.end, list(symbols)]
    else:
        with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
            closes = pool.map(lambda symbol: adjusted_close(symbol, args.end).rename(symbol), symbols)
            sectors = pd.concat(list(closes), axis=1)
    if sectors.reindex(history.index).loc["2013":].isna().any().any():
        raise ValueError("Missing sector ETF closes in the replay period")
    french = read_french_archive(args.french_archive).loc[:args.end] if args.french_archive else None
    flags = diffusion_flags(history, panel, sectors, 0.90)
    for end in ("2018-12-24", "2020-03-24", "2022-10-14", "2025-04-09"):
        expected = diffusion_flags(history.loc[:end], panel.loc[:end], sectors.loc[:end], 0.90)
        pd.testing.assert_frame_equal(expected, flags.loc[:end])
    print(json.dumps({
        "sectors": list(SECTORS),
        "ndx_split": list(NDX_SPLIT),
        "input_sha256": {name: hashlib.sha256(frame.to_csv().encode()).hexdigest()
                         for name, frame in (("market", history), ("breadth", panel), ("sectors", sectors))},
        "results": {f"{bps}bps_{coverage:.0%}": evaluate(history, panel, sectors, french, coverage, bps)
                    for bps, coverage in ((25, 0.85), (25, 0.90), (25, 0.95), (50, 0.90))},
    }, indent=2))


if __name__ == "__main__":
    main()
