"""Build Nasdaq-100 breadth from dated official rosters and adjusted stock closes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from review_tqqq_credit import adjusted_close
from review_tqqq_kelly_rebuild import matched_quarterly_weight
from review_tqqq_macro_defense import load_history
from review_tqqq_sharpe import daily_returns, paired_interval, sharpe
from review_tqqq_swing import START, replay
from review_tqqq_swing_round_trips import open_signal, path_metrics, state_targets

NASDAQ = "https://indexes.nasdaqomx.com/Index/ExportWeightings/NDX"
# Only explicit same-security aliases; leave ambiguous or delisted codes unresolved.
ALIASES = {"FB": "META", "WLTW": "WTW", "PCLN": "BKNG", "CTRP": "TCOM", "FBHS": "FBIN",
           "NLOK": "GEN", "SYMC": "GEN", "RIMM": "BB", "BBRY": "BB"}
PEAK_WINDOWS = {
    "2018": ("2018-08-01", "2018-10-31"),
    "2020": ("2020-01-01", "2020-02-29"),
    "2022": ("2021-11-01", "2022-01-31"),
    "2025": ("2025-01-01", "2025-02-28"),
}


def official_roster(date, cache):
    day = date.date().isoformat()
    path = cache / f"{day}.xlsx"
    try:
        raw = path.read_bytes() if path.exists() else None
        if raw is None:
            url = f"{NASDAQ}?tradeDate={day}"
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=15) as response:
                disposition = response.headers.get("Content-Disposition", "")
                if f"EODWeightings_{date:%Y%m%d}_NDX.xlsx" not in disposition:
                    raise ValueError(f"Unexpected response for {day}")
                raw = response.read()
        frame = pd.read_excel(BytesIO(raw), header=None)
        header = frame.index[frame.iloc[:, 1].eq("Security Symbol")][0]
        members = frame.iloc[header + 1:, 1].dropna().astype(str).str.strip()
        if not 95 <= len(members) <= 110 or not members.is_unique:
            raise ValueError(f"Invalid roster for {day}")
        if not members.str.fullmatch(r"[A-Z0-9.\-]+|\$[A-Z0-9.\-]+\$").all():
            raise ValueError(f"Invalid ticker in {day}")
        if not path.exists():
            path.write_bytes(raw)
        return sorted(members), None
    except (OSError, ValueError, IndexError, BadZipFile) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def stock_close(symbol, cache, end):
    if symbol.startswith("$"):
        return None, "Nasdaq special symbol; no regular ticker price joined"
    path = cache / f"{symbol}-{end}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(symbol), None
    try:
        query = ALIASES.get(symbol, symbol).replace(".", "-")
        prices = adjusted_close(quote(query, safe=""), end).rename(symbol)
        if prices.empty:
            raise ValueError("Empty series")
        prices.to_csv(path)
        return prices, None
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def download_rosters(calendar, cache, workers):
    week_open = np.flatnonzero(~calendar.to_period("W-FRI").duplicated())
    decision_days = calendar[week_open[week_open > 0] - 1]
    decision_days = decision_days[decision_days >= pd.Timestamp(START) - pd.offsets.Day(14)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = pool.map(lambda date: official_roster(date, cache), decision_days)
        return dict(zip(decision_days, records, strict=True))


def download_prices(symbols, cache, end, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = pool.map(lambda symbol: stock_close(symbol, cache, end), symbols)
        return dict(zip(symbols, records, strict=True))


def weekly_panel(history, rosters, stock):
    closes = pd.DataFrame({code: series for code, (series, _) in stock.items() if series is not None})
    closes = closes.reindex(history.index)
    above = closes.gt(closes.rolling(50).mean()).where(closes.rolling(50).count().eq(50))
    rows = []
    for date, (members, _) in rosters.items():
        row = {"signal_date": date}
        if members is not None:
            votes = above.loc[date].reindex(members).dropna()
            row.update({
                "members": len(members), "known": len(votes), "coverage": len(votes) / len(members),
                "breadth": float(votes.mean()) if len(votes) else np.nan,
                "lower": float(votes.sum() / len(members)),
                "upper": float((votes.sum() + len(members) - len(votes)) / len(members)),
                "missing": ",".join(sorted(set(members).difference(votes.index))),
            })
        rows.append(row)
    panel = pd.DataFrame(rows).set_index("signal_date")
    location = history.index.get_indexer(panel.index)
    panel.index = history.index[location + 1]
    return panel


def warning_signals(history, panel, min_coverage):
    positions = history.index.get_indexer(panel.index) - 1
    qqq = history.qqqClose
    near = qqq / qqq.rolling(63).max() >= 0.97
    price_change = qqq.pct_change(20).iloc[positions].set_axis(panel.index)
    eligible = panel.coverage.ge(min_coverage)
    breadth = panel.breadth.where(eligible)
    deterioration = breadth.diff(4)
    cutoff = deterioration.shift().rolling(104, min_periods=26).quantile(0.20)
    price_cutoff = price_change.shift().rolling(104, min_periods=26).quantile(0.20)
    # Unknown members can only widen the possible breadth interval. This
    # conservative variant requires deterioration even at the two extremes.
    worst_case = panel.upper.where(eligible) - panel.lower.where(eligible).shift(4)
    compare = near.iloc[positions].set_axis(panel.index) & cutoff.notna() & deterioration.notna()
    return pd.DataFrame({
        "compare": compare,
        "breadth": compare & deterioration.lt(cutoff),
        "bounded": compare & worst_case.lt(cutoff),
        "price": compare & price_change.lt(price_cutoff),
        "repair": breadth.diff().gt(0),
    })


def fixed_window_events(history, panel, flags, bps, groups=None, *, weeks=4):
    """Use a common eligible start grid and report each non-overlapping phase."""
    opens = history.tqqqOpen
    offsets = history.index.get_indexer(panel.index)
    fee = bps / 10_000
    cost = np.log((1 - fee) / (1 + fee))
    rows = []
    for i in range(len(offsets) - weeks):
        start, end = offsets[i], offsets[i + weeks]
        rows.append({
            "date": panel.index[i], "phase": i % weeks,
            "net_exit": float(np.log(opens.iloc[start] / opens.iloc[end]) + cost),
            "forward_low": float(opens.iloc[start:end + 1].min() / opens.iloc[start] - 1),
        })
    events = pd.DataFrame(rows).set_index("date").join(flags)
    output = {}
    if groups is None:
        groups = {
            "all": flags["compare"], "breadth": flags.breadth,
            "bounded": flags.bounded, "price": flags.price,
            "other": flags["compare"] & ~flags.breadth,
        }
    for name, mask in groups.items():
        sample = events.loc[mask.reindex(events.index, fill_value=False)]
        output[name] = {
            "count": len(sample), "wins": int(sample.net_exit.gt(0).sum()),
            "net_mean": float(sample.net_exit.mean()) if len(sample) else None,
            "forward_low_mean": float(sample.forward_low.mean()) if len(sample) else None,
            "drawdown_20pct": int(sample.forward_low.le(-0.20).sum()),
            "without_2022": float(sample.loc[sample.index.year != 2022, "net_exit"].mean())
            if (sample.index.year != 2022).any() else None,
            "dates": [str(day.date()) for day in sample.index],
            "nonoverlap_phases": {
                phase: {"count": len(group), "mean": float(group.net_exit.mean())}
                for phase, group in sample.groupby("phase")
            },
        }
    return output


def early_overlay(trend, flags, warning, *, use_repair=False):
    weeks_left = 0
    path = []
    for date, invested in trend.items():
        if not invested:
            weeks_left = 0
        elif date in flags.index:
            if weeks_left:
                weeks_left -= 1
                if use_repair and flags.at[date, "repair"]:
                    weeks_left = 0
            elif flags.at[date, warning]:
                weeks_left = 4
        path.append(float(bool(invested) and not weeks_left))
    return pd.Series(path, index=trend.index)


def replay_overlays(history, flags, bps, warnings, *, include_repair=False, wave_windows=()):
    qqq = history.qqqClose
    sma = qqq.rolling(175).mean()
    trend = state_targets(open_signal(qqq.gt(sma)), open_signal(qqq.lt(0.99 * sma)))
    targets = {"hold": pd.Series(1.0, index=history.index), "sma175": trend}
    targets.update({name: early_overlay(trend, flags, name) for name in warnings})
    if include_repair:
        targets["breadth_repair"] = early_overlay(trend, flags, "breadth", use_repair=True)
    prices = history.loc[START:]
    quarter = prices.index[~prices.index.to_period("Q").duplicated()]
    base, _ = replay(prices, trend.loc[START:], bps)
    base_returns = daily_returns(base.nav)
    systems = {}
    for name, target in targets.items():
        frame, fills = replay(prices, target.loc[START:], bps)
        weight = matched_quarterly_weight(prices, frame.weight.mean())
        matched, matched_fills = replay(prices, pd.Series(weight, index=prices.index), bps, rebalance_on=quarter)
        returns = daily_returns(frame.nav)
        systems[name] = {
            **path_metrics(frame, fills),
            "sharpe_without_2022": float(sharpe(returns[prices.index.year != 2022])),
            "sharpe_interval_vs_sma": paired_interval(returns, base_returns),
            "matched": path_metrics(matched, matched_fills),
        }
        if wave_windows:
            active = pd.Series(np.log1p(returns) - np.log1p(base_returns), index=prices.index)
            contributions = {start: float(active.loc[start:end].sum()) for start, end in wave_windows}
            best = max(contributions, key=contributions.get)
            best_end = dict(wave_windows)[best]
            remaining = active.drop(active.loc[best:best_end].index)
            systems[name]["waves"] = {
                "log_excess": contributions, "best_start": best,
                "annual_log_excess_without_best": float(remaining.mean() * 252),
            }
    return systems


def evaluate(history, panel, bps, min_coverage):
    flags = warning_signals(history, panel, min_coverage)
    systems = replay_overlays(history, flags, bps, ("breadth", "bounded", "price"), include_repair=True)
    qqq = history.qqqClose
    waves = {}
    for name, (start, end) in PEAK_WINDOWS.items():
        peak = qqq.loc[start:end].idxmax()
        if pd.isna(peak):
            continue
        peak_window = flags.loc[peak - pd.offsets.Day(60):peak]
        waves[name] = {
            "qqq_high_date_in_window": str(peak.date()),
            "breadth_warnings_before_high": [str(day.date()) for day in peak_window.index[peak_window.breadth]],
            "coverage_at_last_decision": float(panel.loc[:peak, "coverage"].iloc[-1]),
        }
    return {"events": fixed_window_events(history, panel, flags, bps), "systems": systems, "waves": waves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    parser.add_argument("--history-cache", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preload-members", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    stock_cache = args.cache / "prices"
    roster_cache = args.cache / "rosters"
    for path in (stock_cache, roster_cache):
        path.mkdir(parents=True, exist_ok=True)
    if args.preload_members:
        members = pd.read_csv(args.preload_members, parse_dates=["date"])
        rows = members.loc[members.date.between("2012-06-01", args.end), "tickers"]
        symbols = sorted({code for row in rows for code in row.split(",") if re.fullmatch(r"[A-Z0-9.\-]+", code)})
        print(json.dumps({s: error for s, (_, error) in
                          download_prices(symbols, stock_cache, args.end, args.workers).items() if error}))
        return
    history = (pd.read_csv(args.history_cache, index_col=0, parse_dates=True)
               if args.history_cache else load_history(args.end)).loc[:args.end]
    rosters = download_rosters(history.index, roster_cache, args.workers)
    symbols = sorted({code for members, _ in rosters.values() if members for code in members})
    if not symbols:
        raise ValueError("No valid dated Nasdaq rosters were retrieved")
    stock = download_prices(symbols, stock_cache, args.end, args.workers)
    panel = weekly_panel(history, rosters, stock)
    coverage = panel.coverage.dropna()
    panel.to_csv(args.cache / "weekly_breadth.csv")
    for cutoff in ("2018-12-24", "2020-03-24", "2022-10-14", "2025-04-09"):
        pd.testing.assert_frame_equal(
            warning_signals(history.loc[:cutoff], panel.loc[:cutoff], 0.90),
            warning_signals(history, panel, 0.90).loc[:cutoff],
        )
    print(json.dumps({
        "rosters_ok": sum(members is not None for members, _ in rosters.values()),
        "rosters_failed": {str(date.date()): error for date, (_, error) in rosters.items() if error},
        "prices_ok": sum(series is not None for series, _ in stock.values()),
        "prices_failed": {symbol: error for symbol, (_, error) in stock.items() if error},
        "coverage": {"median": float(coverage.median()), "minimum": float(coverage.min()),
                     "at_least_90pct": int(coverage.ge(0.90).sum()),
                     "at_least_95pct": int(coverage.ge(0.95).sum()),
                     "by_year": coverage.groupby(coverage.index.year).agg(["count", "min", "median"])
                     .to_dict(orient="index")},
        "panel_sha256": hashlib.sha256(panel.to_csv().encode()).hexdigest(),
        "results": {f"{bps}bps_{coverage:.0%}": evaluate(history, panel, bps, coverage)
                    for bps, coverage in ((25, 0.85), (25, 0.90), (25, 0.95), (50, 0.90))},
    }, indent=2))


if __name__ == "__main__":
    main()
