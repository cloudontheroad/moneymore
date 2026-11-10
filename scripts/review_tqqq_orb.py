"""Read-only replication of the baseline Zarattini–Aziz 5-minute ORB."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

SYMBOLS = ("QQQ", "TQQQ")
PRICE = ["open", "high", "low", "close"]
PUBLIC_ARCHIVE = "https://huggingface.co/datasets/syncster-data/historical-market-data/resolve/main/data/stocks/1m"
FIRSTRATE_SAMPLE = "https://frd001.s3.us-east-2.amazonaws.com/frd_sample_etf_"


def load(symbol, start, end, minutes, path=None, source="yahoo"):
    if path:
        frame = pd.read_csv(path)
        frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True)
        return frame.sort_index(), hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if source == "firstrate_sample":
        with urllib.request.urlopen(f"{FIRSTRATE_SAMPLE}{symbol}.zip", timeout=30) as response:
            raw = response.read()
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            with archive.open(f"{symbol}_1min_sample.csv") as minute_file:
                frame = pd.read_csv(minute_file)
        frame.index = (pd.to_datetime(frame.pop("timestamp"))
                       .dt.tz_localize("America/New_York").dt.tz_convert("UTC"))
        return frame.sort_index(), hashlib.sha256(raw).hexdigest()
    if source == "syncster":
        url = f"{PUBLIC_ARCHIVE}/{symbol[0]}/{symbol}.parquet"
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = response.read()
        frame = pd.read_parquet(BytesIO(raw))
        if not frame.symbol.eq(symbol).all():
            raise ValueError(f"Unexpected symbol in {symbol} archive")
        frame.index = pd.to_datetime(frame.pop("ts"), utc=True)
        return frame.sort_index(), hashlib.sha256(raw).hexdigest()
    blocks = []
    digest = hashlib.sha256()
    chunk = timedelta(days=6 if minutes == 1 else 54)
    cursor = start
    while cursor < end:
        stop = min(cursor + chunk, end)
        url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?period1={int(cursor.timestamp())}&period2={int(stop.timestamp())}&interval={minutes}m")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
            raw = response.read()
        digest.update(raw)
        result = json.loads(raw)["chart"]["result"][0]
        frame = pd.DataFrame(result["indicators"]["quote"][0],
                             index=pd.to_datetime(result["timestamp"], unit="s", utc=True))
        blocks.append(frame)
        cursor = stop
    return pd.concat(blocks).sort_index(), digest.hexdigest()


def split_factors(symbol, first, last):
    start = int(pd.Timestamp(first, tz="UTC").timestamp())
    end = int((pd.Timestamp(last, tz="UTC") + timedelta(days=1)).timestamp())
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={start}&period2={end}&interval=1mo&events=splits")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
        result = json.load(response)["chart"]["result"][0]
    return {
        pd.Timestamp(event["date"], unit="s", tz="UTC").tz_convert("America/New_York").date().isoformat():
        float(event["numerator"]) / float(event["denominator"])
        for event in result.get("events", {}).get("splits", {}).values()
    }


def complete_sessions(frame, calendar, start, end, minutes):
    sessions, rejected = {}, []
    for day in calendar.sessions_in_range(start, end):
        opening, close = calendar.session_open(day), calendar.session_close(day)
        wanted = pd.date_range(opening, close, freq=f"{minutes}min", inclusive="left")
        left, right = frame.index.searchsorted([opening, close])
        bars = frame.iloc[left:right]
        date = str(day.date())
        if bars.index.has_duplicates or not wanted.isin(bars.index).all():
            rejected.append(date)
            continue
        prices = bars.loc[wanted, PRICE]
        valid = (
            np.isfinite(prices.to_numpy()).all() and (prices > 0).all().all()
            and prices.high.ge(prices[["open", "close", "low"]].max(axis=1)).all()
            and prices.low.le(prices[["open", "close", "high"]].min(axis=1)).all()
        )
        if valid:
            sessions[date] = prices.to_numpy(dtype=float)
        else:
            rejected.append(date)
    return sessions, rejected


def trade_day(bars, equity, *, minutes, leverage, bps, long_only=False, atr_ratio=None):
    opening_range = bars[:5 // minutes]
    after = bars[5 // minutes:]
    direction = int(np.sign(opening_range[-1, 3] - opening_range[0, 0]))
    if direction == 0 or (long_only and direction < 0):
        return None
    if atr_ratio is None:
        stop = opening_range[:, 2].min() if direction > 0 else opening_range[:, 1].max()
    else:
        stop = after[0, 0] - direction * 0.05 * atr_ratio * opening_range[0, 0]
    entry = after[0, 0] * (1 + direction * bps / 10_000)
    risk = direction * (entry - stop)
    # A gap beyond the stop invalidates the trade at entry.
    if direction * (after[0, 0] - stop) <= 0 or risk <= 0:
        return None
    shares = int(min(equity * 0.01 / risk, leverage * equity / (entry + 0.0005)))
    if shares <= 0:
        return None
    target = entry + direction * (10 * risk if atr_ratio is None else np.inf)
    exit_price, reason = after[-1, 3], "close"
    exit_bar = len(bars) - 1
    for index, (opening, high, low, _) in enumerate(after, start=5 // minutes):
        adverse = low if direction > 0 else high
        favorable = high if direction > 0 else low
        if direction * (opening - stop) <= 0:
            exit_price, reason = opening, "stop_gap"
        elif direction * (opening - target) >= 0:
            exit_price, reason = target, "target"
        elif direction * (adverse - stop) <= 0:
            # When both barriers trade in the same bar, the stop takes priority.
            exit_price, reason = stop, "stop"
        elif direction * (favorable - target) >= 0:
            exit_price, reason = target, "target"
        else:
            continue
        exit_bar = index
        break
    exit_price *= 1 - direction * bps / 10_000
    pnl = shares * (direction * (exit_price - entry) - 0.001)
    return {"pnl": pnl, "reason": reason, "direction": direction,
            "r_multiple": pnl / (shares * risk), "shares": shares,
            "entry_price": float(entry), "exit_price": float(exit_price), "exit_bar": exit_bar}


def statistics(nav):
    nav = pd.Series(nav)
    returns = nav.pct_change().dropna()
    sd = returns.std()
    return {"return": float(nav.iloc[-1] / nav.iloc[0] - 1),
            "sharpe": float(returns.mean() / sd * np.sqrt(252)) if sd > 0 else None,
            "max_drawdown": float((nav / nav.cummax() - 1).min())}


def simulate(sessions, dates, *, minutes, leverage, bps, initial_cash, long_only=False, atr_ratios=None):
    nav = [initial_cash]
    trades = []
    for day in dates:
        atr_ratio = None if atr_ratios is None else atr_ratios[day]
        trade = trade_day(sessions[day], nav[-1], minutes=minutes, leverage=leverage, bps=bps,
                          long_only=long_only, atr_ratio=atr_ratio)
        nav.append(nav[-1] + trade["pnl"] if trade else nav[-1])
        if trade:
            trades.append({"date": day, **trade})
    return {**statistics(nav), "trades": len(trades),
            "win_rate": sum(t["pnl"] > 0 for t in trades) / len(trades) if trades else None,
            "mean_r": float(np.mean([t["r_multiple"] for t in trades])) if trades else None,
            "trade_log": trades}


def prior_atr_ratios(frame):
    """Causal 14-session simple ATR, scaled by the prior adjusted open."""
    frame = frame.sort_values("timestamp")
    dates = frame.timestamp.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    previous_close = frame.close.shift()
    adjustment = (1 - frame.cash_dividend.div(previous_close).fillna(0)).rdiv(1).cumprod()
    high, low, opening = (frame[key] * adjustment for key in ("high", "low", "open"))
    prev = (frame.close * adjustment).shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    ratios = (tr.rolling(14, min_periods=14).mean() / opening).shift()
    return dict(zip(dates[ratios.gt(0)], ratios[ratios.gt(0)], strict=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-09-15")
    parser.add_argument("--end", default="2026-11-06", help="Inclusive New York session date")
    parser.add_argument("--minutes", type=int, choices=(1, 5), default=5)
    parser.add_argument("--source", choices=("yahoo", "syncster", "firstrate_sample"), default="yahoo")
    parser.add_argument("--atr", action="store_true", help="Also evaluate the paper's 5%-of-ATR stop variant")
    parser.add_argument("--qqq-csv", help="CSV with UTC timestamp and lowercase OHLC, labeled at bar open")
    parser.add_argument("--tqqq-csv", help="CSV with UTC timestamp and lowercase OHLC, labeled at bar open")
    args = parser.parse_args()
    if bool(args.qqq_csv) != bool(args.tqqq_csv):
        parser.error("Supply both CSVs or neither")
    if args.source != "yahoo" and (args.minutes != 1 or args.qqq_csv):
        parser.error("Public archives require --minutes 1 without CSVs")
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + timedelta(days=1)
    calendar = xcals.get_calendar("XNYS", start=(start - timedelta(days=7)).date().isoformat(),
                                  end=(end + timedelta(days=7)).date().isoformat())
    sessions, raw_sessions, rejected, hashes, splits = {}, {}, {}, {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {symbol: pool.submit(load, symbol, start, end, args.minutes, path, args.source)
                for symbol, path in (("QQQ", args.qqq_csv), ("TQQQ", args.tqqq_csv))}
        loaded = {symbol: job.result() for symbol, job in jobs.items()}
    for symbol, (data, digest) in loaded.items():
        hashes[symbol] = digest
        sessions[symbol], rejected[symbol] = complete_sessions(data, calendar, args.start, args.end, args.minutes)
        if args.source == "syncster":
            splits[symbol] = split_factors(symbol, str(data.index[0].date()), str(data.index[-1].date()))
            raw_sessions[symbol] = {
                day: bars * np.prod([ratio for split, ratio in splits[symbol].items() if day < split])
                for day, bars in sessions[symbol].items()
            }
        else:
            raw_sessions[symbol] = sessions[symbol]
    dates = sorted(set(sessions["QQQ"]) & set(sessions["TQQQ"]))
    if len(dates) < 2:
        raise ValueError(f"Insufficient common complete sessions. Rejected: {rejected}")
    atr = {}
    if args.atr:
        from sync_us_yahoo_adjusted_history import download

        warmup = (start - timedelta(days=90)).date().isoformat()
        for symbol in SYMBOLS:
            atr[symbol] = prior_atr_ratios(download(symbol, warmup, args.end))
            missing = set(dates) - atr[symbol].keys()
            if missing:
                raise ValueError(f"{symbol} lacks prior daily ATR on {len(missing)} complete minute sessions")

    def summarize(days):
        results = {}
        for symbol in SYMBOLS:
            adjusted, bars = sessions[symbol], raw_sessions[symbol]
            benchmark = [1.0] + [adjusted[d][-1, 3] / adjusted[days[0]][0, 0] for d in days]
            results[symbol] = {
                "buy_hold_price_return": statistics(benchmark),
                **{f"paper_4x_{bps}bps": simulate(bars, days, minutes=args.minutes, leverage=4,
                                             bps=bps, initial_cash=25_000) for bps in (0, 5, 10)},
                "long_only_1x_5bps": simulate(bars, days, minutes=args.minutes, leverage=1,
                                             bps=5, initial_cash=100_000, long_only=True),
            }
            if args.atr:
                results[symbol].update({
                    f"atr_4x_{bps}bps": simulate(bars, days, minutes=args.minutes, leverage=4,
                                                bps=bps, initial_cash=25_000, atr_ratios=atr[symbol])
                    for bps in (0, 1, 2, 5, 10)
                })
                results[symbol]["atr_long_only_1x_5bps"] = simulate(
                    bars, days, minutes=args.minutes, leverage=1, bps=5, initial_cash=100_000,
                    long_only=True, atr_ratios=atr[symbol],
                )
        return results

    windows = {}
    for name, lower, upper in (
        ("paper", "2016-01-01", "2023-02-17"),
        ("after_paper", "2023-02-21", args.end),
    ):
        subset = [day for day in dates if lower <= day <= upper]
        if len(subset) >= 2:
            windows[name] = {"first": subset[0], "last": subset[-1], "days": len(subset),
                             "results": summarize(subset)}
    print(json.dumps({
        "source": "CSV" if args.qqq_csv else args.source,
        "minutes": args.minutes, "common_dates": dates, "rejected": rejected,
        "input_sha256": hashes, "splits": splits, "results": summarize(dates), "windows": windows,
    }, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
