"""Small JSON bridge for the broker-compatible XtQuant Python 3.11 SDK.

The main MoneyMore environment stays on Python 3.12.  This process only exposes
read-only market-data calls; it never imports xttrader or opens a trading account.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SDK = Path(
    os.getenv("MONEYMORE_XTQUANT_SDK", ROOT / ".runtime" / "xtquant_250807")
)
sys.path.insert(0, str(SDK))


def _connect():
    # Old broker clients print their connection banner to stdout.  stdout is
    # reserved for the JSON protocol, so keep the banner away from the caller.
    banner = io.StringIO()
    with contextlib.redirect_stdout(banner):
        from xtquant import xtdata

        xtdata.enable_hello = False
        port = os.getenv("MONEYMORE_XTQUANT_PORT")
        client = xtdata.connect("127.0.0.1", int(port)) if port else xtdata.connect()
    if not client:
        raise RuntimeError("unable to connect to the logged-in MiniQMT client")
    return xtdata


def _calendar(xtdata, start_date: str, end_date: str) -> list[dict[str, object]]:
    open_days = set(xtdata.get_trading_calendar("SH", start_date, end_date))
    dates = pd.date_range(
        pd.to_datetime(start_date, format="%Y%m%d"),
        pd.to_datetime(end_date, format="%Y%m%d"),
        freq="D",
    )
    previous = ""
    rows: list[dict[str, object]] = []
    for value in dates:
        day = value.strftime("%Y%m%d")
        is_open = day in open_days
        rows.append(
            {
                "exchange": "SSE",
                "cal_date": day,
                "is_open": "1" if is_open else "0",
                "pretrade_date": previous,
            }
        )
        if is_open:
            previous = day
    return rows


def _daily(xtdata, trade_date: str) -> list[dict[str, object]]:
    universe_path = ROOT / "data" / "processed" / "strategy_universe.parquet"
    if universe_path.exists():
        universe = pd.read_parquet(
            universe_path, columns=["effective_date", "symbol"]
        )
        universe["effective_date"] = universe["effective_date"].astype(str)
        eligible = universe.loc[universe["effective_date"] <= trade_date]
        effective = eligible["effective_date"].max()
        stocks = sorted(
            set(eligible.loc[eligible["effective_date"] == effective, "symbol"].astype(str))
        )
    else:
        stocks = sorted(set(xtdata.get_stock_list_in_sector("沪深A股")))
    execution_symbols: set[str] = set()
    paper_database = ROOT / "state" / "paper_orders.sqlite3"
    if paper_database.exists():
        with sqlite3.connect(paper_database) as connection:
            execution_symbols = {
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT symbol FROM orders WHERE status = 'PENDING'"
                )
            }
    stocks = sorted(set(stocks) | execution_symbols)
    frames = xtdata.get_market_data_ex(
        [], stocks, "1d", trade_date, trade_date, -1, "none", False
    )
    missing = [
        symbol
        for symbol in stocks
        if symbol not in frames
        or frames[symbol] is None
        or frames[symbol].empty
        or trade_date not in set(frames[symbol].index.astype(str))
    ]
    # MiniQMT normally maintains the whole-market daily cache.  After an
    # offline gap, repair only missing symbols in bounded chunks: the old
    # Huatai broker kernel can stall on a single 5,000-symbol download call.
    for offset in range(0, len(missing), 200):
        chunk = missing[offset : offset + 200]
        xtdata.download_history_data2(chunk, "1d", trade_date, trade_date)
        frames.update(
            xtdata.get_market_data_ex(
                [], chunk, "1d", trade_date, trade_date, -1, "none", False
            )
        )
    rows: list[dict[str, object]] = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        selected = frame.loc[frame.index.astype(str) == trade_date]
        if selected.empty:
            continue
        row = selected.iloc[-1]
        if int(row.get("suspendFlag", 0) or 0) != 0:
            continue
        detail = (
            xtdata.get_instrument_detail(symbol, True) or {}
            if symbol in execution_symbols
            and trade_date == datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
            else {}
        )
        close = float(row["close"])
        pre_close = float(row.get("preClose", 0.0) or 0.0)
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": trade_date,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": close,
                "pre_close": pre_close,
                "change": close - pre_close,
                "pct_chg": ((close / pre_close) - 1.0) * 100 if pre_close else None,
                # XtQuant stock volume is in lots, matching Tushare's vol.
                "vol": float(row["volume"]),
                # XtQuant amount is CNY; Tushare amount is thousand CNY.
                "amount": float(row["amount"]) / 1000.0,
                "suspend_flag": int(row.get("suspendFlag", 0) or 0),
                "up_limit": (
                    float(detail["UpStopPrice"]) if detail.get("UpStopPrice") else None
                ),
                "down_limit": (
                    float(detail["DownStopPrice"])
                    if detail.get("DownStopPrice")
                    else None
                ),
                "price_tick": (
                    float(detail["PriceTick"]) if detail.get("PriceTick") else None
                ),
                "instrument_status": (
                    int(detail["InstrumentStatus"])
                    if detail.get("InstrumentStatus") is not None
                    else None
                ),
            }
        )
    return rows


def _intraday(xtdata, symbols: list[str]) -> list[dict[str, object]]:
    ticks = xtdata.get_full_tick(symbols) or {}
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        tick = ticks.get(symbol) or {}
        price = float(tick.get("lastPrice", 0) or 0)
        volume = float(tick.get("volume", 0) or 0)
        amount = float(tick.get("amount", 0) or 0)
        raw_vwap = amount / volume if volume > 0 else price
        # Broker builds have reported stock volume in either shares or lots.
        # Select the unit whose implied VWAP is closest to the live price.
        candidates = [raw_vwap, raw_vwap / 100, raw_vwap * 100]
        vwap = min(candidates, key=lambda value: abs(value - price)) if price else raw_vwap
        bid_prices = tick.get("bidPrice") or []
        ask_prices = tick.get("askPrice") or []
        detail = xtdata.get_instrument_detail(symbol, True) or {}
        rows.append(
            {
                "symbol": symbol,
                "timestamp": int(tick.get("time", 0) or 0),
                "last": price,
                "open": float(tick.get("open", 0) or 0),
                "high": float(tick.get("high", 0) or 0),
                "low": float(tick.get("low", 0) or 0),
                "pre_close": float(tick.get("lastClose", 0) or 0),
                "volume": volume,
                "amount": amount,
                "vwap": float(vwap),
                "bid1": float(bid_prices[0]) if bid_prices else 0.0,
                "ask1": float(ask_prices[0]) if ask_prices else 0.0,
                "up_limit": float(detail.get("UpStopPrice", 0) or 0),
                "down_limit": float(detail.get("DownStopPrice", 0) or 0),
            }
        )
    return rows


def _minute_history(xtdata, symbols: list[str], start: str, end: str) -> list[dict[str, object]]:
    """Read actual QMT one-minute bars for an explicitly dated paper replay."""
    rows: list[dict[str, object]] = []
    for offset in range(0, len(symbols), 50):
        chunk = symbols[offset : offset + 50]
        xtdata.download_history_data2(chunk, "1m", start, end)
        frames = xtdata.get_market_data_ex(
            ["open", "high", "low", "close", "volume", "amount"],
            chunk, period="1m", start_time=start, end_time=end,
            count=-1, fill_data=False,
        )
        for symbol, frame in frames.items():
            for timestamp, bar in frame.iterrows():
                rows.append({
                    "symbol": symbol, "timestamp": str(timestamp),
                    **{field: float(bar[field]) for field in
                       ("open", "high", "low", "close", "volume", "amount")},
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["probe", "calendar", "daily", "intraday", "minute-history"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--trade-date")
    parser.add_argument("--symbols")
    args = parser.parse_args()
    xtdata = _connect()
    if args.action == "probe":
        ticks = xtdata.get_full_tick(["600036.SH"]) or {}
        if not ticks.get("600036.SH"):
            raise RuntimeError("QMT RPC connected but market-data function returned no tick")
        stocks = xtdata.get_stock_list_in_sector("沪深A股")
        detail = xtdata.get_instrument_detail("600036.SH") or {}
        result: object = {
            "connected": True,
            "stock_count": len(stocks),
            "sample_name": detail.get("InstrumentName", ""),
        }
    elif args.action == "calendar":
        if not args.start or not args.end:
            parser.error("calendar requires --start and --end")
        result = _calendar(xtdata, args.start, args.end)
    elif args.action == "daily":
        if not args.trade_date:
            parser.error("daily requires --trade-date")
        result = _daily(xtdata, args.trade_date)
    elif args.action == "minute-history":
        symbols = sorted(set(filter(None, (args.symbols or "").split(","))))
        if not symbols or not args.start or not args.end:
            parser.error("minute-history requires --symbols, --start and --end")
        result = _minute_history(xtdata, symbols, args.start, args.end)
    else:
        symbols = sorted(set(filter(None, (args.symbols or "").split(","))))
        if not symbols:
            parser.error("intraday requires --symbols")
        result = _intraday(xtdata, symbols)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
