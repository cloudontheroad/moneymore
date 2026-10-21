"""Backfill missed *paper* intraday sessions from dated QMT one-minute bars.

This is an estimated counterfactual: historical bid/ask and queue position are
unavailable.  It must never be presented as a live QMT or broker fill.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.data.realtime import create_realtime_provider
from moneymore.execution.paper import ExecutionBar, PaperBroker
from moneymore.intraday_execution import (
    INTRADAY_ACCOUNT, EXPOSURE_INTRADAY_ACCOUNT, decide_intraday_execution,
)
from moneymore.multi_sector_daily import MULTI_SECTOR_ACCOUNT
from moneymore.baseline_exposure_daily import BASELINE_EXPOSURE_ACCOUNT

ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = ZoneInfo("Asia/Shanghai")
PROVENANCE = "HISTORICAL_REPLAY_1M_ESTIMATED_NO_BOOK"


def source_target(database: Path, trade_date: str, plan_date: str,
                  source_account: str = MULTI_SECTOR_ACCOUNT) -> dict[str, int]:
    """Reconstruct the baseline's target as it stood on a historical day."""
    with sqlite3.connect(database) as connection:
        quantities = dict(connection.execute(
            "SELECT symbol, quantity FROM positions WHERE account_id = ?",
            (source_account,),
        ))
        for symbol, side, quantity in connection.execute(
            "SELECT symbol, side, quantity FROM fills WHERE account_id = ? AND trade_date > ?",
            (source_account, trade_date),
        ):
            quantities[symbol] = quantities.get(symbol, 0) - (
                int(quantity) if side == "BUY" else -int(quantity)
            )
        # Include the portion of that day's plan not filled by that day.
        for key, symbol, side, quantity in connection.execute(
            "SELECT idempotency_key, symbol, side, quantity FROM orders "
            "WHERE account_id = ? AND signal_date = ?",
            (source_account, plan_date),
        ):
            filled = connection.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM fills "
                "WHERE idempotency_key = ? AND trade_date <= ?", (key, trade_date),
            ).fetchone()[0]
            remaining = max(0, int(quantity) - int(filled))
            if remaining:
                quantities[symbol] = quantities.get(symbol, 0) + (
                    remaining if side == "BUY" else -remaining
                )
    return {symbol: max(0, quantity) for symbol, quantity in quantities.items()}


def historical_bars(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    provider = create_realtime_provider(ROOT)
    if getattr(provider, "name", "") == "alphafeed":
        start_dt = datetime.strptime(start if len(start) == 14 else start + "000000", "%Y%m%d%H%M%S")
        end_dt = datetime.strptime(end if len(end) == 14 else end + "235959", "%Y%m%d%H%M%S")
        frame = provider.minute_history(symbols, start_dt.isoformat(), end_dt.isoformat())
        if frame.empty:
            raise RuntimeError("AlphaFeed returned no historical minute bars")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert(
            SHANGHAI).dt.strftime("%Y%m%d%H%M%S")
        return frame.sort_values(["timestamp", "symbol"])
    command = [
        str(ROOT / ".runtime/qmt-py311/Scripts/python.exe"),
        str(ROOT / "scripts/qmt_data_bridge.py"), "minute-history",
        "--symbols", ",".join(symbols), "--start", start, "--end", end,
    ]
    # XtQuant's Windows process may emit GBK/GB18030 diagnostics. Decode them
    # explicitly so a disconnected client produces a useful CalledProcessError
    # rather than a background UnicodeDecodeError that hides stderr.
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="gb18030",
        errors="replace",
        timeout=300,
        check=True,
    )
    frame = pd.DataFrame(json.loads(result.stdout))
    if frame.empty:
        raise RuntimeError("QMT returned no historical minute bars")
    frame["timestamp"] = frame["timestamp"].astype(str)
    return frame.sort_values(["timestamp", "symbol"])


def replay_day(broker: PaperBroker, bars: pd.DataFrame, trade_date: str,
               plan_date: str, *, account_id: str, source_account: str,
               daily_table: str, audit_subdir: str, commit: bool) -> dict[str, object]:
    target = source_target(broker.database, trade_date, plan_date, source_account)
    with sqlite3.connect(broker.database) as connection:
        held = dict(connection.execute(
            "SELECT symbol, quantity FROM positions WHERE account_id = ?",
            (account_id,),
        ))
        pending = connection.execute(
            "SELECT COUNT(*) FROM orders WHERE account_id = ? AND status = 'PENDING'",
            (account_id,),
        ).fetchone()[0]
    if not commit and pending:
        # A preview includes the current next-day preparation, which is
        # superseded during the actual replay.
        pass
    intents = []
    for symbol in sorted(set(target) | set(held)):
        difference = target.get(symbol, 0) - held.get(symbol, 0)
        if difference:
            intents.append({"symbol": symbol,
                            "side": "BUY" if difference > 0 else "SELL",
                            "quantity": abs(difference)})
    day = bars[bars["timestamp"].str.startswith(trade_date)].copy()
    available = set(day["symbol"])
    missing = sorted({row["symbol"] for row in intents} - available)
    if missing:
        raise RuntimeError(f"missing QMT 1m history for {trade_date}: {missing}")
    if not commit:
        return {"trade_date": trade_date, "plan_date": plan_date,
                "intents": intents, "minute_rows": len(day), "missing": missing}

    # Existing stale preparation is not a historical fill.  Recreate dated
    # intents against the latest as-of target; preserve a cancellation trail.
    broker.cancel_pending(account_id, "HISTORICAL_REPLAY_SUPERSEDED")
    with sqlite3.connect(broker.database) as connection:
        for row in intents:
            connection.execute(
                "INSERT INTO orders(idempotency_key,created_at,status,strategy_id,"
                "symbol,side,quantity,signal_date,reason_code,account_id) "
                "VALUES (?,?, 'PENDING', ?,?,?,?,?,?,?)",
                (f"{account_id}:replay:{plan_date}:{row['symbol']}:{row['side']}",
                 datetime.now(SHANGHAI).isoformat(),
                 f"adaptive_vwap_sniper_v1:historical_replay:{source_account}", row["symbol"],
                 row["side"], row["quantity"], plan_date, PROVENANCE, account_id),
            )
    day["vwap"] = (
        day.groupby("symbol")["amount"].cumsum()
        / day.groupby("symbol")["volume"].cumsum().replace(0, pd.NA) / 100
    )
    opens = day.groupby("symbol")["open"].first().to_dict()
    config = BacktestConfig.from_yaml(ROOT / "configs/default.yaml")
    events = []
    for timestamp, minute in day.groupby("timestamp", sort=True):
        clock = datetime.strptime(timestamp, "%Y%m%d%H%M%S").time()
        if not (time(9, 35) <= clock <= time(11, 30) or
                time(13, 0) <= clock <= time(15, 0)):
            continue
        by_symbol = {row["symbol"]: row for row in minute.to_dict("records")}
        for intent in sorted(intents, key=lambda row: row["side"] != "SELL"):
            symbol = intent["symbol"]
            quote = by_symbol.get(symbol)
            if quote is None:
                continue
            with sqlite3.connect(broker.database) as connection:
                outstanding = connection.execute(
                    "SELECT COUNT(*) FROM orders WHERE account_id = ? AND symbol = ? "
                    "AND status = 'PENDING' AND signal_date = ?",
                    (account_id, symbol, plan_date),
                ).fetchone()[0]
            if not outstanding:
                continue
            price = float(quote["close"])
            vwap = float(quote["vwap"])
            decision = decide_intraday_execution(
                symbol=symbol, side=intent["side"], price=price,
                open_price=float(opens[symbol]), vwap=vwap,
                bid1=0.0, ask1=0.0, now_time=clock,
            )
            if decision.action != "EXECUTE":
                continue
            outcomes = broker.execute_pending(
                ExecutionBar(symbol=symbol, trade_date=trade_date,
                             open=price, close=price, can_buy=True, can_sell=True),
                config, account_id,
            )
            events.extend({"timestamp": timestamp, "symbol": symbol,
                           "price": price, "decision": decision.reason,
                           "outcome": outcome} for outcome in outcomes)
    # Mark to official close where available, including unchanged holdings.
    closes = day.groupby("symbol")["close"].last().to_dict()
    daily = ParquetStore(ROOT / "data").read(
        "daily", columns=["ts_code", "trade_date", "close"],
        filters=[("trade_date", "==", trade_date)],
    )
    closes.update(dict(zip(daily["ts_code"], daily["close"])))
    account = broker.account_snapshot(closes, account_id)
    row = pd.DataFrame([{**{key: account[key] for key in
                             ("cash", "market_value", "equity")},
                         "account_id": account_id, "trade_date": trade_date,
                         "observed_at": f"{trade_date}T15:00:00+08:00",
                         "gross_exposure": account["market_value"] / account["equity"]
                         if account["equity"] else 0.0}])
    if broker.database.resolve() == (ROOT / "state/paper_orders.sqlite3").resolve():
        ParquetStore(ROOT / "data").merge_curated(
            daily_table, [row], ["account_id", "trade_date"],
        )
    audit_dir = ROOT / "state" / audit_subdir
    if broker.database.resolve() == (ROOT / "state/paper_orders.sqlite3").resolve():
        audit_dir.mkdir(exist_ok=True)
        observation = {
            "trade_date": trade_date, "status": "HISTORICAL_REPLAY_ESTIMATED",
            "observed_at": datetime.now(SHANGHAI).isoformat(),
            "account_id": account_id, "pending_orders": 0,
            "decisions": [], "executions": events, "account": account,
        }
        (audit_dir / f"{trade_date}.json").write_text(json.dumps({
        "trade_date": trade_date, "status": "HISTORICAL_REPLAY_ESTIMATED",
        "provenance": PROVENANCE, "price_source": str(day.iloc[-1].get("source") or "QMT_1M_CLOSE"),
        "limitations": "No historical bid/ask, queue or true tick VWAP; 1m close and cumulative 1m VWAP used.",
        "plan_date": plan_date, "intents": intents, "events": events,
        "account": account, "observations": [observation],
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"trade_date": trade_date, "status": "HISTORICAL_REPLAY_ESTIMATED",
            "intents": len(intents), "events": len(events),
            "fills": len(broker.fills(account_id, trade_date)),
            "equity": account["equity"]}


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--account", choices=["baseline", "controlled", "all"],
                        default="baseline")
    parser.add_argument("--trade-date")
    parser.add_argument("--plan-date")
    parser.add_argument("--database", type=Path,
                        default=ROOT / "state/paper_orders.sqlite3")
    args = parser.parse_args()
    if bool(args.trade_date) != bool(args.plan_date):
        parser.error("--trade-date and --plan-date must be supplied together")
    dates = ([(args.trade_date, args.plan_date)] if args.trade_date else
             [("20260916", "20260915"), ("20260917", "20260916")])
    account_specs = {
        "baseline": (INTRADAY_ACCOUNT, MULTI_SECTOR_ACCOUNT,
                     "baseline_intraday_account_daily", "intraday-execution"),
        "controlled": (EXPOSURE_INTRADAY_ACCOUNT, BASELINE_EXPOSURE_ACCOUNT,
                       "baseline_pysystemtrade_intraday_account_daily",
                       "baseline-pysystemtrade-intraday-execution"),
    }
    selected = list(account_specs) if args.account == "all" else [args.account]
    broker = PaperBroker(args.database)
    for account_name in selected:
        account_id, source_account, daily_table, audit_subdir = account_specs[account_name]
        with sqlite3.connect(args.database) as connection:
            held = {row[0] for row in connection.execute(
                "SELECT symbol FROM positions WHERE account_id = ?", (account_id,),
            )}
        symbols = sorted(held | set().union(*(
            source_target(args.database, day, plan, source_account)
            for day, plan in dates
        )))
        bars = historical_bars(symbols, dates[0][0], dates[-1][0])
        if args.commit and any(broker.fills(account_id, day) for day, _ in dates):
            raise RuntimeError(
                f"{account_id} already has replay-date fills; refusing a second commit"
            )
        for day, plan in dates:
            print(json.dumps(replay_day(
                broker, bars, day, plan, account_id=account_id,
                source_account=source_account, daily_table=daily_table,
                audit_subdir=audit_subdir, commit=args.commit),
                             ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
