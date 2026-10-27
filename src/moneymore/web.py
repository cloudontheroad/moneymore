from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time as clock
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .bank_daily import BANK_ACCOUNT, run_bank_daily_pipeline
from .baseline_exposure_daily import (
    ASYMMETRIC_EXPOSURE_ACCOUNT,
    ASYMMETRIC_EXPOSURE_HISTORY,
    BASELINE_EXPOSURE_ACCOUNT,
    BASELINE_EXPOSURE_HISTORY,
    EXPOSURE_VARIANTS,
    TREND_CAP_EXPOSURE_ACCOUNT,
    TREND_CAP_EXPOSURE_HISTORY,
    run_baseline_exposure_daily,
)
from .challenger_diagnostics import run_challenger_review
from .config import BacktestConfig
from .data.fundamental_sync import (
    sync_daily_basic_universe,
    sync_missing_target_fundamentals,
)
from .data.fundamental_research import financial_quality, research_financial_frame
from .data.qmt_provider import create_market_data_provider
from .data.realtime import create_realtime_provider
from .data.research import load_total_return_stock_bars
from .data.store import ParquetStore
from .desktop_routes import router as desktop_router
from .execution.paper import PaperBroker
from .factors import build_default_registry, build_qmt_candidate_registry
from .intraday_execution import (
    EXPOSURE_INTRADAY_ACCOUNT,
    INTRADAY_ACCOUNT,
    POLICY_ID,
    prepare_exposure_intraday_branch_orders,
    prepare_intraday_branch_orders,
    run_baseline_intraday_once,
)
from .leveraged_etf import LeveragedEtfPaper, LeveragedEtfScheduler
from .long_term_review import evaluate_long_term_review
from .market_risk import build_market_risk_snapshot
from .model_registry import ModelRegistry
from .monthly_acceptance import evaluate_monthly_cycle
from .multi_sector_daily import (
    MULTI_SECTOR_ACCOUNT,
    run_multi_sector_daily,
)
from .open_execution import execute_accounts_at_open, execute_accounts_at_qmt_open
from .point_in_time import materialize_point_in_time_store
from .portfolio_constructor import adjusted_close_panel, trailing_return_correlation
from .qlib_candidate_observer import (
    CANDIDATE_HISTORY_TABLE,
    candidate_account_id,
    candidate_catalog,
    run_candidate_queue_daily,
)
from .qlib_challenger_daily import (
    QLIB_CHALLENGER_ACCOUNT,
    run_qlib_challenger_daily,
)
from .qlib_governance import (
    bootstrap_qlib_release,
    evaluate_qlib_drift,
)
from .research.adaptive import research_weighted_strategies
from .research.detail import (
    allocation_comparison,
    annual_comparison,
    candidate_result,
    trade_ledger,
)
from .research.governance import evaluate_bank_model_promotion
from .research.single_stock import research_single_stock, robustness_single_stock
from .signals import trend_decision
from .strategy_comparison import build_branch_history, build_fair_comparison
from .strategy_universe import (
    active_strategy_universe,
    refresh_market_cap_universe,
    universe_summary,
)
from .trade_cycles import analyze_trade_cycles
from .weekly_training import WeeklyTrainingService

ROOT = Path(__file__).resolve().parents[2]
# Resolve configured providers before health checks and startup schedulers run;
# otherwise a fresh process temporarily advertises the legacy QMT defaults.
load_dotenv(ROOT / ".env", override=False)
STATE = ROOT / "state"
DATA = ROOT / "data"
PAPER_DATABASE = STATE / "paper_orders.sqlite3"
SERVICE_DATABASE = STATE / "service.sqlite3"
LEVERAGED_ETF_DATABASE = STATE / "tqqq_sqqq_paper.sqlite3"
LEVERAGED_ETF_CACHE = DATA / "leveraged-etf"
leveraged_etf_paper_service = LeveragedEtfPaper(
    LEVERAGED_ETF_DATABASE, LEVERAGED_ETF_CACHE
)
leveraged_etf_scheduler = LeveragedEtfScheduler(leveraged_etf_paper_service)
SYMBOL = "600036.SH"
ACCOUNT = "default"
SHANGHAI = ZoneInfo("Asia/Shanghai")
INTRADAY_BASELINE_ACTIVATION_DATE = "20260908"
EXPOSURE_INTRADAY_ACTIVATION_DATE = "20260916"
_LIVE_ACCOUNT_CACHE: dict[str, object] = {"expires_at": 0.0, "payload": {}}
_LIVE_ACCOUNT_LOCK = threading.Lock()


def _live_account_snapshot() -> dict[str, object]:
    """Return live paper ledgers marked with one batched QMT quote request."""
    now = datetime.now(SHANGHAI)
    with _LIVE_ACCOUNT_LOCK:
        if clock.monotonic() < float(_LIVE_ACCOUNT_CACHE["expires_at"]):
            return dict(_LIVE_ACCOUNT_CACHE["payload"])
        broker = PaperBroker(PAPER_DATABASE)
        with sqlite3.connect(PAPER_DATABASE) as connection:
            connection.row_factory = sqlite3.Row
            account_ids = [
                str(row["account_id"])
                for row in connection.execute("SELECT account_id FROM accounts ORDER BY account_id")
            ]
            held_symbols = {
                str(row["symbol"])
                for row in connection.execute(
                    "SELECT DISTINCT symbol FROM positions WHERE quantity != 0"
                )
            }
            pending_symbols = {
                str(row["symbol"])
                for row in connection.execute(
                    "SELECT DISTINCT symbol FROM orders WHERE status = 'PENDING'"
                )
            }
        symbols = sorted(held_symbols | pending_symbols)
        marks: dict[str, float] = {}
        quote_rows: list[dict[str, object]] = []
        quote_source = "DAILY_FALLBACK"
        quote_error = None
        if symbols:
            try:
                daily = ParquetStore(DATA).read(
                    "daily",
                    columns=["ts_code", "trade_date", "close"],
                    filters=[("ts_code", "in", symbols)],
                )
                latest = (
                    daily.sort_values("trade_date")
                    .groupby("ts_code", as_index=False)
                    .tail(1)
                )
                marks = {
                    str(row["ts_code"]): float(row["close"])
                    for row in latest.to_dict("records")
                }
            except (FileNotFoundError, ValueError):
                pass
            try:
                load_dotenv(ROOT / ".env")
                quotes = create_realtime_provider(root=ROOT).intraday_quotes(
                    symbols
                )
                quote_rows = [
                    dict(row)
                    for row in quotes.to_dict("records")
                    if float(row.get("last", 0) or 0) > 0
                ]
                marks.update(
                    {
                        str(row["symbol"]): float(row["last"])
                        for row in quote_rows
                    }
                )
                if quote_rows:
                    quote_source = str(quote_rows[0].get("source") or "QMT_REALTIME")
            except Exception as error:  # noqa: BLE001 - retain official close on outage
                quote_error = type(error).__name__  # Never expose credential-bearing SDK errors.
        all_orders = broker.orders()
        today = now.strftime("%Y%m%d")
        accounts: dict[str, object] = {}
        for account_id in account_ids:
            portfolio = broker.account_snapshot(marks, account_id)
            enriched_positions = []
            for position in portfolio["positions"]:
                symbol = str(position["symbol"])
                last = float(marks.get(symbol, 0))
                quantity = int(position["quantity"])
                cost = float(position["avg_cost"])
                enriched_positions.append(
                    {
                        **position,
                        "last": last,
                        "market_value": quantity * last,
                        "unrealized_pnl": quantity * (last - cost),
                    }
                )
            portfolio["positions"] = enriched_positions
            orders = [
                row for row in reversed(all_orders) if row.get("account_id") == account_id
            ][:100]
            fills = broker.fills(account_id)
            accounts[account_id] = {
                "portfolio": portfolio,
                "positions": enriched_positions,
                "orders": orders,
                "fills": fills[-100:],
                "today_fills": [
                    row for row in fills if str(row.get("trade_date", "")) == today
                ],
                "pending_orders": [
                    row for row in orders if str(row.get("status")) == "PENDING"
                ],
            }
        payload: dict[str, object] = {
            "observed_at": now.isoformat(),
            "quote_source": quote_source,
            "quote_error": quote_error,
            "symbol_count": len(symbols),
            "quotes": quote_rows,
            "accounts": accounts,
        }
        _LIVE_ACCOUNT_CACHE.update(
            {"expires_at": clock.monotonic() + 4.0, "payload": payload}
        )
        return payload


def _append_live_nav_points(
    comparison: list[dict[str, object]],
    live: dict[str, object],
    store: ParquetStore,
) -> list[dict[str, object]]:
    """Overlay today's live account valuation on the close-based chart history.

    The persisted account daily tables remain the official close ledger.  This
    function only enriches the API response, scaling each live equity from its
    own latest official account observation so a branch keeps its existing NAV
    lineage.  The close pipeline naturally replaces the provisional point.
    """
    observed_at = str(live.get("observed_at") or "")
    try:
        trade_date = datetime.fromisoformat(observed_at).astimezone(SHANGHAI).strftime("%Y%m%d")
    except ValueError:
        return comparison
    accounts = live.get("accounts")
    if not isinstance(accounts, dict):
        return comparison
    specifications = {
        "baseline": (MULTI_SECTOR_ACCOUNT, "multi_sector_account_daily", "原基线"),
        "baseline_intraday": (INTRADAY_ACCOUNT, "baseline_intraday_account_daily", "基线 + 日内实时"),
        "baseline_pysystemtrade": (
            BASELINE_EXPOSURE_ACCOUNT,
            "multi_sector_pysystemtrade_account_daily",
            "基线 + pysystemtrade",
        ),
        "baseline_pysystemtrade_asymmetric": (
            ASYMMETRIC_EXPOSURE_ACCOUNT,
            ASYMMETRIC_EXPOSURE_HISTORY,
            "基线 + 非对称调仓",
        ),
        "baseline_pysystemtrade_trend_cap": (
            TREND_CAP_EXPOSURE_ACCOUNT,
            TREND_CAP_EXPOSURE_HISTORY,
            "基线 + 趋势上限",
        ),
        "baseline_pysystemtrade_intraday": (
            EXPOSURE_INTRADAY_ACCOUNT,
            "baseline_pysystemtrade_intraday_account_daily",
            "基线 + 仓位控制 + 日内实时",
        ),
    }
    result = list(comparison)
    for strategy_id, (account_id, table, label) in specifications.items():
        account = accounts.get(account_id)
        if not isinstance(account, dict):
            continue
        portfolio = account.get("portfolio")
        if not isinstance(portfolio, dict) or portfolio.get("equity") is None:
            continue
        # Do not manufacture a new trading-day point from stale closing prices
        # on weekends or during a QMT outage.  A real fill is sufficient evidence
        # that the account changed even if the quote request itself degraded.
        if live.get("quote_source") not in {"QMT_REALTIME", "ALPHAFEED_REALTIME"} and not account.get("today_fills"):
            continue
        series = [row for row in result if row.get("strategy_id") == strategy_id]
        if not series:
            continue
        try:
            official = store.read(table).copy()
        except FileNotFoundError:
            continue
        if official.empty or not {"trade_date", "equity"}.issubset(official.columns):
            continue
        official["trade_date"] = official["trade_date"].astype(str)
        official["equity"] = pd.to_numeric(official["equity"], errors="coerce")
        official = official.dropna(subset=["equity"]).sort_values("trade_date")
        prior = official.loc[official["trade_date"] < trade_date]
        if prior.empty:
            # If the close job has already persisted today, use that row as the
            # anchor and merely replace its chart value with the fresher mark.
            prior = official.loc[official["trade_date"] <= trade_date]
        if prior.empty:
            continue
        anchor = prior.iloc[-1]
        anchor_date = str(anchor["trade_date"])
        anchor_points = [row for row in series if str(row.get("trade_date")) == anchor_date]
        if not anchor_points:
            continue
        anchor_nav = float(anchor_points[-1]["normalized_nav"])
        anchor_equity = float(anchor["equity"])
        live_equity = float(portfolio["equity"])
        if anchor_equity <= 0:
            continue
        result.append(
            {
                "trade_date": trade_date,
                "strategy_id": strategy_id,
                "strategy": label,
                "normalized_nav": anchor_nav * live_equity / anchor_equity,
                "provisional": True,
                "valuation_mode": str(live.get("quote_source") or "UNKNOWN"),
                "observed_at": observed_at,
            }
        )
    return result


def _runtime_health_bus() -> dict[str, object]:
    now = datetime.now(SHANGHAI)
    checks: list[dict[str, object]] = []

    def add(code: str, status: str, message: str, **details: object) -> None:
        checks.append({"code": code, "status": status, "message": message, **details})

    heartbeat_path = STATE / "runtime" / "supervisor-heartbeat.json"
    try:
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8-sig"))
        heartbeat_at = datetime.fromisoformat(str(heartbeat["observed_at"]))
        heartbeat_age = max(0.0, (now - heartbeat_at.astimezone(SHANGHAI)).total_seconds())
        children_ok = bool(heartbeat.get("api_running") and heartbeat.get("web_running"))
        add(
            "SUPERVISOR_HEARTBEAT",
            "PASS" if heartbeat_age <= 35 and children_ok else "BLOCK",
            f"守护心跳 {heartbeat_age:.0f} 秒前，API/Web 子进程{'正常' if children_ok else '异常'}",
            age_seconds=heartbeat_age,
            **heartbeat,
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        add("SUPERVISOR_HEARTBEAT", "BLOCK", "未读取到有效的本地守护心跳")

    scheduler_alive = bool(task_service._thread and task_service._thread.is_alive())
    training_alive = bool(
        weekly_training_service._thread and weekly_training_service._thread.is_alive()
    )
    finrl_training_alive = bool(
        finrl_training_service._thread and finrl_training_service._thread.is_alive()
    )
    add(
        "DAILY_SCHEDULER",
        "PASS" if scheduler_alive else "BLOCK",
        "日度策略调度线程正常" if scheduler_alive else "日度策略调度线程已停止",
    )
    add(
        "TRAINING_SCHEDULER",
        "PASS" if training_alive and finrl_training_alive else "WARN",
        "Qlib/FinRL每周训练调度线程正常"
        if training_alive and finrl_training_alive else "至少一个每周训练调度线程已停止",
    )

    try:
        with sqlite3.connect(PAPER_DATABASE, timeout=3) as connection:
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        add(
            "PAPER_LEDGER",
            "PASS" if integrity.lower() == "ok" else "BLOCK",
            f"模拟交易账本检查：{integrity}",
        )
    except sqlite3.Error as error:
        add("PAPER_LEDGER", "BLOCK", f"模拟交易账本不可用：{error}")

    live = _live_account_snapshot()
    quotes = list(live.get("quotes", []))
    quote_timestamps = [
        int(row.get("timestamp", 0) or 0) for row in quotes if row.get("timestamp")
    ]
    quote_age = (
        max(
            0.0,
            (
                now
                - datetime.fromtimestamp(max(quote_timestamps) / 1000, SHANGHAI)
            ).total_seconds(),
        )
        if quote_timestamps
        else None
    )
    in_session = now.weekday() < 5 and (
        time(9, 30) <= now.time() <= time(11, 30)
        or time(13, 0) <= now.time() <= time(15, 0)
    )
    qmt_ok = live.get("quote_source") in {"QMT_REALTIME", "ALPHAFEED_REALTIME"} and (
        not in_session or (quote_age is not None and quote_age <= 120)
    )
    if in_session and live.get("quote_source") == "ALPHAFEED_REALTIME":
        qmt_ok = qmt_ok and bool(quotes) and all(row.get("execution_ready") for row in quotes)
    add(
        "QMT_MARKET_DATA",
        "PASS" if qmt_ok else "BLOCK" if in_session else "WARN",
        (
            f"{live.get('quote_source')}行情可用，最新行情延迟 {quote_age:.0f} 秒"
            if quote_age is not None and qmt_ok
            else "交易时段实时行情缺失或延迟超过120秒"
            if in_session
            else "非交易时段未取得实时行情，使用正式收盘数据"
        ),
        quote_source=live.get("quote_source"),
        quote_age_seconds=quote_age,
        symbol_count=live.get("symbol_count", 0),
    )

    reconciled = True
    mismatches: list[str] = []
    broker = PaperBroker(PAPER_DATABASE)
    for account_id in live.get("accounts", {}):
        result = broker.reconcile(str(account_id))
        # The intraday comparator was intentionally forked with the baseline's
        # existing cash and positions.  Its post-fork fills cannot reconstruct
        # that opening balance, so validate order/fill linkage for this account.
        branch_matched = (
            str(account_id) == INTRADAY_ACCOUNT
            and result.missing_fill_count == 0
            and result.orphan_fill_count == 0
        )
        if not result.matched and not branch_matched:
            reconciled = False
            mismatches.append(str(account_id))
    add(
        "ACCOUNT_RECONCILIATION",
        "PASS" if reconciled else "BLOCK",
        "全部模拟账户订单成交对账一致" if reconciled else f"账户对账异常：{','.join(mismatches)}",
        mismatched_accounts=mismatches,
    )

    free_gb = shutil.disk_usage(ROOT).free / (1024**3)
    disk_status = "BLOCK" if free_gb < 2 else "WARN" if free_gb < 10 else "PASS"
    add("DISK_CAPACITY", disk_status, f"项目磁盘剩余 {free_gb:.1f} GB", free_gb=free_gb)

    try:
        store = ParquetStore(DATA)
        latest_daily = str(
            store.read("daily", columns=["trade_date"])["trade_date"]
            .astype(str)
            .max()
        )
        calendar = store.read("trade_calendar", columns=["cal_date", "is_open"])
        calendar["cal_date"] = calendar["cal_date"].astype(str)
        today = now.strftime("%Y%m%d")
        expected_cutoff = today
        config = task_service.config()
        if (now.hour, now.minute) < (int(config["hour"]), int(config["minute"])):
            expected_cutoff = (now - timedelta(days=1)).strftime("%Y%m%d")
        eligible = calendar.loc[
            (calendar["cal_date"] <= expected_cutoff)
            & (calendar["is_open"].astype(int) == 1),
            "cal_date",
        ]
        expected_daily = str(eligible.max()) if not eligible.empty else latest_daily
        current = latest_daily >= expected_daily
        add(
            "OFFICIAL_DAILY_DATA",
            "PASS" if current else "WARN",
            f"正式日线最新数据日 {latest_daily}；应到 {expected_daily}",
            latest_trade_date=latest_daily,
            expected_trade_date=expected_daily,
        )
    except (FileNotFoundError, ValueError):
        add("OFFICIAL_DAILY_DATA", "WARN", "尚未确认正式日线数据日期")

    blocking = [row for row in checks if row["status"] == "BLOCK"]
    warnings = [row for row in checks if row["status"] == "WARN"]
    return {
        "observed_at": now.isoformat(),
        "system_status": "FROZEN" if blocking else "DEGRADED" if warnings else "HEALTHY",
        "trading_gate": "FROZEN" if blocking else "OPEN",
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "checks": checks,
    }


def _require_trading_gate() -> None:
    health = _runtime_health_bus()
    if health["trading_gate"] != "OPEN":
        blocked = [
            str(row["code"])
            for row in health["checks"]
            if row["status"] == "BLOCK"
        ]
        raise RuntimeError(f"SYSTEM_HEALTH_FROZEN: {','.join(blocked)}")


def _composite_universe_symbols(store: ParquetStore) -> list[str]:
    try:
        return active_strategy_universe(store)["symbol"].astype(str).tolist()
    except (FileNotFoundError, ValueError):
        pass
    sector_config = yaml.safe_load(
        (ROOT / "configs" / "sector_models.yaml").read_text(encoding="utf-8")
    )
    symbols = {
        str(symbol)
        for universe in sector_config["universes"].values()
        for symbol in universe["holdings"]
    }
    membership = store.read(
        "universe_membership",
        columns=["date", "ts_code"],
        filters=[("universe", "==", "bank_cn")],
    )
    membership["date"] = membership["date"].astype(str)
    latest_date = membership["date"].max()
    symbols.update(membership.loc[membership["date"] == latest_date, "ts_code"].astype(str))
    return sorted(symbols)


def _refresh_research_financial_pool(store: ParquetStore, trade_date: str) -> dict:
    # Long bounded backfills must not occupy the trading/API process. The worker
    # owns its dedicated crash-safe lock and checkpoints, not the broker DB.
    if trade_date != datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"):
        return {"status": "SKIPPED_HISTORICAL_REPLAY", "orders_enabled": False}
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / "financial-research-backfill.out.log").open("ab") as output, (logs / "financial-research-backfill.err.log").open("ab") as error:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts/sync_research_financials.py"),
             "--as-of", trade_date, "--max-requests", "1000"],
            cwd=ROOT, stdout=output, stderr=error, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return {"status": "DISPATCHED_NOT_COMPLETED", "pid": process.pid,
            "orders_enabled": False, "quality_report": "state/fundamental-research-v2/quality.json"}


def _factor_research_quality(store: ParquetStore, symbols: list[str]) -> dict:
    as_of = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    try:
        live = financial_quality(store.read("fina_indicator"), symbols, as_of)
    except FileNotFoundError:
        live = {"complete": 0, "requested": len(symbols), "gate": "BLOCK"}
    result = {"live": live}
    for label, relative in (("repair", "fundamental-research-v2/quality.json"),
                            ("comparison", "fundamental-research-v2/comparison.json"),
                            ("score_archive", "score-evidence/global_multifactor_v1/latest.json")):
        path = STATE / relative
        result[label] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    try:
        staged = research_financial_frame(store)
        result["repair"].update(financial_quality(staged, symbols, as_of, per_field=True))
    except FileNotFoundError:
        pass
    return result


def _selected_factor_symbols(store: ParquetStore) -> list[str]:
    recommendation = store.read("global_factor_recommendation")
    return sorted(recommendation.loc[recommendation["selected"].astype(bool), "symbol"].astype(str))


def _next_open_trade_date(store: ParquetStore, trade_date: str) -> str | None:
    if not trade_date:
        return None
    calendar = store.read("trade_calendar", columns=["cal_date", "is_open"])
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    future = calendar.loc[
        (calendar["cal_date"] > trade_date) & (calendar["is_open"].astype(int) == 1),
        "cal_date",
    ].sort_values()
    return str(future.iloc[0]) if not future.empty else None


class TaskService:
    def __init__(self, database: Path = SERVICE_DATABASE) -> None:
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_lock = threading.Lock()
        self._last_intraday_minute = ""
        self._last_open_execution_date = ""
        self._last_open_execution_minute = ""
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    report_path TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    enabled INTEGER NOT NULL,
                    hour INTEGER NOT NULL,
                    minute INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO task_config(id, enabled, hour, minute, updated_at)
                VALUES (1, 1, 18, 30, ?)
                """,
                (datetime.now(SHANGHAI).isoformat(),),
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    target_weight REAL NOT NULL,
                    research_status TEXT NOT NULL,
                    added_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_step_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    trade_date TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    error TEXT,
                    UNIQUE(run_id, step_name, attempt)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    code TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    trade_date TEXT,
                    run_id INTEGER,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    acknowledged_at TEXT
                )
                """
            )
            for candidate in (
                (
                    "600036.SH",
                    "招商银行",
                    "cmb_ma_120_250_v1",
                    1,
                    0.10,
                    "CANDIDATE",
                ),
                (
                    "600900.SH",
                    "长江电力",
                    "cypc_ma_120_250_v1",
                    1,
                    0.10,
                    "WATCH",
                ),
            ):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO candidates(
                        symbol, name, strategy_id, enabled, target_weight,
                        research_status, added_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*candidate, datetime.now(SHANGHAI).isoformat()),
                )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._recover_interrupted_runs()
        today = datetime.now(SHANGHAI).strftime("%Y%m%d")
        open_audit = STATE / "open-execution" / f"{today}.json"
        if open_audit.exists():
            try:
                if json.loads(open_audit.read_text(encoding="utf-8")).get("status") == "COMPLETED":
                    self._last_open_execution_date = today
            except (json.JSONDecodeError, OSError):
                pass
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            name="moneymore-scheduler",
            daemon=True,
        )
        self._thread.start()
        # A machine can be powered off for one or more complete trading days.
        # Recover those gaps as soon as the API starts instead of waiting for
        # the next 18:30 scheduler tick.  _recovery_dates deliberately excludes
        # today's still-open session before the configured EOD time.
        try:
            if self._recovery_dates(ParquetStore(DATA), today):
                self.trigger_recovery(today, "STARTUP_RECOVERY")
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            self._notify(
                "WARNING",
                "STARTUP_RECOVERY_DEFERRED",
                "启动补偿扫描暂缓",
                f"{type(error).__name__}: {error}",
                trade_date=today,
                dedupe_key=f"{today}:startup-recovery-deferred",
            )

    def _recover_interrupted_runs(self) -> None:
        now = datetime.now(SHANGHAI).isoformat()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE task_step_runs
                SET status = 'FAILED', finished_at = ?,
                    error = 'SERVICE_RESTART_INTERRUPTED'
                WHERE status = 'RUNNING'
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE task_runs
                SET status = 'FAILED', finished_at = ?,
                    error = 'SERVICE_RESTART_INTERRUPTED'
                WHERE status IN ('QUEUED', 'RUNNING')
                """,
                (now,),
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def trigger(self, trade_date: str, source: str = "MANUAL") -> int:
        run_id = self._create_run("daily_pipeline", trade_date, source)
        threading.Thread(
            target=self._execute,
            args=(run_id, trade_date),
            name=f"daily-run-{run_id}",
            daemon=True,
        ).start()
        return run_id

    def trigger_recovery(
        self,
        as_of_date: str,
        source: str = "MANUAL_RECOVERY",
        force_current_session: bool = False,
    ) -> int:
        run_id = self._create_run("recovery_pipeline", as_of_date, source)
        threading.Thread(
            target=self._execute_recovery,
            args=(run_id, as_of_date, force_current_session),
            name=f"recovery-run-{run_id}",
            daemon=True,
        ).start()
        return run_id

    def _create_run(self, task_name: str, trade_date: str, source: str) -> int:
        with sqlite3.connect(self.database) as connection:
            cursor = connection.execute(
                """
                INSERT INTO task_runs(
                    task_name, trade_date, source, status, started_at
                ) VALUES (?, ?, ?, 'QUEUED', ?)
                """,
                (task_name, trade_date, source, datetime.now(SHANGHAI).isoformat()),
            )
            return int(cursor.lastrowid)

    def runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM task_runs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in rows]

    def steps(self, limit: int = 100, run_id: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM task_step_runs"
        params: list[object] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query, params)]

    def notifications(self, limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,)
            )
            result = [dict(row) for row in rows]
        for row in result:
            row["acknowledged"] = bool(row["acknowledged"])
        return result

    def acknowledge_notification(self, notification_id: int) -> dict[str, Any]:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE notifications
                SET acknowledged = 1, acknowledged_at = ?
                WHERE id = ?
                """,
                (datetime.now(SHANGHAI).isoformat(), notification_id),
            )
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM notifications WHERE id = ?", (notification_id,)
            ).fetchone()
        if row is None:
            raise KeyError(notification_id)
        result = dict(row)
        result["acknowledged"] = bool(result["acknowledged"])
        return result

    def preview(self, trade_date: str) -> dict[str, object]:
        step_names = [
            "bank_pipeline",
            "strategy_universe_refresh",
            "composite_daily_basic",
            "sector_research",
            "selected_fundamental_backfill",
            "multi_sector_execution",
            "qlib_challenger_execution",
            "qlib_candidate_observation",
            "research_financial_pool_refresh",
            "qlib_forward_and_style_review",
        ]
        with sqlite3.connect(self.database) as connection:
            completed = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT step_name FROM task_step_runs
                    WHERE trade_date = ? AND status = 'COMPLETED'
                    """,
                    (trade_date,),
                )
            }
            prior_runs = int(
                connection.execute(
                    "SELECT COUNT(*) FROM task_runs WHERE trade_date = ?",
                    (trade_date,),
                ).fetchone()[0]
            )
        return {
            "trade_date": trade_date,
            "prior_run_count": prior_runs,
            "steps": [
                {
                    "step_name": name,
                    "action": "SKIP_COMPLETED" if name in completed else "RUN",
                }
                for name in step_names
            ],
            "mutates_state": False,
        }

    def _notify(
        self,
        severity: str,
        code: str,
        title: str,
        message: str,
        *,
        trade_date: str | None = None,
        run_id: int | None = None,
        dedupe_key: str | None = None,
    ) -> None:
        key = dedupe_key or f"{trade_date}:{run_id}:{code}"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO notifications(
                    created_at, severity, code, title, message,
                    trade_date, run_id, dedupe_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(SHANGHAI).isoformat(),
                    severity,
                    code,
                    title,
                    message,
                    trade_date,
                    run_id,
                    key,
                ),
            )

    def _run_step(
        self,
        run_id: int,
        trade_date: str,
        step_name: str,
        operation: Any,
        max_attempts: int = 2,
        skip_if_completed: bool = True,
    ) -> tuple[str, Any]:
        with sqlite3.connect(self.database) as connection:
            completed = connection.execute(
                """
                SELECT 1
                FROM task_step_runs
                WHERE trade_date = ? AND step_name = ? AND status = 'COMPLETED'
                LIMIT 1
                """,
                (trade_date, step_name),
            ).fetchone()
            if completed and skip_if_completed:
                now = datetime.now(SHANGHAI).isoformat()
                connection.execute(
                    """
                    INSERT INTO task_step_runs(
                        run_id, trade_date, step_name, attempt, status,
                        started_at, finished_at
                    ) VALUES (?, ?, ?, 0, 'SKIPPED_COMPLETED', ?, ?)
                    """,
                    (run_id, trade_date, step_name, now, now),
                )
                return "SKIPPED_COMPLETED", None

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            started_at = datetime.now(SHANGHAI).isoformat()
            with sqlite3.connect(self.database) as connection:
                connection.execute(
                    """
                    INSERT INTO task_step_runs(
                        run_id, trade_date, step_name, attempt, status, started_at
                    ) VALUES (?, ?, ?, ?, 'RUNNING', ?)
                    """,
                    (run_id, trade_date, step_name, attempt, started_at),
                )
            try:
                result = operation()
                with sqlite3.connect(self.database) as connection:
                    connection.execute(
                        """
                        UPDATE task_step_runs
                        SET status = 'COMPLETED', finished_at = ?
                        WHERE run_id = ? AND step_name = ? AND attempt = ?
                        """,
                        (
                            datetime.now(SHANGHAI).isoformat(),
                            run_id,
                            step_name,
                            attempt,
                        ),
                    )
                return "COMPLETED", result
            except Exception as error:  # noqa: BLE001 - retry boundary
                last_error = error
                detail = f"{type(error).__name__}: {error}"
                with sqlite3.connect(self.database) as connection:
                    connection.execute(
                        """
                        UPDATE task_step_runs
                        SET status = 'FAILED', finished_at = ?, error = ?
                        WHERE run_id = ? AND step_name = ? AND attempt = ?
                        """,
                        (
                            datetime.now(SHANGHAI).isoformat(),
                            detail,
                            run_id,
                            step_name,
                            attempt,
                        ),
                    )
        assert last_error is not None
        self._notify(
            "ERROR",
            "PIPELINE_STEP_FAILED",
            f"流水线步骤失败：{step_name}",
            f"自动重试 {max_attempts} 次后仍失败：{type(last_error).__name__}: {last_error}",
            trade_date=trade_date,
            run_id=run_id,
            dedupe_key=f"{trade_date}:{run_id}:step:{step_name}",
        )
        raise last_error

    def config(self) -> dict[str, object]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT enabled, hour, minute, updated_at FROM task_config WHERE id = 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("task configuration is missing")
            result = dict(row)
            result["enabled"] = bool(result["enabled"])
            result["schedule"] = f"每日 {int(result['hour']):02d}:{int(result['minute']):02d}"
            result["timezone"] = "Asia/Shanghai"
            return result

    def update_config(self, enabled: bool, hour: int, minute: int) -> dict[str, object]:
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError("invalid scheduler time")
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE task_config
                SET enabled = ?, hour = ?, minute = ?, updated_at = ?
                WHERE id = 1
                """,
                (
                    int(enabled),
                    hour,
                    minute,
                    datetime.now(SHANGHAI).isoformat(),
                ),
            )
        return self.config()

    def candidates(self, enabled_only: bool = False) -> list[dict[str, object]]:
        query = "SELECT * FROM candidates"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY added_at, symbol"
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            rows = [dict(row) for row in connection.execute(query)]
        for row in rows:
            row["enabled"] = bool(row["enabled"])
        return rows

    def add_candidate(self, symbol: str, name: str) -> dict[str, object]:
        normalized = symbol.strip().upper()
        if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", normalized):
            raise ValueError("symbol must look like 600900.SH")
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 30:
            raise ValueError("name must contain 1-30 characters")
        store = ParquetStore(DATA)
        if store.read(
            "daily",
            columns=["ts_code"],
            filters=[("ts_code", "==", normalized)],
        ).empty:
            raise ValueError(f"symbol is not available in local data: {normalized}")
        strategy_id = "stock_ma_120_250_v1"
        with sqlite3.connect(self.database) as connection:
            exists = connection.execute(
                "SELECT 1 FROM candidates WHERE symbol = ?", (normalized,)
            ).fetchone()
            if not exists:
                gross = float(
                    connection.execute(
                        """
                        SELECT COALESCE(SUM(target_weight), 0)
                        FROM candidates WHERE enabled = 1
                        """
                    ).fetchone()[0]
                )
                max_gross = BacktestConfig.from_yaml(
                    ROOT / "configs" / "default.yaml"
                ).max_gross_exposure
                if gross + 0.10 > max_gross + 1e-9:
                    raise ValueError("candidate pool would exceed gross exposure limit")
            connection.execute(
                """
                INSERT INTO candidates(
                    symbol, name, strategy_id, enabled, target_weight,
                    research_status, added_at
                ) VALUES (?, ?, ?, 1, 0.10, 'UNREVIEWED', ?)
                ON CONFLICT(symbol) DO UPDATE SET name = excluded.name
                """,
                (
                    normalized,
                    clean_name,
                    strategy_id,
                    datetime.now(SHANGHAI).isoformat(),
                ),
            )
        return next(row for row in self.candidates() if row["symbol"] == normalized)

    def _scheduler_loop(self) -> None:
        while not self._stop.wait(20):
            now = datetime.now(SHANGHAI)
            minute_key = now.strftime("%Y%m%d%H%M")
            trade_date = now.strftime("%Y%m%d")
            open_execution_window = (
                now.weekday() < 5
                and time(9, 31) <= now.time() <= time(18, 25)
                and self._is_open_trade_date(trade_date)
            )
            if (
                open_execution_window
                and trade_date != self._last_open_execution_date
                and minute_key != self._last_open_execution_minute
            ):
                self._last_open_execution_minute = minute_key
                try:
                    load_dotenv(ROOT / ".env")
                    _require_trading_gate()
                    broker = PaperBroker(PAPER_DATABASE)
                    prepare_intraday_branch_orders(broker, trade_date)
                    prepare_exposure_intraday_branch_orders(broker, trade_date)
                    result = execute_accounts_at_qmt_open(
                        provider=create_realtime_provider(root=ROOT),
                        broker=broker,
                        config=BacktestConfig.from_yaml(ROOT / "configs" / "default.yaml"),
                        trade_date=trade_date,
                        account_ids=[
                            BANK_ACCOUNT,
                            MULTI_SECTOR_ACCOUNT,
                            BASELINE_EXPOSURE_ACCOUNT,
                            ASYMMETRIC_EXPOSURE_ACCOUNT,
                            TREND_CAP_EXPOSURE_ACCOUNT,
                            QLIB_CHALLENGER_ACCOUNT,
                            *[
                                candidate_account_id(str(row["candidate_tag"]))
                                for row in candidate_catalog(ROOT)
                            ],
                        ],
                        audit_dir=STATE / "open-execution",
                    )
                    if result["status"] == "COMPLETED":
                        self._last_open_execution_date = trade_date
                except Exception as open_error:  # noqa: BLE001
                    self._notify(
                        "ERROR",
                        "OPEN_EXECUTION_FAILED",
                        "开盘撮合失败",
                        f"{type(open_error).__name__}: {open_error}",
                        trade_date=trade_date,
                        dedupe_key=f"{minute_key}:open-execution-failed",
                    )
            in_market_session = time(9, 35) <= now.time() <= time(11, 30) or time(
                13, 0
            ) <= now.time() <= time(15, 0)
            in_execution_window = (
                now.weekday() < 5
                and in_market_session
                and now.strftime("%Y%m%d") >= INTRADAY_BASELINE_ACTIVATION_DATE
            )
            if in_execution_window and minute_key != self._last_intraday_minute:
                self._last_intraday_minute = minute_key
                try:
                    load_dotenv(ROOT / ".env")
                    _require_trading_gate()
                    intraday_results = [run_baseline_intraday_once(root=ROOT, now=now)]
                    intraday_results.append(run_baseline_intraday_once(
                        root=ROOT,
                        now=now,
                        source_account=BASELINE_EXPOSURE_ACCOUNT,
                        account_id=EXPOSURE_INTRADAY_ACCOUNT,
                        snapshot_table="baseline_pysystemtrade_intraday_snapshots",
                        tick_table="baseline_pysystemtrade_intraday_account_ticks",
                        daily_table="baseline_pysystemtrade_intraday_account_daily",
                        audit_subdirectory="baseline-pysystemtrade-intraday-execution",
                    ))
                    for result in intraday_results:
                        if result.get("status") in {"QMT_QUOTES_EMPTY", "REALTIME_QUOTES_EMPTY", "PARTIAL_COMPENSATION"}:
                            self._notify(
                                "ERROR" if result["status"] in {"QMT_QUOTES_EMPTY", "REALTIME_QUOTES_EMPTY"} else "WARN",
                                "INTRADAY_TARGET_GAP",
                                "日内目标仓位未补齐",
                                f"{result['account_id']}: {result['status']}; "
                                f"blocked={result.get('blocked_target_gaps', [])}",
                                trade_date=trade_date,
                                dedupe_key=f"{trade_date}:{result['account_id']}:{result['status']}",
                            )
                except Exception as intraday_error:  # noqa: BLE001
                    self._notify(
                        "ERROR",
                        "INTRADAY_EXECUTION_FAILED",
                        "基线日内执行失败",
                        f"{type(intraday_error).__name__}: {intraday_error}",
                        trade_date=now.strftime("%Y%m%d"),
                        dedupe_key=f"{minute_key}:intraday-execution-failed",
                    )
            config = self.config()
            if not config["enabled"]:
                continue
            if (now.hour, now.minute) < (int(config["hour"]), int(config["minute"])):
                continue
            if self._scheduler_should_trigger(trade_date, now):
                self.trigger_recovery(trade_date, "SCHEDULED")

    @staticmethod
    def _is_open_trade_date(trade_date: str) -> bool:
        try:
            calendar = ParquetStore(DATA).read(
                "trade_calendar",
                columns=["cal_date", "is_open"],
                filters=[("cal_date", "==", trade_date)],
            )
        except (FileNotFoundError, ValueError):
            return False
        return bool(len(calendar) == 1 and int(calendar.iloc[0]["is_open"]) == 1)

    def _scheduler_should_trigger(
        self,
        trade_date: str,
        now: datetime,
        retry_interval: timedelta = timedelta(minutes=10),
    ) -> bool:
        """Avoid duplicate recovery storms while retaining delayed-data retries."""
        with sqlite3.connect(self.database) as connection:
            active = connection.execute(
                """
                SELECT 1 FROM task_runs
                WHERE status IN ('QUEUED', 'RUNNING')
                  AND task_name IN ('recovery_pipeline', 'daily_pipeline')
                LIMIT 1
                """
            ).fetchone()
            if active:
                return False
            completed = connection.execute(
                """
                SELECT 1 FROM task_runs
                WHERE task_name = 'daily_pipeline' AND trade_date = ?
                  AND status = 'COMPLETED'
                LIMIT 1
                """,
                (trade_date,),
            ).fetchone()
            if completed:
                return False
            latest = connection.execute(
                """
                SELECT status, started_at FROM task_runs
                WHERE task_name = 'recovery_pipeline' AND trade_date = ?
                  AND source = 'SCHEDULED'
                ORDER BY id DESC LIMIT 1
                """,
                (trade_date,),
            ).fetchone()
        if latest is None:
            return True
        status, started_at = str(latest[0]), datetime.fromisoformat(str(latest[1]))
        if status in {"QUEUED", "RUNNING", "COMPLETED"}:
            return False
        return now - started_at >= retry_interval

    def _recovery_dates(
        self, store: ParquetStore, as_of_date: str, force_current_session: bool = False
    ) -> list[str]:
        daily = store.read("daily", columns=["trade_date"])
        if daily.empty:
            raise RuntimeError("cannot recover without locally stored daily history")
        latest_data_date = str(daily["trade_date"].astype(str).max())
        calendar = store.read("trade_calendar", columns=["cal_date", "is_open"])
        calendar["cal_date"] = calendar["cal_date"].astype(str)
        dates = (
            calendar.loc[
                (calendar["cal_date"] > latest_data_date)
                & (calendar["cal_date"] <= as_of_date)
                & (calendar["is_open"].astype(int) == 1),
                "cal_date",
            ]
            .sort_values()
            .tolist()
        )
        # Data can be present while the service was offline before the
        # downstream strategy/execution stages ran.  The recovery watermark is
        # therefore the latest *completed workflow*, not only the latest bar.
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                """
                SELECT MAX(trade_date) FROM task_runs
                WHERE task_name = 'daily_pipeline' AND status = 'COMPLETED'
                  AND trade_date <= ?
                """,
                (latest_data_date,),
            ).fetchone()
        latest_completed_date = str(row[0]) if row and row[0] else ""
        workflow_gap = calendar.loc[
            (calendar["cal_date"] > latest_completed_date)
            & (calendar["cal_date"] <= latest_data_date)
            & (calendar["is_open"].astype(int) == 1),
            "cal_date",
        ].tolist()
        dates = sorted(set(dates) | set(workflow_gap))
        # A completed daily pipeline is not sufficient evidence for the two
        # intraday comparison branches: those normally run only while the
        # market-session loop is alive.  Missing audit artifacts therefore
        # remain a recovery gap even when official daily bars and parent
        # strategy runs are already complete.
        state_root = self.database.parent
        recovery_sessions = [
            str(item) for item in calendar.loc[
                (calendar["cal_date"] >= INTRADAY_BASELINE_ACTIVATION_DATE)
                & (calendar["cal_date"] <= latest_data_date)
                & (calendar["is_open"].astype(int) == 1),
                "cal_date",
            ].tolist()
        ]
        intraday_gap = [
            item for item in recovery_sessions
            if not (state_root / "intraday-execution" / f"{item}.json").exists()
            or (
                item >= EXPOSURE_INTRADAY_ACTIVATION_DATE
                and not (
                state_root
                / "baseline-pysystemtrade-intraday-execution"
                / f"{item}.json"
                ).exists()
            )
        ]
        dates = sorted(set(dates) | set(intraday_gap))
        # Intraday vendors do not guarantee a complete daily bar.  A manual
        # recovery before the configured end-of-day schedule must leave the
        # current session pending instead of creating an empty-data failure.
        now = datetime.now(SHANGHAI)
        config = self.config()
        if (
            as_of_date == now.strftime("%Y%m%d")
            and (now.hour, now.minute) < (int(config["hour"]), int(config["minute"]))
            and not force_current_session
        ):
            dates = [item for item in dates if item != as_of_date]
        return dates

    def _execute_recovery(
        self, run_id: int, as_of_date: str, force_current_session: bool = False
    ) -> None:
        try:
            self._set_running(run_id)
            store = ParquetStore(DATA)
            dates = self._recovery_dates(store, as_of_date, force_current_session)
            if force_current_session and as_of_date in dates:
                load_dotenv(ROOT / ".env")
                probe = create_market_data_provider().daily_bars(as_of_date)
                if probe.empty:
                    dates = [item for item in dates if item != as_of_date]
            if not dates:
                self._defer_empty_daily_failures(as_of_date)
                self._finish(
                    run_id,
                    "WAITING_MARKET_DATA",
                )
                self._notify(
                    "INFO",
                    "MARKET_DATA_PENDING",
                    "等待行情发布",
                    "当日尚无完整日线；已保留上一交易日的 T+1 待撮合委托，行情发布后将自动恢复执行。",
                    trade_date=as_of_date,
                    run_id=run_id,
                    dedupe_key=f"{as_of_date}:market-data-pending",
                )
                return
            completed: list[str] = []
            for trade_date in dates:
                child_run_id = self._create_run("daily_pipeline", trade_date, "RECOVERY")
                self._execute(child_run_id, trade_date, historical_recovery=True)
                child = next(row for row in self.runs(100) if row["id"] == child_run_id)
                if child["status"] != "COMPLETED":
                    raise RuntimeError(
                        f"recovery stopped at {trade_date}: {child['status']} {child.get('error') or ''}"
                    )
                completed.append(trade_date)
            self._finish(run_id, "COMPLETED", report_path=",".join(completed))
        except Exception as error:  # noqa: BLE001 - recovery must remain observable
            detail = f"{type(error).__name__}: {error}"
            self._finish(run_id, "FAILED", error=detail)
            self._notify(
                "ERROR",
                "RECOVERY_FAILED",
                "恢复流水线失败",
                detail,
                trade_date=as_of_date,
                run_id=run_id,
            )

    def _defer_empty_daily_failures(self, trade_date: str) -> None:
        """Resolve obsolete same-day empty-bar failures into a retryable wait state."""
        now = datetime.now(SHANGHAI).isoformat()
        marker = f"daily[{trade_date}] is empty"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE task_runs
                SET status = 'WAITING_MARKET_DATA', error = NULL, finished_at = ?
                WHERE task_name = 'daily_pipeline' AND trade_date = ?
                  AND status = 'FAILED' AND error LIKE ?
                """,
                (now, trade_date, f"%{marker}%"),
            )
            connection.execute(
                """
                UPDATE notifications
                SET acknowledged = 1, acknowledged_at = ?
                WHERE trade_date = ? AND acknowledged = 0
                  AND code IN ('PIPELINE_FAILED', 'PIPELINE_STEP_FAILED')
                  AND message LIKE ?
                """,
                (now, trade_date, f"%{marker}%"),
            )

    def _execute(
        self, run_id: int, trade_date: str, historical_recovery: bool = False
    ) -> None:
        if not self._run_lock.acquire(blocking=False):
            self._finish(run_id, "SKIPPED_BUSY", error="another pipeline is running")
            return
        try:
            self._set_running(run_id)
            load_dotenv(ROOT / ".env")
            provider = create_market_data_provider()
            store = ParquetStore(DATA)
            broker = PaperBroker(PAPER_DATABASE)
            config = BacktestConfig.from_yaml(ROOT / "configs" / "default.yaml")
            _, result = self._run_step(
                run_id,
                trade_date,
                "bank_pipeline",
                lambda: run_bank_daily_pipeline(
                    provider=provider,
                    store=store,
                    broker=broker,
                    config=config,
                    trade_date=trade_date,
                    signal_dir=STATE / "bank-signals",
                    report_dir=STATE / "bank-shadow",
                ),
            )
            if result is not None and result.status == "SKIPPED_MARKET_CLOSED":
                self._finish(run_id, "COMPLETED", report_path=result.report_path)
                return
            if result is not None and result.status != "COMPLETED":
                raise RuntimeError(f"bank pipeline status: {result.status}")

            if historical_recovery:
                # The live scheduler normally executes every prior-session
                # plan from QMT shortly after the open.  When the service or
                # MiniQMT was offline, daily research alone is not enough:
                # overlay accounts (notably the exposure-controlled baseline)
                # otherwise retain old PENDING orders indefinitely.  Replay
                # those orders at the immutable official daily open before
                # generating this session's new targets.  PaperBroker keeps
                # the operation idempotent, so already executed accounts are
                # harmlessly skipped.
                self._run_step(
                    run_id,
                    trade_date,
                    "historical_open_execution_recovery",
                    lambda: execute_accounts_at_open(
                        store=store,
                        broker=broker,
                        config=config,
                        trade_date=trade_date,
                        account_ids=[
                            BANK_ACCOUNT,
                            MULTI_SECTOR_ACCOUNT,
                            BASELINE_EXPOSURE_ACCOUNT,
                            ASYMMETRIC_EXPOSURE_ACCOUNT,
                            TREND_CAP_EXPOSURE_ACCOUNT,
                            QLIB_CHALLENGER_ACCOUNT,
                            *[
                                candidate_account_id(str(row["candidate_tag"]))
                                for row in candidate_catalog(ROOT)
                            ],
                        ],
                        audit_dir=STATE / "open-execution-recovery",
                    ),
                )

            self._run_step(
                run_id,
                trade_date,
                "strategy_universe_refresh",
                lambda: refresh_market_cap_universe(provider, store, trade_date),
            )
            _, daily_basic_result = self._run_step(
                run_id,
                trade_date,
                "composite_daily_basic",
                lambda: sync_daily_basic_universe(
                    provider,
                    store,
                    _composite_universe_symbols(store),
                    trade_date,
                ),
            )
            if daily_basic_result and int(daily_basic_result.get("missing", 0)):
                missing_symbols = ",".join(
                    str(item) for item in daily_basic_result.get("missing_symbols", [])
                )
                self._notify(
                    "WARNING",
                    "DAILY_BASIC_SUSPENDED_SYMBOLS",
                    "停牌股票缺少当日估值快照",
                    f"已沿用最近估值，其余股票继续运行：{missing_symbols}",
                    trade_date=trade_date,
                    run_id=run_id,
                    dedupe_key=f"{trade_date}:daily-basic-suspended",
                )
            self._run_step(
                run_id,
                trade_date,
                "point_in_time_refresh",
                lambda: materialize_point_in_time_store(
                    ROOT,
                    captured_date=trade_date,
                    feature_start=pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
                ),
            )
            self._run_step(
                run_id,
                trade_date,
                "sector_research",
                lambda: subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "research_sector_portfolio.py"),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=900,
                ),
            )
            strategy_failures: list[str] = []
            try:
                self._run_step(
                    run_id,
                    trade_date,
                    "selected_fundamental_backfill",
                    lambda: sync_missing_target_fundamentals(
                        provider,
                        store,
                        _selected_factor_symbols(store),
                        trade_date,
                    ),
                    skip_if_completed=False,
                )
            except Exception as fundamental_error:  # noqa: BLE001
                self._notify(
                    "WARNING",
                    "TARGET_FUNDAMENTAL_BACKFILL_DEGRADED",
                    "新目标股基本面补齐降级",
                    f"{fundamental_error}；继续由数据健康检查判断 QMT/本地基本面覆盖是否足够。",
                    trade_date=trade_date,
                    run_id=run_id,
                )

            def execute_multi_sector() -> Any:
                value = run_multi_sector_daily(
                    store=store,
                    broker=broker,
                    config=config,
                    trade_date=trade_date,
                    signal_dir=STATE / "multi-sector-signals",
                    report_dir=STATE / "multi-sector-shadow",
                )
                if value.status != "COMPLETED":
                    code = {
                        "BLOCKED_DATA_QUALITY": "DATA_QUALITY_BLOCKED",
                        "SUSPENDED_RISK": "RISK_SUSPENDED",
                        "FAILED_RECONCILIATION": "RECONCILIATION_FAILED",
                    }.get(value.status, "PIPELINE_STATUS_BLOCKED")
                    self._notify(
                        "ERROR",
                        code,
                        "综合组合流水线未完成",
                        f"状态：{value.status}",
                        trade_date=trade_date,
                        run_id=run_id,
                    )
                    raise RuntimeError(f"multi-sector pipeline status: {value.status}")
                return value

            multi_result = None
            try:
                _, multi_result = self._run_step(
                    run_id,
                    trade_date,
                    "multi_sector_execution",
                    execute_multi_sector,
                )
            except Exception as multi_error:  # noqa: BLE001
                strategy_failures.append(
                    f"multi_sector_execution: {type(multi_error).__name__}: {multi_error}"
                )
            deferred = (
                []
                if multi_result is None
                else [
                    row
                    for row in multi_result.executions
                    if str(row.get("status", "")).upper() in {"DEFERRED", "REJECTED", "BLOCKED"}
                ]
            )
            if deferred:
                self._notify(
                    "WARN",
                    "ORDER_DEFERRED",
                    "存在延迟或受阻订单",
                    f"本次流水线共有 {len(deferred)} 条执行未即时完成，请在撮合明细复核。",
                    trade_date=trade_date,
                    run_id=run_id,
                )
            if multi_result is not None:
                try:
                    def execute_baseline_exposure_league() -> Any:
                        daily_path = DATA / "processed" / "daily.parquet"
                        snapshot = build_market_risk_snapshot(
                            str(DATA), daily_path.stat().st_mtime_ns
                        )
                        contestants = {
                            str(row["method_id"]): row
                            for row in snapshot["exposure_league"]["contestants"]
                        }
                        results = {}
                        for method_id, variant in EXPOSURE_VARIANTS.items():
                            contestant = contestants.get(method_id)
                            if contestant is None or contestant.get("target_exposure") is None:
                                continue
                            slug = str(variant["slug"])
                            results[method_id] = run_baseline_exposure_daily(
                                store=store,
                                broker=broker,
                                config=config,
                                trade_date=trade_date,
                                baseline_report=asdict(multi_result),
                                target_exposure=float(contestant["target_exposure"]),
                                signal_dir=STATE / f"{slug}-signals",
                                report_dir=STATE / f"{slug}-shadow",
                                account_id=str(variant["account_id"]),
                                strategy_id=str(variant["strategy_id"]),
                                history_table=str(variant["history_table"]),
                            )
                        return results
                    self._run_step(
                        run_id,
                        trade_date,
                        "baseline_exposure_league_execution",
                        execute_baseline_exposure_league,
                    )
                except Exception as overlay_error:  # noqa: BLE001
                    strategy_failures.append(
                        f"baseline_exposure_league_execution: {type(overlay_error).__name__}: {overlay_error}"
                    )
            independent_steps = [
                (
                    "tqqq_sqqq_paper_execution",
                    lambda: LeveragedEtfPaper(
                        LEVERAGED_ETF_DATABASE, LEVERAGED_ETF_CACHE
                    ).run(),
                ),
                (
                    "qlib_challenger_execution",
                    lambda: run_qlib_challenger_daily(
                        root=ROOT,
                        store=store,
                        broker=broker,
                        config=config,
                        trade_date=trade_date,
                        signal_dir=STATE / "qlib-challenger-signals",
                        report_dir=STATE / "qlib-challenger-shadow",
                    ),
                ),
                (
                    "qlib_candidate_observation",
                    lambda: run_candidate_queue_daily(
                        root=ROOT,
                        store=store,
                        broker=broker,
                        config=config,
                        trade_date=trade_date,
                    ),
                ),
                (
                    "qlib_forward_and_style_review",
                    lambda: run_challenger_review(ROOT, store, trade_date),
                ),
                (
                    "research_financial_pool_refresh",
                    lambda: _refresh_research_financial_pool(store, trade_date),
                ),
                (
                    "qlib_drift_monitoring",
                    lambda: evaluate_qlib_drift(ROOT, store, broker, trade_date),
                ),
                (
                    "qlib_long_term_review",
                    lambda: evaluate_long_term_review(ROOT, store, broker, trade_date),
                ),
            ]
            for step_name, operation in independent_steps:
                try:
                    self._run_step(run_id, trade_date, step_name, operation)
                except Exception as strategy_error:  # noqa: BLE001
                    detail = f"{type(strategy_error).__name__}: {strategy_error}"
                    strategy_failures.append(f"{step_name}: {detail}")
                    self._notify(
                        "WARN",
                        "INDEPENDENT_STRATEGY_STEP_FAILED",
                        f"独立策略步骤失败：{step_name}",
                        f"{detail}; 其他策略继续运行",
                        trade_date=trade_date,
                        run_id=run_id,
                        dedupe_key=f"{trade_date}:{step_name}:failed",
                    )
            if historical_recovery:
                # Minute-history replay repairs comparison accounts; it is not
                # a prerequisite for the daily factor baseline. QMT can be
                # offline during weekends and holidays. A replay outage must
                # not pin the recovery watermark and block newer daily NAVs.
                try:
                    self._run_step(
                        run_id,
                        trade_date,
                        "intraday_historical_replay",
                        lambda: self._recover_intraday_branches(
                            store, broker, trade_date
                        ),
                    )
                except Exception as replay_error:  # noqa: BLE001
                    detail = f"{type(replay_error).__name__}: {replay_error}"
                    self._notify(
                        "WARN",
                        "INTRADAY_REPLAY_DEFERRED",
                        "日内历史补演暂缓",
                        f"{detail}; 日线基线继续恢复，分钟数据可用后自动重试日内分支。",
                        trade_date=trade_date,
                        run_id=run_id,
                        dedupe_key=f"{trade_date}:intraday-replay-deferred",
                    )
            if strategy_failures:
                raise RuntimeError(
                    "independent strategy failures: " + " | ".join(strategy_failures)
                )
            prior_failure = any(
                row["id"] != run_id
                and row["trade_date"] == trade_date
                and row["status"] == "FAILED"
                for row in self.runs(100)
            )
            if prior_failure:
                self._notify(
                    "INFO",
                    "PIPELINE_RECOVERED",
                    "流水线恢复成功",
                    "失败任务已从完成步骤后续跑通，无需重复执行已完成步骤。",
                    trade_date=trade_date,
                    run_id=run_id,
                )
            report_path = None if result is None else result.report_path
            self._finish(run_id, "COMPLETED", report_path=report_path)
        except Exception as error:  # noqa: BLE001 - task failures must stay observable
            detail = f"{type(error).__name__}: {error}"
            self._finish(run_id, "FAILED", error=detail)
            self._notify(
                "ERROR",
                "PIPELINE_FAILED",
                "每日流水线失败",
                detail,
                trade_date=trade_date,
                run_id=run_id,
            )
        finally:
            self._run_lock.release()

    def _recover_intraday_branches(
        self, store: ParquetStore, broker: PaperBroker, trade_date: str
    ) -> dict[str, object]:
        calendar = store.read("trade_calendar", columns=["cal_date", "is_open"])
        calendar["cal_date"] = calendar["cal_date"].astype(str)
        previous = calendar.loc[
            (calendar["cal_date"] < trade_date)
            & (calendar["is_open"].astype(int) == 1),
            "cal_date",
        ].sort_values()
        future = calendar.loc[
            (calendar["cal_date"] > trade_date)
            & (calendar["is_open"].astype(int) == 1),
            "cal_date",
        ].sort_values()
        if previous.empty:
            raise RuntimeError(f"no prior open session for intraday replay {trade_date}")
        plan_date = str(previous.iloc[-1])
        state_root = self.database.parent
        replayed: list[str] = []
        for account_name, audit_subdir in (
            ("baseline", "intraday-execution"),
            ("controlled", "baseline-pysystemtrade-intraday-execution"),
        ):
            if (
                account_name == "controlled"
                and trade_date < EXPOSURE_INTRADAY_ACTIVATION_DATE
            ):
                continue
            audit = state_root / audit_subdir / f"{trade_date}.json"
            if audit.exists():
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "replay_intraday_gap.py"),
                    "--account",
                    account_name,
                    "--trade-date",
                    trade_date,
                    "--plan-date",
                    plan_date,
                    "--commit",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
            replayed.append(account_name)
        next_trade_date = str(future.iloc[0]) if not future.empty else trade_date
        prepared = {
            "baseline": prepare_intraday_branch_orders(broker, next_trade_date),
            "controlled": prepare_exposure_intraday_branch_orders(
                broker, next_trade_date
            ),
        }
        return {
            "trade_date": trade_date,
            "plan_date": plan_date,
            "replayed": replayed,
            "next_trade_date": next_trade_date,
            "prepared_orders": prepared,
        }

    def _set_running(self, run_id: int) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE task_runs SET status = 'RUNNING' WHERE id = ?", (run_id,))

    def _finish(
        self,
        run_id: int,
        status: str,
        report_path: str | None = None,
        error: str | None = None,
    ) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE task_runs
                SET status = ?, finished_at = ?, report_path = ?, error = ?
                WHERE id = ?
                """,
                (
                    status,
                    datetime.now(SHANGHAI).isoformat(),
                    report_path,
                    error,
                    run_id,
                ),
            )


task_service = TaskService()
weekly_training_service = WeeklyTrainingService(ROOT)
finrl_training_service = WeeklyTrainingService(
    ROOT,
    ROOT / "state" / "finrl-training.sqlite3",
    runner_script="run_weekly_finrl_training.py",
    schedule_label="每周六 09:00（FinRL总仓位）",
    log_subdirectory="training/finrl",
)
deepdow_training_service = WeeklyTrainingService(
    ROOT,
    ROOT / "state" / "deepdow-training.sqlite3",
    runner_script="run_weekly_deepdow_training.py",
    schedule_label="每周六 09:00（DeepDow总仓位）",
    log_subdirectory="training/deepdow",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task_service.start()
    weekly_training_service.start()
    finrl_training_service.start()
    leveraged_etf_scheduler.start()
    if task_service.config().get("enabled"):
        try:
            _refresh_research_financial_pool(ParquetStore(DATA), datetime.now(SHANGHAI).strftime("%Y%m%d"))
        except OSError:
            # Research repair is optional; failure must not take the trading API down.
            pass
    yield
    leveraged_etf_scheduler.stop()
    finrl_training_service.stop()
    weekly_training_service.stop()
    task_service.stop()


app = FastAPI(title="MoneyMore API", version="0.2.0", lifespan=lifespan)
app.include_router(desktop_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "mode": "PAPER_ONLY",
        "realtime_data_source": os.getenv("MONEYMORE_REALTIME_DATA_SOURCE", "qmt"),
        "research_data_source": os.getenv("MONEYMORE_MARKET_DATA_SOURCE", "qmt"),
        "scheduler": bool(task_service._thread and task_service._thread.is_alive()),
        "weekly_training_scheduler": bool(
            weekly_training_service._thread and weekly_training_service._thread.is_alive()
        ),
        "finrl_training_scheduler": bool(
            finrl_training_service._thread and finrl_training_service._thread.is_alive()
        ),
        "deepdow_training_scheduler": False,
        "server_time": datetime.now(SHANGHAI).isoformat(),
    }


@app.get("/api/leveraged-etf-paper")
def leveraged_etf_paper() -> dict[str, object]:
    payload = leveraged_etf_paper_service.snapshot()
    payload["scheduler"] = leveraged_etf_scheduler.status()
    return payload


@app.post("/api/leveraged-etf-paper/run")
def run_leveraged_etf_paper() -> dict[str, object]:
    try:
        payload = leveraged_etf_paper_service.run()
        payload["scheduler"] = leveraged_etf_scheduler.status()
        return payload
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"US ETF paper run failed: {error}") from error


@app.get("/api/live-accounts")
def live_accounts() -> dict[str, object]:
    return _live_account_snapshot()


@app.get("/api/system-health")
def system_health() -> dict[str, object]:
    return _runtime_health_bus()


@app.get("/api/market-risk")
def market_risk() -> dict[str, object]:
    daily_path = DATA / "processed" / "daily.parquet"
    baseline_path = DATA / "curated" / "multi_sector_account_daily.parquet"
    if not baseline_path.exists():
        baseline_path = DATA / "processed" / "multi_sector_account_daily.parquet"
    # build_market_risk_snapshot is cached.  This endpoint enriches the snapshot
    # with live branch histories, so never mutate the cached object itself.
    payload = copy.deepcopy(
        build_market_risk_snapshot(
            str(DATA),
            daily_path.stat().st_mtime_ns,
            baseline_path.stat().st_mtime_ns if baseline_path.exists() else 0,
        )
    )
    broker = PaperBroker(PAPER_DATABASE)
    broker.initialize_account(1_000_000, BASELINE_EXPOSURE_ACCOUNT)
    broker.initialize_account(1_000_000, ASYMMETRIC_EXPOSURE_ACCOUNT)
    broker.initialize_account(1_000_000, TREND_CAP_EXPOSURE_ACCOUNT)
    try:
        history = _records(ParquetStore(DATA).read(BASELINE_EXPOSURE_HISTORY).sort_values("trade_date"))
    except FileNotFoundError:
        history = []
    comparison = list(payload["exposure_league"].get("strategy_comparison", []))
    store = ParquetStore(DATA)
    for method_id, variant in EXPOSURE_VARIANTS.items():
        if method_id == "pysystemtrade_vol_target":
            continue
        strategy_id = f"baseline_{method_id}"
        theoretical = [
            row for row in comparison if row.get("strategy_id") == strategy_id
        ]
        try:
            actual = store.read(str(variant["history_table"])).sort_values("trade_date")
        except FileNotFoundError:
            actual = pd.DataFrame()
        if actual.empty or not theoretical:
            continue
        first = str(actual.iloc[0]["trade_date"])
        lineage = [row for row in theoretical if str(row["trade_date"]) < first]
        base_nav = float(lineage[-1]["normalized_nav"]) if lineage else 1.0
        first_equity = float(actual.iloc[0]["equity"])
        comparison.extend(
            [
                {
                    "trade_date": str(row["trade_date"]),
                    "strategy_id": strategy_id,
                    "strategy": str(variant["label"]),
                    "normalized_nav": base_nav * float(row["equity"]) / first_equity,
                    "history_source": "INDEPENDENT_PAPER_ACCOUNT",
                }
                for row in actual.to_dict("records")
            ]
        )
    try:
        baseline_history = store.read("multi_sector_account_daily").sort_values("trade_date")
        intraday_history = store.read("baseline_intraday_account_daily").sort_values("trade_date")
        branch = build_branch_history(
            baseline_history, intraday_history,
            branch_account_id=INTRADAY_ACCOUNT,
            activation_date=INTRADAY_BASELINE_ACTIVATION_DATE,
        )
        if not branch.empty:
            branch["normalized_nav"] = branch["equity"] / float(branch["equity"].iloc[0])
            comparison.extend(
                {"trade_date": str(row["trade_date"]), "strategy_id": "baseline_intraday", "strategy": "基线 + 日内实时", "normalized_nav": float(row["normalized_nav"])}
                for row in branch.to_dict("records")
            )
    except FileNotFoundError:
        pass
    overlay_counterfactual = [
        dict(row) for row in comparison if row.get("strategy_id") == "baseline_pysystemtrade"
    ]
    try:
        controlled_intraday = store.read("baseline_pysystemtrade_intraday_account_daily").sort_values("trade_date")
    except FileNotFoundError:
        controlled_intraday = pd.DataFrame()
    if controlled_intraday.empty:
        comparison.extend(
            {**row, "strategy_id": "baseline_pysystemtrade_intraday", "strategy": "基线 + 仓位控制 + 日内实时"}
            for row in overlay_counterfactual
        )
    else:
        first = str(controlled_intraday.iloc[0]["trade_date"])
        lineage = [row for row in overlay_counterfactual if str(row["trade_date"]) < first]
        base_nav = float(lineage[-1]["normalized_nav"]) if lineage else 1.0
        first_equity = float(controlled_intraday.iloc[0]["equity"])
        comparison.extend(
            [{**row, "strategy_id": "baseline_pysystemtrade_intraday", "strategy": "基线 + 仓位控制 + 日内实时"} for row in lineage]
            + [
                {"trade_date": str(row["trade_date"]), "strategy_id": "baseline_pysystemtrade_intraday", "strategy": "基线 + 仓位控制 + 日内实时", "normalized_nav": base_nav * float(row["equity"]) / first_equity}
                for row in controlled_intraday.to_dict("records")
            ]
        )
    comparison = _append_live_nav_points(comparison, _live_account_snapshot(), store)
    # Intraday history may contain several account observations for one session.
    # Charts are daily, therefore the final observation for each strategy/date is
    # the only meaningful point.  This also bounds the API response size.
    daily_comparison = {
        (str(row.get("strategy_id")), str(row.get("trade_date"))): row
        for row in comparison
    }
    payload["exposure_league"]["strategy_comparison"] = sorted(
        daily_comparison.values(),
        key=lambda row: (str(row.get("trade_date")), str(row.get("strategy_id"))),
    )
    payload["paper_accounts"] = {
        "baseline": {
            "account_id": MULTI_SECTOR_ACCOUNT,
            "orders": [row for row in reversed(broker.orders()) if row.get("account_id") == MULTI_SECTOR_ACCOUNT][:100],
            "fills": broker.fills(MULTI_SECTOR_ACCOUNT)[-100:],
        },
        "baseline_pysystemtrade": {
            "account_id": BASELINE_EXPOSURE_ACCOUNT,
            "orders": [row for row in reversed(broker.orders()) if row.get("account_id") == BASELINE_EXPOSURE_ACCOUNT][:100],
            "fills": broker.fills(BASELINE_EXPOSURE_ACCOUNT)[-100:],
            "history": history,
        },
        "baseline_pysystemtrade_asymmetric": {
            "account_id": ASYMMETRIC_EXPOSURE_ACCOUNT,
            "orders": [row for row in reversed(broker.orders()) if row.get("account_id") == ASYMMETRIC_EXPOSURE_ACCOUNT][:100],
            "fills": broker.fills(ASYMMETRIC_EXPOSURE_ACCOUNT)[-100:],
        },
        "baseline_pysystemtrade_trend_cap": {
            "account_id": TREND_CAP_EXPOSURE_ACCOUNT,
            "orders": [row for row in reversed(broker.orders()) if row.get("account_id") == TREND_CAP_EXPOSURE_ACCOUNT][:100],
            "fills": broker.fills(TREND_CAP_EXPOSURE_ACCOUNT)[-100:],
        },
        "baseline_intraday": {
            "account_id": INTRADAY_ACCOUNT,
            "orders": [row for row in reversed(broker.orders()) if row.get("account_id") == INTRADAY_ACCOUNT][:100],
            "fills": broker.fills(INTRADAY_ACCOUNT)[-100:],
        },
        "baseline_pysystemtrade_intraday": {
            "account_id": EXPOSURE_INTRADAY_ACCOUNT,
            "orders": [row for row in reversed(broker.orders()) if row.get("account_id") == EXPOSURE_INTRADAY_ACCOUNT][:100],
            "fills": broker.fills(EXPOSURE_INTRADAY_ACCOUNT)[-100:],
        },
    }
    return payload


@app.get("/api/security-history")
def security_history(symbol: str) -> dict[str, object]:
    normalized = symbol.strip().upper()
    if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", normalized):
        raise HTTPException(status_code=400, detail="invalid security symbol")
    store = ParquetStore(DATA)
    try:
        bars = store.read(
            "daily",
            columns=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ],
            filters=[("ts_code", "==", normalized)],
        ).sort_values("trade_date")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if bars.empty:
        raise HTTPException(status_code=404, detail="security history not found")
    bars = _append_live_daily_bar(bars, normalized)
    with sqlite3.connect(PAPER_DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        fills = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, account_id, symbol, side, quantity, price, fee, trade_date
                FROM fills WHERE symbol = ?
                ORDER BY trade_date, id
                """,
                (normalized,),
            )
        ]
    # The intraday account is a branch of the factor baseline, not a fresh
    # account.  Expose the parent's pre-activation fills as inherited display
    # rows so a security chart shows the branch's complete trading lineage.
    inherited_fills = [
        {
            **row,
            "account_id": INTRADAY_ACCOUNT,
            "source_account_id": MULTI_SECTOR_ACCOUNT,
            "inherited": True,
        }
        for row in fills
        if str(row.get("account_id")) == MULTI_SECTOR_ACCOUNT
        and str(row.get("trade_date", "")) < INTRADAY_BASELINE_ACTIVATION_DATE
    ]
    fills.extend(inherited_fills)
    fills.sort(key=lambda row: (str(row.get("trade_date", "")), int(row.get("id", 0))))
    names = _instrument_names(store)
    return {
        "symbol": normalized,
        "name": names.get(normalized, normalized),
        "bars": _records(bars),
        "fills": fills,
        "coverage": {
            "start_date": str(bars.iloc[0]["trade_date"]),
            "end_date": str(bars.iloc[-1]["trade_date"]),
            "trading_days": len(bars),
            "fill_count": len(fills),
        },
    }


def _append_live_daily_bar(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Append today's QMT tick as a provisional daily candle when needed."""
    today = datetime.now(SHANGHAI).strftime("%Y%m%d")
    if today in set(bars["trade_date"].astype(str)):
        result = bars.copy()
        result["is_realtime"] = False
        return result
    try:
        cached_quotes = _live_account_snapshot().get("quotes", [])
        quotes = pd.DataFrame(
            [row for row in cached_quotes if str(row.get("symbol")) == symbol]
        )
        if quotes.empty:
            load_dotenv(ROOT / ".env")
            quotes = create_realtime_provider(root=ROOT).intraday_quotes(
                [symbol]
            )
    except Exception:  # noqa: BLE001 - stale charts are preferable to a failed detail view
        return bars
    if quotes.empty:
        return bars
    quote = quotes.iloc[0]
    timestamp = int(quote.get("timestamp", 0) or 0)
    quote_date = (
        datetime.fromtimestamp(timestamp / 1000, SHANGHAI).strftime("%Y%m%d")
        if timestamp > 0
        else ""
    )
    prices = [
        float(quote.get(column, 0) or 0)
        for column in ("open", "high", "low", "last")
    ]
    if quote_date != today or any(price <= 0 for price in prices):
        return bars
    result = bars.copy()
    result["is_realtime"] = False
    live = pd.DataFrame(
        [
            {
                "ts_code": symbol,
                "trade_date": today,
                "open": prices[0],
                "high": prices[1],
                "low": prices[2],
                "close": prices[3],
                "vol": float(quote.get("volume", 0) or 0),
                "amount": float(quote.get("amount", 0) or 0) / 1000.0,
                "is_realtime": True,
            }
        ]
    )
    return pd.concat([result, live], ignore_index=True)


@app.get("/api/intraday-execution")
def intraday_execution_status() -> dict[str, object]:
    today = datetime.now(SHANGHAI).strftime("%Y%m%d")
    target = STATE / "intraday-execution" / f"{today}.json"
    latest: dict[str, object] = {}
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        observations = payload.get("observations", [])
        latest = observations[-1] if observations else {}
    broker = PaperBroker(PAPER_DATABASE)
    pending = [
        row
        for row in broker.orders()
        if row.get("account_id") == INTRADAY_ACCOUNT and row.get("status") == "PENDING"
    ]
    try:
        ticks = _records(
            ParquetStore(DATA)
            .read(
                "baseline_intraday_account_ticks",
                filters=[("trade_date", "==", today)],
            )
            .sort_values("observed_at")
            .tail(300)
        )
    except FileNotFoundError:
        ticks = []
    return {
        "policy_id": POLICY_ID,
        "activation_date": INTRADAY_BASELINE_ACTIVATION_DATE,
        "account_id": INTRADAY_ACCOUNT,
        "pending_orders": pending,
        "fills": broker.fills(INTRADAY_ACCOUNT, today),
        "account_ticks": ticks,
        "latest": latest,
    }


def _open_execution_status(trade_date: str) -> dict[str, object]:
    target = STATE / "open-execution" / f"{trade_date}.json"
    audit = (
        json.loads(target.read_text(encoding="utf-8"))
        if target.exists()
        else {
            "trade_date": trade_date,
            "status": "WAITING_FOR_OPEN",
            "price_source": f"{os.getenv('MONEYMORE_REALTIME_DATA_SOURCE', 'qmt').upper()}_SESSION_OPEN",
            "accounts": [],
        }
    )
    account_ids = [
        MULTI_SECTOR_ACCOUNT,
        BASELINE_EXPOSURE_ACCOUNT,
        ASYMMETRIC_EXPOSURE_ACCOUNT,
        TREND_CAP_EXPOSURE_ACCOUNT,
        EXPOSURE_INTRADAY_ACCOUNT,
        QLIB_CHALLENGER_ACCOUNT,
        *[candidate_account_id(str(row["candidate_tag"])) for row in candidate_catalog(ROOT)],
    ]
    broker = PaperBroker(PAPER_DATABASE)
    pending = [
        row
        for row in broker.orders()
        if str(row.get("account_id")) in account_ids
        and row.get("status") == "PENDING"
        and str(row.get("signal_date", "")) < trade_date
    ]
    fills = [row for account_id in account_ids for row in broker.fills(account_id, trade_date)]
    return {
        **audit,
        "scheduled_time": "09:31",
        "account_ids": account_ids,
        "pending_orders": pending,
        "fills": sorted(fills, key=lambda row: int(row.get("id", 0)), reverse=True),
    }


@app.get("/api/open-execution")
def open_execution_status() -> dict[str, object]:
    return _open_execution_status(datetime.now(SHANGHAI).strftime("%Y%m%d"))


@app.get("/api/weekly-training")
def weekly_training() -> dict[str, object]:
    status = weekly_training_service.status()
    status["finrl"] = finrl_training_service.status()
    deepdow_status = deepdow_training_service.status()
    deepdow_status.update({"enabled": False, "retired": True})
    status["deepdow"] = deepdow_status
    return status


@app.get("/api/factors")
def factors() -> dict[str, object]:
    registry = build_default_registry()
    catalog = registry.catalog()
    candidate_catalog = build_qmt_candidate_registry().catalog()
    active_names = {str(item["name"]) for item in catalog}
    qmt_candidates = [
        {**item, "lifecycle": "RESEARCH_CANDIDATE", "active": False}
        for item in candidate_catalog
        if str(item["name"]) not in active_names
    ]
    return {
        "count": len(catalog),
        "categories": sorted({str(item["category"]) for item in catalog}),
        "items": catalog,
        "qmt_candidate_count": len(qmt_candidates),
        "qmt_candidates": qmt_candidates,
    }


@app.get("/api/factor-research")
def factor_research(universe: str = "bank_cn") -> dict[str, object]:
    if not re.fullmatch(r"[a-z0-9_]{1,40}", universe):
        raise HTTPException(status_code=400, detail="invalid universe")
    store = ParquetStore(DATA)
    try:
        membership = store.read("universe_membership", filters=[("universe", "==", universe)])
        ic = store.read("factor_ic_report")
        period_ic = store.read("factor_ic_period_report")
        quantiles = store.read("factor_quantile_report")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    latest_date = membership["date"].max()
    return {
        "universe": universe,
        "membership_rows": len(membership),
        "latest_date": str(latest_date)[:10],
        "latest_members": int((membership["date"] == latest_date).sum()),
        "ic": _records(ic),
        "period_ic": _records(period_ic),
        "quantiles": _records(quantiles),
    }


@app.get("/api/bank-model")
def bank_model() -> dict[str, object]:
    store = ParquetStore(DATA)
    try:
        scores = store.read("bank_model_scores")
        targets = store.read("bank_model_targets")
        report = store.read("bank_model_report")
        robustness = store.read("bank_model_robustness")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    latest_date = scores["date"].max()
    latest_scores = scores.loc[scores["date"] == latest_date].sort_values("score", ascending=False)
    latest_targets = targets.loc[
        (targets["date"] == targets["date"].max()) & targets["selected"]
    ].sort_values("rank")
    return {
        "model_id": "bank_multifactor_v1",
        "status": "RESEARCH_ONLY",
        "warning": (
            "The 2022+ period has been inspected during model development and is "
            "validation evidence, not a pristine untouched test set."
        ),
        "latest_date": str(latest_date)[:10],
        "report": _records(report),
        "robustness": _records(robustness),
        "promotion": evaluate_bank_model_promotion(
            report, robustness, has_pristine_forward_period=False
        ),
        "latest_scores": _records(
            latest_scores[
                [
                    "symbol",
                    "score",
                    "value_score",
                    "defensive_score",
                    "momentum_score",
                    "quality_score",
                ]
            ].head(20)
        ),
        "latest_holdings": _records(latest_targets[["symbol", "rank", "target"]]),
    }


@app.get("/api/bank-dashboard")
def bank_dashboard() -> dict[str, object]:
    model = bank_model()
    store = ParquetStore(DATA)
    equity = store.read("bank_model_equity").copy()
    equity["date"] = equity["date"].astype(str).str[:10]
    sampled = equity.iloc[::20]
    shadow = _latest_bank_shadow()
    return {
        **model,
        "mode": "PAPER_ONLY",
        "shadow": shadow,
        "timing": bank_timing(),
        "scheduler": task_service.config(),
        "recent_runs": task_service.runs(10),
        "equity_curve": _records(sampled[["date", "equity", "drawdown"]]),
        "symbol_names": _instrument_names(store),
    }


@app.get("/api/bank-timing")
def bank_timing() -> dict[str, object]:
    store = ParquetStore(DATA)
    try:
        report = store.read("bank_timing_report")
        degrees = store.read("bank_timing_degrees")
        equity = store.read("bank_timing_equity")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    latest = (
        degrees.sort_values("date")
        .groupby("strategy", as_index=False)
        .tail(1)
        .sort_values("strategy")
    )
    active = "vol_target_12"
    current = latest.loc[latest["strategy"] == active].iloc[0]
    active_equity = equity.loc[equity["strategy"] == active].copy()
    active_equity["date"] = active_equity["date"].astype(str).str[:10]
    return {
        "interface": "qlib_risk_degree",
        "active_strategy": active,
        "status": "PROSPECTIVE_CANDIDATE",
        "risk_degree": float(current["risk_degree"]),
        "base_gross_exposure": 0.8,
        "bank_budget_fraction": float(current["risk_degree"]) / 0.8,
        "latest_date": str(current["date"])[:10],
        "report": _records(report),
        "current": _records(latest),
        "equity_curve": _records(active_equity.iloc[::20][["date", "equity", "drawdown"]]),
        "decision": (
            "FULL_BANK_BUDGET" if float(current["risk_degree"]) >= 0.8 else "REDUCED_BANK_BUDGET"
        ),
        "evidence_note": (
            "The timing candidates were selected using already inspected history. "
            "Only observations after 2026-07-27 are pristine forward evidence."
        ),
    }


@app.get("/api/sector-portfolio")
def sector_portfolio() -> dict[str, object]:
    store = ParquetStore(DATA)
    try:
        sector_diagnostics = store.read("sector_portfolio_recommendation")
        global_recommendation = store.read("global_factor_recommendation")
        report = store.read("sector_model_report")
        scores = store.read("sector_model_scores")
        targets = store.read("sector_model_targets")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    config = yaml.safe_load((ROOT / "configs" / "sector_models.yaml").read_text(encoding="utf-8"))
    try:
        strategy_universe = active_strategy_universe(store)
        strategy_universe_info = universe_summary(strategy_universe)
        strategy_constituents = strategy_universe.merge(
            global_recommendation[
                [
                    "symbol",
                    "score",
                    "global_rank",
                    "selected",
                    "target_weight",
                ]
            ],
            on="symbol",
            how="left",
        ).rename(columns={"rank": "market_cap_rank", "score": "factor_score"})
        strategy_constituents["selected"] = (
            strategy_constituents["selected"].fillna(False).astype(bool)
        )
        strategy_constituents["target_weight"] = strategy_constituents["target_weight"].fillna(0.0)
        industry_catalog = _records(
            strategy_constituents.groupby("industry", as_index=False)
            .agg(
                stock_count=("symbol", "count"),
                selected_count=("selected", "sum"),
                market_value=("total_mv", "sum"),
            )
            .sort_values("market_value", ascending=False)
        )
        board_catalog = _records(
            strategy_universe.groupby("board", as_index=False)
            .agg(stock_count=("symbol", "count"), market_value=("total_mv", "sum"))
            .sort_values("market_value", ascending=False)
        )
    except (FileNotFoundError, ValueError):
        strategy_universe_info = {}
        strategy_constituents = pd.DataFrame()
        industry_catalog = []
        board_catalog = []
    latest_date = scores["date"].max()
    latest_scores = scores.loc[scores["date"] == latest_date].copy()
    latest_targets = targets.loc[targets["date"] == targets["date"].max()].copy()
    ranked = latest_scores.merge(
        latest_targets[["sector", "symbol", "rank", "selected", "target"]],
        on=["sector", "symbol"],
        how="left",
    ).sort_values(["sector", "rank", "score"], ascending=[True, True, False])
    diagnostic_risk = dict(
        zip(
            sector_diagnostics["sector"].astype(str),
            sector_diagnostics["risk_degree"].astype(float),
            strict=True,
        )
    )
    selected_global = global_recommendation.loc[
        global_recommendation["selected"].astype(bool)
    ].copy()
    allocation = (
        selected_global.groupby("sector", as_index=False)
        .agg(
            target_weight=("target_weight", "sum"),
            selected=("symbol", lambda values: ",".join(map(str, values))),
        )
        .sort_values("target_weight", ascending=False)
    )
    allocation["budget_weight"] = allocation["target_weight"]
    allocation["cash_weight"] = 0.0
    allocation["risk_degree"] = allocation["sector"].map(diagnostic_risk)
    selected_symbols = selected_global["symbol"].astype(str).tolist()
    correlation = trailing_return_correlation(
        adjusted_close_panel(store, selected_symbols),
        global_recommendation["date"].max(),
        int(config["global_selection"]["correlation_lookback"]),
    )
    correlation_values = correlation.values[np.triu_indices(len(correlation), 1)]
    average_correlation = float(np.nanmean(correlation_values))
    effective_bets = len(selected_symbols) / (1 + (len(selected_symbols) - 1) * average_correlation)

    universes = []
    for sector, definition in config["universes"].items():
        universes.append(
            {
                "sector": sector,
                "name": definition["name"],
                "fund_code": definition["etf_code"],
                "style": definition["style"],
                "factor_weights": definition["factors"],
                "constituents": definition["holdings"],
                "ranking": _records(ranked.loc[ranked["sector"] == sector]),
            }
        )
    return {
        "status": "PAPER_RESEARCH_ONLY",
        "latest_date": str(latest_date)[:10],
        "disclosure_date": str(config["disclosure_date"]),
        "evidence_status": "FORWARD_MARKET_CAP_SNAPSHOT",
        "warning": (
            "统一候选池按当前总市值选取沪深主板、创业板和科创板前1000只。"
            "行业采用Tushare当前分类，仅用于选股后的展示和风险归因；"
            "候选池迁移前的历史收益保持冻结，迁移后证据从生效日继续累计。"
        ),
        "allocation_method": "global_cross_section_topk",
        "strategy_universe": strategy_universe_info,
        "industry_catalog": industry_catalog,
        "board_catalog": board_catalog,
        "strategy_constituents": _records(strategy_constituents),
        "portfolio_policy": config["global_selection"],
        "factor_data_quality": _factor_research_quality(store, strategy_constituents.symbol.astype(str).tolist() if not strategy_constituents.empty else []),
        "risk_metrics": {
            "average_pair_correlation": average_correlation,
            "effective_independent_bets": effective_bets,
            "selected_count": len(selected_symbols),
        },
        "allocation": _records(allocation),
        "report": _records(report),
        "universes": universes,
        "symbol_names": _instrument_names(store),
    }


@app.get("/api/bank-execution")
def bank_execution() -> dict[str, object]:
    broker = PaperBroker(PAPER_DATABASE)
    broker.initialize_account(1_000_000, BANK_ACCOUNT)
    with sqlite3.connect(PAPER_DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        fills = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM fills WHERE account_id = ? ORDER BY id DESC LIMIT 100",
                (BANK_ACCOUNT,),
            )
        ]
        positions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM positions WHERE account_id = ? AND quantity != 0
                ORDER BY symbol
                """,
                (BANK_ACCOUNT,),
            )
        ]
    orders = [row for row in reversed(broker.orders()) if row.get("account_id") == BANK_ACCOUNT][
        :100
    ]
    latest = _latest_bank_shadow()
    return {
        "orders": orders,
        "fills": fills,
        "positions": positions,
        "portfolio": latest.get("portfolio"),
        "reconciliation": asdict(broker.reconcile(BANK_ACCOUNT)),
    }


@app.get("/api/multi-sector-execution")
def multi_sector_execution() -> dict[str, object]:
    broker = PaperBroker(PAPER_DATABASE)
    broker.initialize_account(1_000_000, MULTI_SECTOR_ACCOUNT)
    store = ParquetStore(DATA)
    with sqlite3.connect(PAPER_DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        fills = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM fills WHERE account_id = ? ORDER BY id DESC LIMIT 100",
                (MULTI_SECTOR_ACCOUNT,),
            )
        ]
        positions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM positions WHERE account_id = ? AND quantity != 0
                ORDER BY symbol
                """,
                (MULTI_SECTOR_ACCOUNT,),
            )
        ]
    orders = [
        row for row in reversed(broker.orders()) if row.get("account_id") == MULTI_SECTOR_ACCOUNT
    ][:100]
    latest = _latest_multi_sector_shadow()
    position_symbols = [str(row["symbol"]) for row in positions]
    live_marks: dict[str, float] = {}
    if position_symbols:
        daily = store.read(
            "daily",
            columns=["ts_code", "trade_date", "close"],
            filters=[("ts_code", "in", position_symbols)],
        )
        latest_daily = daily.sort_values("trade_date").groupby("ts_code", as_index=False).tail(1)
        live_marks = {
            str(row["ts_code"]): float(row["close"]) for row in latest_daily.to_dict("records")
        }
    live_portfolio = broker.account_snapshot(live_marks, MULTI_SECTOR_ACCOUNT)
    catch_up_dir = STATE / "catch-up"
    catch_up_reports = sorted(catch_up_dir.glob("*.json")) if catch_up_dir.exists() else []
    latest_catch_up = (
        json.loads(catch_up_reports[-1].read_text(encoding="utf-8")) if catch_up_reports else None
    )
    analytics: dict[str, list[dict[str, object]]] = {}
    for api_key, table in {
        "history": "multi_sector_account_daily",
        "deviations": "multi_sector_deviation_daily",
        "risk_alerts": "multi_sector_risk_alerts",
    }.items():
        try:
            frame = store.read(table)
            if api_key != "history" and not frame.empty:
                frame = frame.loc[frame["trade_date"] == frame["trade_date"].max()]
            analytics[api_key] = _records(frame)
        except FileNotFoundError:
            analytics[api_key] = []
    return {
        "account_id": MULTI_SECTOR_ACCOUNT,
        "status": (
            "COMPLETED_CATCH_UP"
            if latest_catch_up and latest_catch_up.get("trade_date") == latest.get("trade_date")
            else latest.get("status", "AWAITING_FIRST_RUN")
        ),
        "trade_date": latest.get("trade_date"),
        "target_weights": latest.get("target_weights", {}),
        "theoretical_target_weights": latest.get(
            "theoretical_target_weights", latest.get("target_weights", {})
        ),
        "target_adjustments": latest.get("target_adjustments", []),
        "symbol_sectors": latest.get("symbol_sectors", {}),
        "orders": orders,
        "fills": fills,
        "positions": positions,
        "portfolio": live_portfolio,
        "latest_catch_up": latest_catch_up,
        "attribution": latest.get("attribution", []),
        "return_attribution": latest.get("return_attribution", []),
        "risk_attribution": latest.get("risk_attribution", []),
        "execution_attribution": latest.get("execution_attribution", []),
        "attribution_reconciliation": latest.get("attribution_reconciliation", {}),
        "corporate_actions_today": latest.get(
            "corporate_actions", {"registered": [], "settled": []}
        ),
        "corporate_action_ledger": broker.corporate_actions(MULTI_SECTOR_ACCOUNT),
        "data_health": latest.get("data_health", {}),
        "risk_state": broker.risk_state(MULTI_SECTOR_ACCOUNT),
        "risk_transitions": broker.risk_transitions(MULTI_SECTOR_ACCOUNT)[:100],
        "model_version": latest.get("model_version", {}),
        "metrics": latest.get("metrics", {}),
        **analytics,
        "reconciliation": asdict(broker.reconcile(MULTI_SECTOR_ACCOUNT)),
        "symbol_names": _instrument_names(store),
    }


@app.post("/api/risk-state/recover")
def recover_risk_state() -> dict[str, object]:
    broker = PaperBroker(PAPER_DATABASE)
    broker.initialize_account(1_000_000, MULTI_SECTOR_ACCOUNT)
    return broker.approve_risk_recovery(MULTI_SECTOR_ACCOUNT)


@app.get("/api/data-quality")
def data_quality() -> dict[str, object]:
    store = ParquetStore(DATA)
    try:
        history = store.read("data_health_daily").sort_values("trade_date")
        checks = store.read("data_health_checks")
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    latest_date = str(history.iloc[-1]["trade_date"])
    latest_checks = checks.loc[checks["trade_date"].astype(str) == latest_date]
    return {
        "latest": _records(history.tail(1))[0],
        "checks": _records(latest_checks.sort_values(["status", "category", "code"])),
        "history": _records(history.tail(90)),
        "symbol_names": _instrument_names(store),
    }


@app.get("/api/model-registry")
def model_registry() -> dict[str, list[dict[str, Any]]]:
    registry = ModelRegistry(
        STATE / "model_registry.sqlite3",
        STATE / "model-versions",
    )
    return registry.snapshot()


@app.get("/api/qlib-challenger")
def qlib_challenger() -> dict[str, object]:
    broker = PaperBroker(PAPER_DATABASE)
    store = ParquetStore(DATA)
    broker.initialize_account(1_000_000, QLIB_CHALLENGER_ACCOUNT)
    research_path = STATE / "qlib-challenger" / "latest-research.json"
    forward_path = STATE / "qlib-challenger" / "forward-evaluation.json"
    historical_path = STATE / "historical-pk" / "latest.json"
    reports_dir = STATE / "qlib-challenger-shadow"
    reports = sorted(reports_dir.glob("*.json")) if reports_dir.exists() else []
    latest = (
        max(
            (json.loads(path.read_text(encoding="utf-8")) for path in reports),
            key=lambda report: str(report.get("created_at", "")),
        )
        if reports
        else {
            "status": "AWAITING_TRAINING",
            "selected": [],
            "scores": [],
            "portfolio": {
                "cash": 1_000_000,
                "market_value": 0,
                "equity": 1_000_000,
                "positions": [],
            },
        }
    )
    research = (
        json.loads(research_path.read_text(encoding="utf-8"))
        if research_path.exists()
        else {
            "status": "AWAITING_TRAINING",
            "metrics": [],
            "cuda_available": False,
        }
    )
    forward_evaluation = (
        json.loads(forward_path.read_text(encoding="utf-8"))
        if forward_path.exists()
        else {"matured_days": 0, "rank_ic": None, "rank_ic_ir": None, "daily": []}
    )
    historical_execution = (
        json.loads(historical_path.read_text(encoding="utf-8"))
        if historical_path.exists()
        else {
            "status": "AWAITING_REPLAY",
            "evidence_status": "NO_EXECUTION_EVIDENCE",
            "strategies": [],
        }
    )
    point_in_time_path = STATE / "point-in-time" / "latest.json"
    point_in_time = (
        json.loads(point_in_time_path.read_text(encoding="utf-8"))
        if point_in_time_path.exists()
        else {"status": "AWAITING_MATERIALIZATION"}
    )
    governance = bootstrap_qlib_release(ROOT)
    drift_path = STATE / "qlib-governance" / "latest-drift.json"
    drift = (
        json.loads(drift_path.read_text(encoding="utf-8"))
        if drift_path.exists()
        else {"status": "AWAITING_OBSERVATIONS"}
    )
    review_path = STATE / "qlib-governance" / "long-term-review.json"
    long_term_review = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.exists()
        else {"status": "COLLECTING_EVIDENCE", "criteria": {}, "evaluation": {}}
    )
    challenger_config = yaml.safe_load(
        (ROOT / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
    )
    gate_config = challenger_config["research_gate"]
    model_metrics = next(
        (
            row
            for row in research.get("metrics", [])
            if row.get("model_id") == challenger_config["model_id"]
        ),
        None,
    )
    research["gate"] = {
        **gate_config,
        "passed": bool(
            model_metrics
            and int(model_metrics["samples"]) >= int(gate_config["minimum_samples"])
            and float(model_metrics["rank_ic"]) >= float(gate_config["minimum_rank_ic"])
            and float(model_metrics["rank_ic_ir"]) >= float(gate_config["minimum_rank_ic_ir"])
            and float(model_metrics.get("cost_adjusted_top_k_excess_return", -1))
            > float(gate_config["minimum_cost_adjusted_excess_return"])
            and int(research.get("stability", {}).get("seed_count", 0))
            >= int(gate_config["minimum_seed_count"])
            and float(research.get("stability", {}).get("positive_seed_ratio", 0))
            >= float(gate_config["minimum_positive_seed_ratio"])
        ),
    }
    promotion_gate = challenger_config["promotion_gate"]
    research["promotion_gate"] = {
        **promotion_gate,
        "passed": bool(
            model_metrics
            and float(model_metrics["rank_ic"]) >= float(promotion_gate["minimum_rank_ic"])
            and float(model_metrics["rank_ic_ir"]) >= float(promotion_gate["minimum_rank_ic_ir"])
            and int(forward_evaluation.get("matured_days", 0))
            >= int(promotion_gate["minimum_forward_days"])
            and float(forward_evaluation.get("rank_ic") or -1)
            > float(promotion_gate["minimum_forward_rank_ic"])
        ),
    }
    orders = [
        row for row in reversed(broker.orders()) if row.get("account_id") == QLIB_CHALLENGER_ACCOUNT
    ][:100]
    comparison_sources: dict[str, pd.DataFrame] = {}
    for account_id, table in (
        (MULTI_SECTOR_ACCOUNT, "multi_sector_account_daily"),
        (INTRADAY_ACCOUNT, "baseline_intraday_account_daily"),
        (QLIB_CHALLENGER_ACCOUNT, "qlib_challenger_account_daily"),
    ):
        try:
            history = store.read(table).sort_values("trade_date")
        except FileNotFoundError:
            history = pd.DataFrame()
        comparison_sources[account_id] = history
    baseline_history = comparison_sources.get(MULTI_SECTOR_ACCOUNT, pd.DataFrame())
    comparison_sources[INTRADAY_ACCOUNT] = build_branch_history(
        baseline_history,
        comparison_sources.get(INTRADAY_ACCOUNT, pd.DataFrame()),
        branch_account_id=INTRADAY_ACCOUNT,
        activation_date=INTRADAY_BASELINE_ACTIVATION_DATE,
    )
    fair_comparison = build_fair_comparison(
        comparison_sources,
        minimum_observation_days=20,
        target_volatility=0.10,
    )
    candidate_rows = candidate_catalog(ROOT)
    observed_candidate = next((row for row in candidate_rows if not row.get("archived")), None)
    candidate_history_all = pd.DataFrame()
    try:
        candidate_history_all = store.read(CANDIDATE_HISTORY_TABLE).sort_values("trade_date")
    except FileNotFoundError:
        pass
    active_history = comparison_sources.get(QLIB_CHALLENGER_ACCOUNT, pd.DataFrame())
    observation_policy = challenger_config.get("candidate_observation_policy", {})
    preliminary_days = int(observation_policy.get("preliminary_review_days", 20))
    formal_days = int(observation_policy.get("formal_review_days", 60))
    candidate_details: list[dict[str, object]] = []
    for row in candidate_rows:
        account_id = str(row["account_id"])
        history = (
            candidate_history_all.loc[
                candidate_history_all["account_id"].astype(str) == account_id
            ].copy()
            if not candidate_history_all.empty
            else pd.DataFrame()
        )
        version_comparison = build_fair_comparison(
            {
                MULTI_SECTOR_ACCOUNT: baseline_history,
                QLIB_CHALLENGER_ACCOUNT: active_history,
                account_id: history,
            },
            minimum_observation_days=preliminary_days,
            target_volatility=0.10,
        )
        common_days = int(version_comparison["common_observation_days"])
        stage = (
            "FORMAL_REVIEW_READY"
            if common_days >= formal_days
            else "PRELIMINARY_REVIEW"
            if common_days >= preliminary_days
            else "OBSERVING"
        )
        public_row = {
            key: value
            for key, value in row.items()
            if key not in {"latest_report", "latest_portfolio", "model_dir", "research_path"}
        }
        public_row.update(
            {
                "common_observation_days": common_days,
                "preliminary_review_days": preliminary_days,
                "formal_review_days": formal_days,
                "review_stage": (row.get("lifecycle") or {}).get("status", stage),
                "review_reason": (row.get("lifecycle") or {}).get("reason", "等待每日流水线评审"),
                "review_date": (row.get("lifecycle") or {}).get("review_date"),
                "checkpoint": (row.get("lifecycle") or {}).get("checkpoint", 0),
                "eligible_observation_days": (row.get("lifecycle") or {}).get("common_observation_days", 0),
            }
        )
        broker.initialize_account(1_000_000, account_id)
        candidate_details.append(
            {
                **public_row,
                "latest": row.get("latest_report") or {},
                "orders": [
                    order
                    for order in reversed(broker.orders())
                    if order.get("account_id") == account_id
                ][:100],
                "fills": broker.fills(account_id),
                "reconciliation": asdict(broker.reconcile(account_id)),
                "comparison": version_comparison,
            }
        )
    candidate_leaderboard = [
        {
            key: value
            for key, value in row.items()
            if key not in {"latest", "orders", "fills", "reconciliation", "comparison"}
        }
        for row in candidate_details
    ]
    all_strategy_histories = {
        MULTI_SECTOR_ACCOUNT: baseline_history,
        INTRADAY_ACCOUNT: comparison_sources.get(INTRADAY_ACCOUNT, pd.DataFrame()),
        QLIB_CHALLENGER_ACCOUNT: active_history,
        **{
            str(row["account_id"]): (
                candidate_history_all.loc[
                    candidate_history_all["account_id"].astype(str) == str(row["account_id"])
                ].copy()
                if not candidate_history_all.empty
                else pd.DataFrame()
            )
            for row in candidate_rows
        },
    }
    global_strategy_view = build_fair_comparison(
        all_strategy_histories,
        minimum_observation_days=preliminary_days,
        target_volatility=0.10,
    )
    # Global display metrics use each account's complete real operating span.
    # Candidate promotion evidence remains version-specific in candidate_details.
    global_strategy_view["metrics"] = [
        _account_performance(account_id, history)
        for account_id, history in all_strategy_histories.items()
    ]
    global_strategy_view["strategy_count"] = len(all_strategy_histories)
    candidate_observation: dict[str, object] = {
        "status": "AWAITING_CANDIDATE",
        "leaderboard": candidate_leaderboard,
        "latest": {},
        "orders": [],
        "fills": [],
        "candidates": [],
        "archived_candidates": [row for row in candidate_details if row.get("archived")],
        "comparison": {"status": "AWAITING_COMMON_DATES", "metrics": []},
    }
    if observed_candidate is not None:
        candidate_account = str(observed_candidate["account_id"])
        broker.initialize_account(1_000_000, candidate_account)
        try:
            candidate_history = store.read(CANDIDATE_HISTORY_TABLE)
            candidate_history = candidate_history.loc[
                candidate_history["account_id"].astype(str) == candidate_account
            ].sort_values("trade_date")
        except FileNotFoundError:
            candidate_history = pd.DataFrame()
        comparison_sources[candidate_account] = candidate_history
        fair_comparison = build_fair_comparison(
            comparison_sources,
            minimum_observation_days=20,
            target_volatility=0.10,
        )
        candidate_comparison = build_fair_comparison(
            {
                MULTI_SECTOR_ACCOUNT: baseline_history,
                QLIB_CHALLENGER_ACCOUNT: active_history,
                candidate_account: candidate_history,
            },
            minimum_observation_days=20,
            target_volatility=0.10,
        )
        candidate_orders = [
            row for row in reversed(broker.orders()) if row.get("account_id") == candidate_account
        ][:100]
        candidate_observation = {
            "status": "OBSERVING",
            "candidate_tag": observed_candidate["candidate_tag"],
            "account_id": candidate_account,
            "leaderboard": candidate_leaderboard,
            "latest": observed_candidate.get("latest_report") or {},
            "latest_trade_date": observed_candidate.get("latest_trade_date"),
            "research_gate_passed": observed_candidate.get("research_gate_passed", False),
            "orders": candidate_orders,
            "fills": broker.fills(candidate_account),
            "reconciliation": asdict(broker.reconcile(candidate_account)),
            "comparison": candidate_comparison,
            "candidates": [row for row in candidate_details if not row.get("archived")],
            "archived_candidates": [row for row in candidate_details if row.get("archived")],
            "preliminary_review_days": preliminary_days,
            "formal_review_days": formal_days,
            "global_strategy_view": global_strategy_view,
        }
    candidate_observation["leaderboard"] = [row for row in candidate_leaderboard if not row.get("archived")]
    candidate_observation["archived_leaderboard"] = [row for row in candidate_leaderboard if row.get("archived")]
    candidate_observation["lifecycle_policy"] = observation_policy
    candidate_observation.setdefault("global_strategy_view", global_strategy_view)
    instrument_names = _instrument_names(store)
    cycle_marks: dict[str, float] = {}
    comparison_account_ids = [
        MULTI_SECTOR_ACCOUNT,
        INTRADAY_ACCOUNT,
        QLIB_CHALLENGER_ACCOUNT,
        *(str(row["account_id"]) for row in candidate_rows),
    ]
    cycle_symbols = {
        str(row["symbol"])
        for account_id in comparison_account_ids
        for row in broker.fills(account_id)
    }
    if cycle_symbols:
        daily = store.read(
            "daily",
            columns=["ts_code", "trade_date", "close"],
            filters=[("ts_code", "in", sorted(cycle_symbols))],
        )
        latest_marks = daily.sort_values("trade_date").groupby("ts_code", as_index=False).tail(1)
        cycle_marks = {
            str(row["ts_code"]): float(row["close"]) for row in latest_marks.to_dict("records")
        }
    trade_cycle_analysis = {}
    for account_id in comparison_account_ids:
        fills = broker.fills(account_id)
        actions = broker.corporate_actions(account_id)
        if account_id == INTRADAY_ACCOUNT:
            # The intraday account is a true fork of the factor baseline.  Its
            # opening positions therefore need the parent's pre-fork ledger
            # when reconstructing complete holding cycles; only post-fork
            # executions come from the independent account.
            fills = [
                row
                for row in broker.fills(MULTI_SECTOR_ACCOUNT)
                if str(row.get("trade_date", "")) < INTRADAY_BASELINE_ACTIVATION_DATE
            ] + [
                row
                for row in fills
                if str(row.get("trade_date", "")) >= INTRADAY_BASELINE_ACTIVATION_DATE
            ]
            actions = [
                row
                for row in broker.corporate_actions(MULTI_SECTOR_ACCOUNT)
                if str(row.get("trade_date", "")) < INTRADAY_BASELINE_ACTIVATION_DATE
            ] + [
                row
                for row in actions
                if str(row.get("trade_date", "")) >= INTRADAY_BASELINE_ACTIVATION_DATE
            ]
        trade_cycle_analysis[account_id] = analyze_trade_cycles(
            account_id,
            fills,
            actions,
            cycle_marks,
        )
    try:
        intraday_portfolio = broker.account_snapshot(cycle_marks, INTRADAY_ACCOUNT)
    except ValueError:
        intraday_portfolio = {
            "equity": 0,
            "cash": 0,
            "market_value": 0,
            "positions": [],
        }
    return {
        "account_id": QLIB_CHALLENGER_ACCOUNT,
        "research": research,
        "forward": forward_evaluation,
        "diagnostics": (
            json.loads((STATE / "challenger-diagnostics/latest.json").read_text(encoding="utf-8"))
            if (STATE / "challenger-diagnostics/latest.json").exists() else {}
        ),
        "latest": latest,
        "next_trade_date": _next_open_trade_date(store, str(latest.get("trade_date", ""))),
        "orders": orders,
        "fills": broker.fills(QLIB_CHALLENGER_ACCOUNT),
        "reconciliation": asdict(broker.reconcile(QLIB_CHALLENGER_ACCOUNT)),
        "comparison": fair_comparison,
        "trade_cycle_analysis": trade_cycle_analysis,
        "historical_execution": historical_execution,
        "point_in_time": point_in_time,
        "governance": governance,
        "drift": drift,
        "long_term_review": long_term_review,
        "candidate_observation": candidate_observation,
        "intraday_baseline": {
            "account_id": INTRADAY_ACCOUNT,
            "status": "BRANCHED_FROM_BASELINE_20260908",
            "parent_account_id": MULTI_SECTOR_ACCOUNT,
            "activation_date": INTRADAY_BASELINE_ACTIVATION_DATE,
            "portfolio": intraday_portfolio,
            "orders": [
                row
                for row in reversed(broker.orders())
                if row.get("account_id") == INTRADAY_ACCOUNT
            ][:100],
            "fills": broker.fills(INTRADAY_ACCOUNT),
        },
        "baseline": {
            "account_id": MULTI_SECTOR_ACCOUNT,
            "latest": _latest_multi_sector_shadow(),
            "orders": [
                row
                for row in reversed(broker.orders())
                if row.get("account_id") == MULTI_SECTOR_ACCOUNT
            ][:100],
            "fills": broker.fills(MULTI_SECTOR_ACCOUNT),
        },
        "symbol_names": instrument_names,
    }


def _account_performance(account_id: str, history: pd.DataFrame) -> dict[str, object]:
    if history.empty or "equity" not in history:
        return {
            "account_id": account_id,
            "observation_days": 0,
            "total_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
        }
    equity = pd.to_numeric(history["equity"], errors="coerce").dropna()
    returns = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1
    volatility = float(returns.std(ddof=1) * (252**0.5)) if len(returns) > 1 else 0.0
    return {
        "account_id": account_id,
        "observation_days": len(equity),
        "start_date": str(history.sort_values("trade_date").iloc[0]["trade_date"]),
        "end_date": str(history.sort_values("trade_date").iloc[-1]["trade_date"]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annualized_volatility": volatility,
        "sharpe": (
            float(returns.mean() / returns.std(ddof=1) * (252**0.5))
            if len(returns) > 1 and returns.std(ddof=1)
            else None
        ),
        "max_drawdown": float(drawdown.min()),
    }


@app.get("/api/monthly-acceptance")
def monthly_acceptance() -> dict[str, object]:
    store = ParquetStore(DATA)
    try:
        account_daily = store.read("multi_sector_account_daily")
        records = _records(account_daily)
        observation_date = str(account_daily["trade_date"].astype(str).max())
    except FileNotFoundError:
        records = []
        observation_date = datetime.now(SHANGHAI).strftime("%Y%m%d")
    broker = PaperBroker(PAPER_DATABASE)
    registry = ModelRegistry(
        STATE / "model_registry.sqlite3",
        STATE / "model-versions",
    )
    result = evaluate_monthly_cycle(
        account_daily=records,
        fills=broker.fills(MULTI_SECTOR_ACCOUNT),
        model_versions=registry.snapshot(100)["versions"],
        start_date="20260727",
        observation_date=observation_date,
    )
    payload = asdict(result)
    report = STATE / "monthly-acceptance" / f"{result.cycle_id}.json"
    payload["report_path"] = str(report) if report.exists() else None
    return payload


def _latest_bank_shadow() -> dict[str, object]:
    directory = STATE / "bank-shadow"
    reports = sorted(directory.glob("*.json")) if directory.exists() else []
    if not reports:
        return {
            "status": "AWAITING_FIRST_RUN",
            "holdings": [],
            "timing": {},
            "ranking": [],
            "orders": [],
            "executions": [],
            "portfolio": {
                "cash": 1_000_000,
                "market_value": 0,
                "equity": 1_000_000,
                "positions": [],
            },
        }
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def _latest_multi_sector_shadow() -> dict[str, object]:
    directory = STATE / "multi-sector-shadow"
    reports = sorted(directory.glob("*.json")) if directory.exists() else []
    if not reports:
        return {
            "status": "AWAITING_FIRST_RUN",
            "target_weights": {},
            "symbol_sectors": {},
            "attribution": [],
            "portfolio": {
                "cash": 1_000_000,
                "market_value": 0,
                "equity": 1_000_000,
                "positions": [],
            },
        }
    return json.loads(reports[-1].read_text(encoding="utf-8"))


@app.get("/api/overview")
def overview(symbol: str = SYMBOL) -> dict[str, object]:
    candidates = task_service.candidates()
    candidate = next(
        (item for item in candidates if item["symbol"] == symbol),
        None,
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    broker = PaperBroker(PAPER_DATABASE)
    store = ParquetStore(DATA)
    bars = load_total_return_stock_bars(store, symbol)
    latest = bars.iloc[-1]
    latest_date = str(latest["date"]).replace("-", "")[:8]
    decision = trend_decision(
        bars,
        latest_date,
        active_weight=float(candidate["target_weight"]),
        strategy_id=str(candidate["strategy_id"]),
    )
    portfolio = broker.portfolio(decision.close, symbol, ACCOUNT)
    reconciliation = asdict(broker.reconcile(ACCOUNT))
    return {
        "mode": "PAPER_ONLY",
        "symbol": symbol,
        "symbol_name": candidate["name"],
        "latest_data_date": decision.as_of_date,
        "latest_price": decision.close,
        "decision": decision.to_dict(),
        "portfolio": portfolio,
        "reconciliation": reconciliation,
        "recent_runs": task_service.runs(6),
        "scheduler": task_service.config(),
    }


@app.get("/api/candidates")
def candidates() -> dict[str, object]:
    store = ParquetStore(DATA)
    broker = PaperBroker(PAPER_DATABASE)
    rows = []
    for candidate in task_service.candidates():
        symbol = str(candidate["symbol"])
        bars = load_total_return_stock_bars(store, symbol)
        latest_date = str(bars.iloc[-1]["date"]).replace("-", "")[:8]
        decision = trend_decision(
            bars,
            latest_date,
            active_weight=float(candidate["target_weight"]),
            strategy_id=str(candidate["strategy_id"]),
        )
        position = broker.portfolio(decision.close, symbol, ACCOUNT)
        rows.append(
            {
                **candidate,
                "latest_date": decision.as_of_date,
                "latest_price": decision.close,
                "action": decision.action,
                "reason_code": decision.reason_code,
                "signal_weight": decision.target_weight,
                "position_quantity": position["position_quantity"],
                "market_value": int(position["position_quantity"]) * decision.close,
            }
        )
    return {
        "items": rows,
        "active_count": sum(bool(row["enabled"]) for row in rows),
        "target_gross_weight": sum(float(row["target_weight"]) for row in rows if row["enabled"]),
        "max_gross_weight": BacktestConfig.from_yaml(
            ROOT / "configs" / "default.yaml"
        ).max_gross_exposure,
    }


@app.post("/api/candidates")
def add_candidate(payload: dict[str, str]) -> dict[str, object]:
    try:
        return task_service.add_candidate(payload["symbol"], payload["name"])
    except (KeyError, ValueError, sqlite3.IntegrityError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/orders")
def orders() -> list[dict[str, object]]:
    database = PaperBroker(PAPER_DATABASE)
    return list(reversed(database.orders()))[:50]


@app.get("/api/fills")
def fills() -> list[dict[str, object]]:
    if not PAPER_DATABASE.exists():
        return []
    with sqlite3.connect(PAPER_DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM fills ORDER BY id DESC LIMIT 50").fetchall()
        return [dict(row) for row in rows]


@app.get("/api/execution")
def execution() -> dict[str, object]:
    broker = PaperBroker(PAPER_DATABASE)
    with sqlite3.connect(PAPER_DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        fills_rows = connection.execute("SELECT * FROM fills ORDER BY id DESC LIMIT 100").fetchall()
        attempt_rows = connection.execute(
            "SELECT * FROM execution_attempts ORDER BY id DESC LIMIT 100"
        ).fetchall()
        position_rows = connection.execute("SELECT * FROM positions ORDER BY symbol").fetchall()
    return {
        "orders": list(reversed(broker.orders()))[:100],
        "fills": [dict(row) for row in fills_rows],
        "attempts": [dict(row) for row in attempt_rows],
        "positions": [dict(row) for row in position_rows],
        "reconciliation": asdict(broker.reconcile(ACCOUNT)),
    }


@app.get("/api/research")
def research(symbol: str = SYMBOL) -> dict[str, object]:
    candidate = next(
        (item for item in task_service.candidates() if item["symbol"] == symbol),
        None,
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return _research_payload(
        symbol,
        str(candidate["name"]),
        str(candidate["research_status"]),
    )


@app.get("/api/tasks")
def tasks() -> list[dict[str, Any]]:
    return task_service.runs()


@app.get("/api/task-operations")
def task_operations() -> dict[str, object]:
    return {
        "runs": task_service.runs(30),
        "steps": task_service.steps(100),
        "notifications": task_service.notifications(100),
    }


@app.get("/api/operations-center")
def operations_center() -> dict[str, object]:
    now = datetime.now(SHANGHAI)
    config = task_service.config()
    scheduled = now.replace(
        hour=int(config["hour"]),
        minute=int(config["minute"]),
        second=0,
        microsecond=0,
    )
    if scheduled <= now:
        scheduled += timedelta(days=1)
    broker = PaperBroker(PAPER_DATABASE)
    pending_orders = [
        row
        for row in reversed(broker.orders())
        if row.get("account_id") == MULTI_SECTOR_ACCOUNT and row.get("status") == "PENDING"
    ]
    with sqlite3.connect(PAPER_DATABASE) as connection:
        connection.row_factory = sqlite3.Row
        attempts = [
            dict(row)
            for row in connection.execute(
                """
                SELECT a.*, o.account_id, o.symbol, o.side, o.quantity
                FROM execution_attempts a
                JOIN orders o ON o.idempotency_key = a.idempotency_key
                WHERE o.account_id = ?
                ORDER BY a.id DESC LIMIT 100
                """,
                (MULTI_SECTOR_ACCOUNT,),
            )
        ]
    today = now.strftime("%Y%m%d")
    preview = _daily_run_preview(today)
    return {
        "server_time": now.isoformat(),
        "scheduler": {
            **config,
            "next_run": scheduled.isoformat(),
            "seconds_to_next_run": max(0, int((scheduled - now).total_seconds())),
        },
        "pending_orders": pending_orders,
        "deferred_attempts": [row for row in attempts if row["outcome"] != "FILLED"],
        "recent_attempts": attempts,
        "task_runs": task_service.runs(30),
        "task_steps": task_service.steps(100),
        "notifications": task_service.notifications(100),
        "preview": preview,
        "readiness": _operational_readiness(today),
        "open_execution": _open_execution_status(today),
    }


@app.get("/api/operational-readiness")
def operational_readiness() -> dict[str, object]:
    return _operational_readiness(datetime.now(SHANGHAI).strftime("%Y%m%d"))


def _operational_readiness(trade_date: str) -> dict[str, object]:
    store = ParquetStore(DATA)
    checks = []
    try:
        calendar = store.read("trade_calendar")
        current = calendar.loc[calendar["cal_date"].astype(str) == trade_date]
        covered = len(current) == 1
        session = (
            "OPEN"
            if covered and int(current.iloc[0]["is_open"]) == 1
            else "CLOSED"
            if covered
            else "UNKNOWN"
        )
    except FileNotFoundError:
        covered = False
        session = "UNKNOWN"
    checks.append(
        {
            "code": "OFFICIAL_CALENDAR",
            "status": "PASS" if covered else "REPAIR",
            "message": (
                f"正式日历确认当日{session}"
                if covered
                else "日度流水线启动后必须先同步正式交易日历"
            ),
        }
    )
    config = task_service.config()
    scheduler_alive = bool(task_service._thread and task_service._thread.is_alive())
    checks.extend(
        [
            {
                "code": "SCHEDULER_ENABLED",
                "status": "PASS" if config["enabled"] else "BLOCK",
                "message": "服务端日度调度已启用" if config["enabled"] else "调度已暂停",
            },
            {
                "code": "SCHEDULER_THREAD",
                "status": "PASS" if scheduler_alive else "BLOCK",
                "message": (
                    "调度线程正在运行" if scheduler_alive else "调度线程未运行，需要重启API服务"
                ),
            },
        ]
    )
    blocking = sum(row["status"] == "BLOCK" for row in checks)
    repair = sum(row["status"] == "REPAIR" for row in checks)
    return {
        "trade_date": trade_date,
        "status": ("BLOCKED" if blocking else "REQUIRES_SYNC" if repair else "READY"),
        "market_session": session,
        "blocking_count": blocking,
        "repair_count": repair,
        "checks": checks,
    }


@app.get("/api/tasks/daily-run/preview")
def preview_daily_run(trade_date: str | None = None) -> dict[str, object]:
    selected = trade_date or datetime.now(SHANGHAI).strftime("%Y%m%d")
    if len(selected) != 8 or not selected.isdigit():
        raise HTTPException(status_code=400, detail="trade_date must use YYYYMMDD")
    try:
        datetime.strptime(selected, "%Y%m%d").replace(tzinfo=SHANGHAI)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="invalid trade_date") from error
    return _daily_run_preview(selected)


def _daily_run_preview(trade_date: str) -> dict[str, object]:
    preview = task_service.preview(trade_date)
    store = ParquetStore(DATA)
    market_session = "UNKNOWN"
    calendar_source = "LOCAL_TRADE_CALENDAR"
    try:
        calendar = store.read("trade_calendar")
        row = calendar.loc[calendar["cal_date"].astype(str) == trade_date]
        if not row.empty:
            market_session = "OPEN" if int(row.iloc[-1]["is_open"]) == 1 else "CLOSED"
        else:
            calendar_source = "WEEKDAY_ESTIMATE"
            market_session = (
                "OPEN_ESTIMATED"
                if datetime.strptime(trade_date, "%Y%m%d").replace(tzinfo=SHANGHAI).weekday() < 5
                else "CLOSED_ESTIMATED"
            )
    except FileNotFoundError:
        calendar_source = "WEEKDAY_ESTIMATE"
        market_session = (
            "OPEN_ESTIMATED"
            if datetime.strptime(trade_date, "%Y%m%d").replace(tzinfo=SHANGHAI).weekday() < 5
            else "CLOSED_ESTIMATED"
        )
    broker = PaperBroker(PAPER_DATABASE)
    pending = [
        row
        for row in broker.orders()
        if row.get("account_id") == MULTI_SECTOR_ACCOUNT and row.get("status") == "PENDING"
    ]
    eligible = [row for row in pending if str(row["signal_date"]) < trade_date]
    latest_shadow = _latest_multi_sector_shadow()
    return {
        **preview,
        "market_session": market_session,
        "calendar_source": calendar_source,
        "pending_order_count": len(pending),
        "eligible_for_execution_count": len(eligible),
        "eligible_buy_count": sum(row["side"] == "BUY" for row in eligible),
        "eligible_sell_count": sum(row["side"] == "SELL" for row in eligible),
        "latest_shadow_date": latest_shadow.get("trade_date"),
        "risk_state": broker.risk_state(MULTI_SECTOR_ACCOUNT),
        "expected_effects": [
            "同步并校验指定交易日数据",
            f"最多尝试撮合 {len(eligible)} 笔此前待执行订单",
            "重新计算五行业目标权重并生成幂等信号",
            "更新模拟持仓、现金、归因、风险状态和对账",
        ],
        "warnings": (
            ["本地交易日历未覆盖该日期，开闭市状态仅按工作日估计"]
            if calendar_source == "WEEKDAY_ESTIMATE"
            else []
        ),
        "mutates_state": False,
    }


@app.post("/api/notifications/{notification_id}/acknowledge")
def acknowledge_notification(notification_id: int) -> dict[str, Any]:
    try:
        return task_service.acknowledge_notification(notification_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="notification not found") from error


@app.get("/api/task-config")
def task_config() -> dict[str, object]:
    return task_service.config()


@app.post("/api/task-config")
def update_task_config(payload: dict[str, object]) -> dict[str, object]:
    try:
        enabled = bool(payload["enabled"])
        hour = int(payload["hour"])
        minute = int(payload["minute"])
        return task_service.update_config(enabled, hour, minute)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/tasks/daily-run")
def trigger_daily(payload: dict[str, str] | None = None) -> dict[str, object]:
    trade_date = (payload or {}).get("trade_date", datetime.now(SHANGHAI).strftime("%Y%m%d"))
    if len(trade_date) != 8 or not trade_date.isdigit():
        raise HTTPException(status_code=400, detail="trade_date must use YYYYMMDD")
    return {
        "run_id": task_service.trigger_recovery(trade_date, force_current_session=True),
        "status": "QUEUED_RECOVERY",
        "mode": "RECOVER_MISSING_TRADING_DAYS_THEN_RECALCULATE",
    }


@app.get("/api/reports/{filename}")
def report(filename: str) -> dict[str, object]:
    safe_name = Path(filename).name
    target = STATE / "daily-runs" / safe_name
    if not target.exists() or target.suffix != ".json":
        raise HTTPException(status_code=404, detail="report not found")
    return json.loads(target.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def _research_payload(symbol: str, name: str, research_status: str) -> dict[str, object]:
    store = ParquetStore(DATA)
    config = BacktestConfig.from_yaml(ROOT / "configs" / "default.yaml")
    primary = research_single_stock(store, symbol, config)
    robustness = robustness_single_stock(store, symbol, config)
    weighted = research_weighted_strategies(store, symbol, config)
    allocations = allocation_comparison(store, symbol, config)
    annual = annual_comparison(store, symbol, config)
    result = candidate_result(store, symbol, config, 0.8)
    ledger = trade_ledger(result)
    curve = result.equity.copy()
    curve["date"] = curve["date"].astype(str).str[:10]
    sampled = curve.iloc[::20].copy()
    if sampled.iloc[-1]["date"] != curve.iloc[-1]["date"]:
        sampled = sampled._append(curve.iloc[-1], ignore_index=True)
    coverage = {}
    for table in ("daily_basic", "dividend", "fina_indicator"):
        try:
            frame = store.read(table, filters=[("ts_code", "==", symbol)])
            coverage[table] = {"rows": len(frame)}
            date_column = "trade_date" if table == "daily_basic" else "ann_date"
            dates = frame[date_column].dropna().astype(str)
            coverage[table].update(
                {
                    "first_date": dates.min() if not dates.empty else None,
                    "last_date": dates.max() if not dates.empty else None,
                }
            )
        except FileNotFoundError:
            coverage[table] = {"rows": 0, "first_date": None, "last_date": None}
    return {
        "symbol": symbol,
        "symbol_name": name,
        "strategy": {
            "id": "cmb_ma_120_250_v1",
            "fast": 120,
            "slow": 250,
            "account_weight": 0.10,
            "research_sleeve_weight": 0.80,
            "execution": "收盘信号，下一交易日开盘成交",
            "status": research_status,
        },
        "fundamental_context": _fundamental_context(symbol),
        "primary": _records(primary),
        "robustness": _records(robustness),
        "weighted_strategies": _records(weighted),
        "data_coverage": coverage,
        "allocations": _records(allocations),
        "annual": _records(annual),
        "trades": _records(ledger),
        "equity_curve": _records(sampled[["date", "equity", "drawdown"]]),
    }


def _instrument_names(store: ParquetStore) -> dict[str, str]:
    instruments = store.read("instruments", columns=["ts_code", "name"])
    return dict(
        zip(
            instruments["ts_code"].astype(str),
            instruments["name"].astype(str),
            strict=True,
        )
    )


def _records(frame: Any) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _fundamental_context(symbol: str) -> dict[str, object] | None:
    if symbol != "600900.SH":
        return None
    return {
        "summary": (
            "公司以大型水电运营为核心，运行管理长江干流六座梯级电站；"
            "量化趋势策略仅刻画价格路径，不替代对水文、电价和安全生产的判断。"
        ),
        "strengths": [
            "六座梯级电站联合调度形成规模和运营能力",
            "大型水电兼具电能量、调节与绿色环境价值",
            "现金分红特征突出，适合与价格趋势模型分开评估",
        ],
        "risks": [
            "长江中上游来水不确定性直接影响发电量",
            "大型机组、大坝及极端天气带来安全生产风险",
            "统一电力市场改革使电量、电价和辅助服务收益面临变化",
            "境内外投资面临市场、政策、汇率与地缘政治风险",
        ],
        "source": "中国长江电力股份有限公司2025年年度报告",
        "source_url": ("https://www.cypc.com.cn/cypc/attachDir/2026/05/2026051815345898336.pdf"),
    }
