from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from .provider import MarketDataProvider
from .quality import (
    validate_adjustment_factors,
    validate_calendar,
    validate_daily_bars,
    validate_instruments,
)
from .store import ParquetStore


@dataclass(frozen=True)
class SyncSummary:
    open_days: int
    daily_rows: int
    factor_rows: int


def sync_daily_frontier(
    provider: MarketDataProvider,
    store: ParquetStore,
    cutoff: str,
    max_sessions: int = 20,
) -> SyncSummary:
    """Advance official market data independently of older strategy failures.

    Caller supplies a completed-close cutoff, not today's intraday date. Existing
    history is required; this bounded repair is not an implicit full backfill.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    frontiers = []
    for table in ("daily", "adj_factor"):
        dates = store.curated_values(table, "trade_date")
        if not dates:
            raise ValueError("official frontier requires existing history: " + table)
        frontiers.append(max(dates))
    calendar = store.read("trade_calendar", columns=["cal_date", "is_open"])
    dates = sorted(set(calendar.loc[
        (calendar.cal_date.astype(str) > min(frontiers))
        & (calendar.cal_date.astype(str) <= cutoff)
        & calendar.is_open.astype(str).eq("1"), "cal_date"
    ].astype(str)))[:max_sessions]
    if not dates:
        return SyncSummary(0, 0, 0)
    return sync_daily_history(provider, store, dates[0], dates[-1], batch_days=5)


def sync_reference_data(
    provider: MarketDataProvider,
    store: ParquetStore,
    start_date: str,
    end_date: str,
) -> tuple[object, object]:
    instruments = provider.instruments()
    validate_instruments(instruments)
    instrument_snapshot = store.save_snapshot(
        "instruments",
        instruments,
        provider.name,
        ["ts_code"],
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )

    calendar = provider.trading_calendar(start_date, end_date)
    validate_calendar(calendar)
    calendar = calendar.copy()
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="raise").astype("int64")
    calendar_key = f"{start_date}_{end_date}"
    try:
        calendar_snapshot = store.save_snapshot(
            "trade_calendar", calendar, provider.name, ["cal_date"], calendar_key
        )
    except FileExistsError:
        # Future exchange calendars can be revised after the first capture.
        # Preserve both immutable versions and let the curated table keep the
        # latest observation instead of blocking the entire daily pipeline.
        digest = hashlib.sha256(
            calendar.sort_values("cal_date").to_json(orient="records").encode()
        ).hexdigest()[:12]
        calendar_snapshot = store.save_snapshot(
            "trade_calendar",
            calendar,
            provider.name,
            ["cal_date"],
            f"{calendar_key}_revision_{digest}",
        )
    return instrument_snapshot, calendar_snapshot


def sync_daily_history(
    provider: MarketDataProvider,
    store: ParquetStore,
    start_date: str,
    end_date: str,
    batch_days: int = 20,
    progress: object | None = None,
) -> SyncSummary:
    if batch_days <= 0:
        raise ValueError("batch_days must be positive")
    calendar = provider.trading_calendar(start_date, end_date)
    validate_calendar(calendar)
    open_days = sorted(
        calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"].astype(str)
    )
    daily_rows = 0
    factor_rows = 0
    pending_bars: list[tuple[str, object]] = []
    pending_factors: list[tuple[str, object]] = []
    recovered_bars: list[object] = []
    recovered_factors: list[object] = []
    curated_daily_dates = store.curated_values("daily", "trade_date")
    curated_factor_dates = store.curated_values("adj_factor", "trade_date")
    for index, trade_date in enumerate(open_days, start=1):
        if not store.has_snapshot("daily", provider.name, trade_date):
            bars = provider.daily_bars(trade_date)
            validate_daily_bars(bars, trade_date)
            pending_bars.append((trade_date, bars))
            daily_rows += len(bars)
        elif trade_date not in curated_daily_dates:
            recovered_bars.append(
                store.read_snapshot("daily", provider.name, trade_date)
            )

        if not store.has_snapshot("adj_factor", provider.name, trade_date):
            factors = provider.adjustment_factors(trade_date)
            validate_adjustment_factors(factors, trade_date)
            pending_factors.append((trade_date, factors))
            factor_rows += len(factors)
        elif trade_date not in curated_factor_dates:
            recovered_factors.append(
                store.read_snapshot("adj_factor", provider.name, trade_date)
            )

        should_flush = index % batch_days == 0 or index == len(open_days)
        if should_flush:
            store.save_snapshots(
                "daily",
                pending_bars,
                provider.name,
                ["ts_code", "trade_date"],
            )
            store.save_snapshots(
                "adj_factor",
                pending_factors,
                provider.name,
                ["ts_code", "trade_date"],
            )
            store.merge_curated(
                "daily", recovered_bars, ["ts_code", "trade_date"]
            )
            store.merge_curated(
                "adj_factor", recovered_factors, ["ts_code", "trade_date"]
            )
            pending_bars.clear()
            pending_factors.clear()
            recovered_bars.clear()
            recovered_factors.clear()
            if callable(progress):
                progress(index, len(open_days), trade_date, daily_rows, factor_rows)
    return SyncSummary(len(open_days), daily_rows, factor_rows)


def sync_etf_reference_data(
    provider: MarketDataProvider,
    store: ParquetStore,
) -> object:
    instruments = provider.etf_instruments()
    validate_instruments(instruments)
    return store.save_snapshot(
        "etf_instruments",
        instruments,
        provider.name,
        ["ts_code"],
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    )


def sync_etf_history(
    provider: MarketDataProvider,
    store: ParquetStore,
    start_date: str,
    end_date: str,
) -> SyncSummary:
    calendar = provider.trading_calendar(start_date, end_date)
    validate_calendar(calendar)
    open_days = sorted(
        calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"].astype(str)
    )
    daily_rows = 0
    factor_rows = 0
    for trade_date in open_days:
        bars = provider.etf_daily_bars(trade_date)
        validate_daily_bars(bars, trade_date)
        store.save_snapshot(
            "etf_daily", bars, provider.name, ["ts_code", "trade_date"], trade_date
        )
        daily_rows += len(bars)

        factors = provider.etf_adjustment_factors(trade_date)
        validate_adjustment_factors(factors, trade_date)
        store.save_snapshot(
            "etf_adj_factor",
            factors,
            provider.name,
            ["ts_code", "trade_date"],
            trade_date,
        )
        factor_rows += len(factors)
    return SyncSummary(len(open_days), daily_rows, factor_rows)
