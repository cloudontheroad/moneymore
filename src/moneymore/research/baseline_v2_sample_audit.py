"""Explain historical exclusions without relabeling missing prices as suspensions.

Reads only cached sources and writes a new isolated evidence directory. Does not
rebuild the fitted panel, relax its gates, or change any strategy account.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ..challenger_diagnostics import trading_dates, write_diagnostic
from ..data.fundamental_research import research_financial_frame
from .baseline_v2_financials import common_period_features
from .baseline_v2_industry import industry_as_of


def exclusion_reason(*, observations, expected, financial_ok, industry_ok):
    """Exclusive priority for accounting; separate flags retain overlapping causes."""
    if observations == 0:
        return "NO_VALID_PRICE_HISTORY"
    if observations < expected:
        return "INCOMPLETE_PRICE_WINDOW_NOT_PROVEN_SUSPENSION_OR_IPO"
    if not financial_ok:
        return "REQUIRED_FINANCIAL_MISSING"
    if not industry_ok:
        return "HISTORICAL_INDUSTRY_UNKNOWN"
    return "ELIGIBLE"


def classify_price_history(detail, instruments, calendar):
    """Listing date explains insufficient age, not absence of older stock prices."""
    result = detail.copy()
    listing = pd.to_datetime(
        instruments.set_index("ts_code").list_date, format="%Y%m%d", errors="coerce"
    )
    result["listing_date"] = result.symbol.map(listing)
    starts = {
        d: pd.Timestamp(calendar[max(0, calendar.index(d) - 252)]) for d in result.date.unique()
    }
    result["price_window_start"] = result.date.map(starts)
    incomplete = result.reason.eq("INCOMPLETE_PRICE_WINDOW_NOT_PROVEN_SUSPENSION_OR_IPO")
    result["price_gap_class"] = "NOT_PRICE_EXCLUDED"
    result.loc[incomplete, "price_gap_class"] = "LISTING_METADATA_UNKNOWN"
    result.loc[
        incomplete & result.listing_date.gt(result.price_window_start), "price_gap_class"
    ] = "LISTED_TOO_RECENT_FOR_253_SESSION_RULE"
    result.loc[
        incomplete & result.listing_date.le(result.price_window_start), "price_gap_class"
    ] = "OLDER_STOCK_GAP_REQUIRES_SUSPENSION_OR_BACKFILL_PROOF"
    return result


def refine_cached_price_audit(root, store, protocol):
    import json

    pointer = root / "state/baseline-v2-validation/sample-audit.json"
    source = json.loads(pointer.read_text(encoding="utf-8"))
    directory = root / "state/baseline-v2-validation/price-gap-audits" / uuid4().hex
    directory.mkdir(parents=True)
    frame = pd.read_parquet(Path(source["output_directory"]) / "sample-exclusions.parquet")
    refined = classify_price_history(
        frame, store.read("instruments"), trading_dates(store, protocol["as_of"])
    )
    refined.to_parquet(directory / "price-gap-classification.parquet", index=False)
    report = {
        "status": "CLASSIFIED_NOT_RELEASED",
        "source_audit": source["output_directory"],
        "source_panel_sha256": source["source_panel_sha256"],
        "output_directory": str(directory),
        "class_counts": refined.price_gap_class.value_counts().to_dict(),
        "eligibility_changed": False,
        "real_orders_enabled": False,
        "caveat": "Listing date is not suspension proof; no price values filled.",
    }
    write_diagnostic(directory / "report.json", report)
    write_diagnostic(root / "state/baseline-v2-validation/price-gap-audit.json", report)
    return report


def audit_samples(root, store, protocol, schedule):
    run = (
        root
        / "state/baseline-v2-validation/sample-audits"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    run.mkdir(parents=True)
    basic = store.read("baseline_v2_daily_basic")
    snapshots = {}
    for signal, source in schedule:
        f = basic.loc[basic.trade_date.eq(source)].copy()
        s = f.ts_code.astype(str)
        f = f.loc[
            (
                (s.str.endswith(".SH") & s.str.startswith(("60", "68")))
                | (s.str.endswith(".SZ") & s.str.startswith(("00", "30")))
            )
            & pd.to_numeric(f.total_mv, errors="coerce").gt(0)
        ]
        f = f.sort_values(["total_mv", "ts_code"], ascending=[False, True]).head(1000)
        if len(f) != 1000 or f.ts_code.duplicated().any():
            raise ValueError("INVALID_TOP1000_SNAPSHOT:" + signal)
        snapshots[signal] = f
    symbols = sorted(set().union(*(set(f.ts_code) for f in snapshots.values())))
    calendar = [d for d in trading_dates(store, protocol["as_of"]) if d <= protocol["as_of"]]
    start = calendar[max(0, calendar.index(schedule[0][0]) - 252)]
    filters = [
        ("ts_code", "in", symbols),
        ("trade_date", ">=", start),
        ("trade_date", "<=", protocol["as_of"]),
    ]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    close = daily.pivot(index="trade_date", columns="ts_code", values="close")
    factors = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor")
    history_calendar = [d for d in calendar if start <= d]
    close = close.reindex(index=history_calendar, columns=symbols)
    factors = factors.reindex(index=history_calendar, columns=symbols)
    valid_close = np.isfinite(close) & close.gt(0)
    valid_adj = np.isfinite(factors) & factors.gt(0)
    valid = valid_close & valid_adj
    finances = research_financial_frame(store)
    membership = store.read("baseline_v2_industry_membership")
    rows = []
    for signal, source in schedule:
        idx = history_calendar.index(signal)
        window = history_calendar[max(0, idx - 252) : idx + 1]
        financial = common_period_features(finances, signal).set_index("ts_code")
        industries = industry_as_of(membership, signal)
        for symbol in snapshots[signal].ts_code:
            flags = valid.loc[window, symbol]
            count = int(flags.sum())
            fiscal_ok = (
                symbol in financial.index
                and pd.notna(financial.loc[symbol, "roe"])
                and pd.notna(financial.loc[symbol, "q_sales_yoy"])
            )
            industry_ok = symbol in industries.index
            # Prior-day daily_basic is positive source evidence, but only for
            # that day. It cannot prove every absent day was a suspension.
            snapshot_price_missing = not bool(valid_close.loc[source, symbol])
            row = {
                "date": signal,
                "symbol": symbol,
                "expected_price_sessions": 253,
                "valid_price_sessions": count,
                "missing_close_sessions": int((~valid_close.loc[window, symbol]).sum()),
                "missing_adjustment_with_close": int(
                    (valid_close.loc[window, symbol] & ~valid_adj.loc[window, symbol]).sum()
                ),
                "snapshot_day_close_missing": snapshot_price_missing,
                "financial_ok": bool(fiscal_ok),
                "industry_ok": bool(industry_ok),
                "first_cached_price_date": (
                    str(valid.index[valid[symbol]][0]) if valid[symbol].any() else None
                ),
            }
            row["reason"] = exclusion_reason(
                observations=count, expected=253, financial_ok=fiscal_ok, industry_ok=industry_ok
            )
            rows.append(row)
    detail = pd.DataFrame(rows)
    detail.to_parquet(run / "sample-exclusions.parquet", index=False)
    monthly = detail.groupby(["date", "reason"]).size().unstack(fill_value=0)
    monthly.to_parquet(run / "monthly-reasons.parquet")
    existing = pd.read_parquet(root / "state/baseline-v2-validation/coverage.parquet")
    actual = detail.assign(eligible=detail.reason.eq("ELIGIBLE")).groupby("date").eligible.sum()
    expected = existing.set_index("date").eligible
    if not actual.reindex(expected.index).eq(expected).all():
        raise ValueError("AUDIT_DOES_NOT_RECONCILE_TO_FITTED_PANEL")
    report = {
        "status": "AUDITED_NOT_RELEASED",
        "output_directory": str(run),
        "real_orders_enabled": False,
        "production_policy_updated": False,
        "months": len(monthly),
        "historical_symbols": len(symbols),
        "reason_counts": detail.reason.value_counts().to_dict(),
        "excluded_unique_symbols": int(
            detail.loc[detail.reason.ne("ELIGIBLE"), "symbol"].nunique()
        ),
        "snapshot_day_close_missing_rows": int(detail.snapshot_day_close_missing.sum()),
        "missing_adjustment_with_close_rows": int(detail.missing_adjustment_with_close.gt(0).sum()),
        "coverage_reconciled": True,
        "source_panel_sha256": hashlib.sha256(
            (root / "state/baseline-v2-validation/monthly-panel.parquet").read_bytes()
        ).hexdigest(),
        "caveats": [
            "Missing cached history is not proof of IPO or suspension.",
            "No forward filling, gate relaxation or policy retraining performed.",
        ],
    }
    write_diagnostic(run / "report.json", report)
    write_diagnostic(root / "state/baseline-v2-validation/sample-audit.json", report)
    return report
