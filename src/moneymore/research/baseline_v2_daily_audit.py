"""Historical daily-universe coverage audit; no eligibility relaxation or trading."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from ..challenger_diagnostics import trading_dates, write_diagnostic
from ..data.fundamental_research import research_financial_frame
from .baseline_v2_daily_readiness import daily_schedule, validate_snapshot
from .baseline_v2_financials import common_period_features
from .baseline_v2_industry import industry_as_of
from .baseline_v2_sample_audit import classify_price_history


def coverage_rows(top, signal, source, counts, missing_close, missing_adj, financial, industry):
    """One daily cross-section, aligned by symbol rather than incidental row order."""
    out = top[["ts_code"]].rename(columns={"ts_code": "symbol"}).copy()
    out["date"], out["source_date"] = signal, source
    for field, series in [
        ("valid_price_sessions", counts),
        ("missing_close_sessions", missing_close),
        ("missing_adjustment_with_close", missing_adj),
    ]:
        out[field] = out.symbol.map(series).fillna(0).astype(int)
    out["financial_ok"] = out.symbol.isin(financial)
    out["industry_ok"] = out.symbol.isin(industry)
    out["reason"] = np.select(
        [
            out.valid_price_sessions.eq(0),
            out.valid_price_sessions.lt(253),
            ~out.financial_ok,
            ~out.industry_ok,
        ],
        [
            "NO_VALID_PRICE_HISTORY",
            "INCOMPLETE_PRICE_WINDOW_NOT_PROVEN_SUSPENSION_OR_IPO",
            "REQUIRED_FINANCIAL_MISSING",
            "HISTORICAL_INDUSTRY_UNKNOWN",
        ],
        default="ELIGIBLE",
    )
    return out


def audit_daily_inputs(root, store, *, identity=None):
    folder = root / "state/baseline-v2-validation/daily-inputs"
    pointer = folder / "readiness.json"
    readiness = json.loads(pointer.read_text(encoding="utf-8"))
    if readiness["missing"] or not readiness["union_complete"]:
        raise ValueError("EXACT_DAILY_SNAPSHOTS_INCOMPLETE")
    calendar = trading_dates(store, "20261028")
    expected = daily_schedule(calendar, "20240131", "20251230")
    observed = [(r["signal_date"], r["source_date"]) for r in readiness["schedule"]]
    if observed != expected or not all(r["status"] == "READY" for r in readiness["schedule"]):
        raise ValueError("DAILY_AUDIT_SCHEDULE_MISMATCH")
    industry_proof = json.loads(
        (root / "state/baseline-v2-validation/industry-download.json").read_text(encoding="utf-8")
    )
    if not industry_proof.get("complete"):
        raise ValueError("INCOMPLETE_HISTORICAL_INDUSTRY_DOWNLOAD")
    base = store.read("baseline_v2_daily_basic")
    groups = {str(d): f for d, f in base.groupby("trade_date")}
    snapshots, evidence = {}, []
    for row in readiness["schedule"]:
        source = row["source_date"]
        frame = groups.get(source)
        if frame is None:
            path = folder / (source + ".parquet")
            proof = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != proof["sha256"]:
                raise ValueError("DAILY_CACHE_HASH_MISMATCH")
            frame = pd.read_parquet(path)
        else:
            digest = hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=False).values.tobytes()
            ).hexdigest()
        evidence.append({"source_date": source, "sha256": digest})
        if identity is not None:
            frame = identity.market_rows(
                frame, "trade_date", ["close", "pb", "pe_ttm", "dv_ttm", "total_mv"]
            )
        snapshots[row["signal_date"]] = validate_snapshot(frame, source)
    symbols = sorted(set().union(*(set(f.ts_code) for f in snapshots.values())))
    source_symbols = set(symbols)
    if identity is not None:
        for row in identity.changes:
            if row["old"] in source_symbols:
                source_symbols.add(row["new"])
    start = calendar[max(0, calendar.index(min(snapshots)) - 252)]
    dates = [d for d in calendar if start <= d <= max(snapshots)]
    filters = [
        ("ts_code", "in", sorted(source_symbols)),
        ("trade_date", ">=", start),
        ("trade_date", "<=", max(snapshots)),
    ]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    if identity is not None:
        daily = identity.market_rows(daily, "trade_date", ["close"])
        adj = identity.market_rows(adj, "trade_date", ["adj_factor"])
    close = daily.pivot(index="trade_date", columns="ts_code", values="close").reindex(
        index=dates, columns=symbols
    )
    factors = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex(
        index=dates, columns=symbols
    )
    good_close = np.isfinite(close) & close.gt(0)
    good_adj = np.isfinite(factors) & factors.gt(0)
    counts = (good_close & good_adj).rolling(253, min_periods=253).sum()
    gaps = (~good_close).rolling(253, min_periods=253).sum()
    adj_gaps = (good_close & ~good_adj).rolling(253, min_periods=253).sum()
    financials = research_financial_frame(store)
    membership = store.read("baseline_v2_industry_membership")
    instruments = store.read("instruments")
    if identity is not None:
        financials = identity.market_rows(
            financials,
            "ann_date",
            ["roe", "q_sales_yoy"],
            record_keys=["ann_date", "end_date", "growth_source"],
        )
        membership = identity.industry_rows(membership)
        instruments = instruments.copy()
        instruments["ts_code"] = instruments.ts_code.map(identity.stable)
        if instruments.ts_code.duplicated().any():
            raise ValueError("DUPLICATE_IDENTITY_LISTING_METADATA")
    frames = []
    for row in readiness["schedule"]:
        signal, source = row["signal_date"], row["source_date"]
        finance = common_period_features(financials, signal)
        known = set(finance.loc[finance[["roe", "q_sales_yoy"]].notna().all(axis=1), "ts_code"])
        industries = industry_as_of(membership, signal)
        industry = set(
            industries.loc[industries.l1_name.notna() & industries.l1_name.str.strip().ne("")].index
        )
        frames.append(
            coverage_rows(
                snapshots[signal],
                signal,
                source,
                counts.loc[signal],
                gaps.loc[signal],
                adj_gaps.loc[signal],
                known,
                industry,
            )
        )
    detail = pd.concat(frames, ignore_index=True)
    detail = classify_price_history(detail, instruments, calendar)
    run = (
        folder
        / "coverage-audits"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    run.mkdir(parents=True)
    detail.to_parquet(run / "sample-exclusions.parquet", index=False)
    summary = detail.groupby(["date", "reason"]).size().unstack(fill_value=0)
    summary.to_parquet(run / "daily-reasons.parquet")
    write_diagnostic(run / "snapshot-manifest.json", {"sources": evidence})
    report = {
        "status": "DAILY_COVERAGE_AUDITED_NOT_RELEASED",
        "output_directory": str(run),
        "signals": len(summary),
        "rows": len(detail),
        "historical_symbols": len(symbols),
        "identity_policy": identity.changes if identity is not None else None,
        "identity_scope_start": start,
        "source_logical_hashes": {
            name: hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            ).hexdigest()
            for name, frame in {
                "raw_close_matrix": close,
                "source_adjustment_matrix": factors,
                "financial_records": financials,
                "industry_intervals": membership,
            }.items()
        },
        "readiness_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
        "details_sha256": hashlib.sha256(
            (run / "sample-exclusions.parquet").read_bytes()
        ).hexdigest(),
        "reason_counts": detail.reason.value_counts().to_dict(),
        "price_gap_classes": detail.price_gap_class.value_counts().to_dict(),
        "minimum_daily_eligible": int(summary.get("ELIGIBLE", pd.Series([0])).min()),
        "financial_history_absent_symbols": sorted(set(symbols) - set(financials.ts_code)),
        "real_orders_enabled": False,
        "production_changed": False,
        "limitations": [
            "CURRENT_CACHED_FINANCIAL_VINTAGES_NOT_ORIGINAL_VINTAGE_PROOF",
            "PRICE_COMPLETENESS_IS_NOT_ADJUSTMENT_ECONOMIC_VALIDATION",
            "NO_DAILY_SCORES_OR_EXECUTION_RESULTS_YET",
        ],
    }
    write_diagnostic(run / "report.json", report)
    write_diagnostic(
        folder
        / ("identity-coverage-audit.json" if identity is not None else "coverage-audit.json"),
        report,
    )
    return report
