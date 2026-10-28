"""Bounded official suspension evidence and financial-hole diagnosis, research only."""

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.baseline_v2_execution_data import (
    cache_entry,
    read_verified_cache,
    validate_response,
)
from moneymore.research.baseline_v2_financials import common_period_features, required_period
from moneymore.research.baseline_v2_gap_evidence import financial_hole_reason, full_session_dates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-requests", type=int, default=60)
    args = parser.parse_args()
    if not 0 <= args.max_requests <= 150:
        parser.error("request budget must be 0..150")
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    audit = json.loads((base / "coverage-audit.json").read_text(encoding="utf-8"))
    source = Path(audit["output_directory"]) / "sample-exclusions.parquet"
    if hashlib.sha256(source.read_bytes()).hexdigest() != audit["details_sha256"]:
        raise ValueError("SOURCE_AUDIT_HASH_MISMATCH")
    store = ParquetStore(root / "data")
    detail = pd.read_parquet(source)
    old = detail.loc[
        detail.price_gap_class.isin(
            ["OLDER_STOCK_GAP_REQUIRES_SUSPENSION_OR_BACKFILL_PROOF", "LISTING_METADATA_UNKNOWN"]
        )
    ]
    price_scope = old[["symbol", "date", "price_window_start"]].sort_values(["symbol", "date"])
    scope_hash = hashlib.sha256(
        pd.util.hash_pandas_object(price_scope, index=False).values.tobytes()
    ).hexdigest()
    folder = base / "gap-investigation" / scope_hash[:20]
    folder.mkdir(parents=True, exist_ok=True)
    calendar = trading_dates(store, "20261028")
    requests, provider, error = 0, None, None
    gaps, completed, pending = [], 0, []
    with sqlite3.connect(folder / "sync-lock.sqlite3", timeout=0) as lock:
        lock.execute("BEGIN IMMEDIATE")
        cooldown = folder / "cooldown.json"
        cooling = (
            cooldown.exists() and json.loads(cooldown.read_text())["retry_after"] > time.time()
        )
        for symbol, affected in old.groupby("symbol"):
            start = affected.price_window_start.min().strftime("%Y%m%d")
            end = affected.date.max()
            request = {"kind": "suspensions", "symbol": symbol, "start": start, "end": end}
            path = cache_entry(folder, request)
            frame = read_verified_cache(path, request)
            if frame is None:
                # Financial-only audit changes must not trigger the same price queries again.
                for prior in sorted(
                    (base / "gap-investigation").glob("*/suspensions/" + path.name)
                ):
                    frame = read_verified_cache(prior, request)
                    if frame is not None:
                        path = prior
                        break
            if (
                frame is None
                and args.download
                and not cooling
                and error is None
                and requests < args.max_requests
            ):
                try:
                    if provider is None:
                        load_dotenv(root / ".env", override=False)
                        provider = TushareProvider()
                    time.sleep(1.3)
                    requests += 1
                    frame = validate_response(
                        provider.suspensions(symbol, start, end), "suspensions", symbol, start, end
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix(".tmp")
                    frame.to_parquet(temp, index=False)
                    temp.replace(path)
                    write_diagnostic(
                        path.with_suffix(".json"),
                        {
                            "request": request,
                            "rows": len(frame),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "captured_at": datetime.now(UTC).isoformat(),
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - provider circuit breaker; no secret-bearing text
                    error = type(exc).__name__
                    write_diagnostic(
                        cooldown, {"retry_after": time.time() + 1800, "error_type": error}
                    )
                    frame = None
            if frame is None:
                pending.append(symbol)
                continue
            completed += 1
            full = full_session_dates(frame)
            filters = [
                ("ts_code", "=", symbol),
                ("trade_date", ">=", start),
                ("trade_date", "<=", end),
            ]
            daily = store.read("daily", columns=["trade_date", "close"], filters=filters)
            good = set(daily.loc[np.isfinite(daily.close) & daily.close.gt(0), "trade_date"])
            # Only days participating in an actual deficient scoring window.
            relevant = set()
            for row in affected.itertuples():
                idx = calendar.index(row.date)
                relevant.update(calendar[max(0, idx - 252) : idx + 1])
            for date in sorted(relevant - good):
                gaps.append(
                    {
                        "symbol": symbol,
                        "trade_date": date,
                        "status": "OFFICIAL_FULL_SESSION_SUSPENSION"
                        if date in full
                        else "UNEXPLAINED_PRICE_GAP",
                        "suspension_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        pd.DataFrame(
            gaps, columns=["symbol", "trade_date", "status", "suspension_source_sha256"]
        ).to_parquet(folder / "price-gaps.parquet", index=False)
        financial = research_financial_frame(store)
        failures = detail.loc[detail.reason.eq("REQUIRED_FINANCIAL_MISSING")].copy()
        failures["period"] = failures.date.map(required_period)
        rows = []
        for (symbol, period), affected in failures.groupby(["symbol", "period"]):
            at = affected.date.max()
            f = common_period_features(financial.loc[financial.ts_code.eq(symbol)], at)
            row = f.iloc[0] if len(f) else pd.Series(dtype=object)
            period_rows = financial.loc[
                financial.ts_code.eq(symbol) & financial.end_date.eq(period)
            ]
            rows.append(
                {
                    "symbol": symbol,
                    "period": period,
                    "last_affected_date": at,
                    "affected_days": len(affected),
                    "roe_missing_at_last_date": pd.isna(row.get("roe", np.nan)),
                    "growth_missing_at_last_date": pd.isna(row.get("q_sales_yoy", np.nan)),
                    "roe_hole_reason": financial_hole_reason(period_rows, at, "roe")
                    if pd.isna(row.get("roe", np.nan))
                    else "PRESENT",
                    "growth_hole_reason": financial_hole_reason(period_rows, at, "q_sales_yoy")
                    if pd.isna(row.get("q_sales_yoy", np.nan))
                    else "PRESENT",
                    "period_payload_rows": int(
                        (financial.ts_code.eq(symbol) & financial.end_date.eq(period)).sum()
                    ),
                }
            )
        pd.DataFrame(rows).to_parquet(folder / "financial-period-gaps.parquet", index=False)
        report = {
            "status": "GAP_EVIDENCE_ONLY_NO_POLICY_CHANGE",
            "output_directory": str(folder),
            "source_audit": audit["output_directory"],
            "requests": requests,
            "suspension_symbols_complete": completed,
            "suspension_symbols_pending": pending,
            "provider_error_type": error,
            "provider_cooldown": cooling,
            "unique_price_gap_days": len(gaps),
            "price_gap_reasons": pd.Series([r["status"] for r in gaps], dtype=str)
            .value_counts()
            .to_dict(),
            "financial_symbol_periods": len(rows),
            "financial_symbols": failures.symbol.nunique(),
            "eligibility_changed": False,
            "prices_filled": False,
            "real_orders_enabled": False,
        }
        write_diagnostic(folder / "report.json", report)
        write_diagnostic(base / "gap-investigation.json", report)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
