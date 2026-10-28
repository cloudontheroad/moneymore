"""Bounded, resumable official execution evidence; no production-table writes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ..challenger_diagnostics import write_diagnostic

LIMITS = {"limits": 5800, "dividends": 2000, "suspensions": 5000, "adjustment_histories": 6000}
FIELDS = {
    "adjustment_histories": ["ts_code", "trade_date", "adj_factor"],
    "limits": ["ts_code", "trade_date", "up_limit", "down_limit"],
    "dividends": [
        "ts_code",
        "end_date",
        "ann_date",
        "div_proc",
        "record_date",
        "ex_date",
        "cash_div_tax",
        "stk_div",
        "pay_date",
        "div_listdate",
        "imp_ann_date",
    ],
    "suspensions": ["ts_code", "trade_date", "suspend_type", "suspend_timing"],
}


def validate_response(frame, kind, symbol, start, end):
    if frame is None or not set(FIELDS[kind]).issubset(frame.columns):
        raise ValueError("NULL_OR_MISSING_SCHEMA")
    if len(frame) >= LIMITS[kind]:
        raise ValueError("POSSIBLE_TRUNCATION")
    f = frame.copy()
    if not f.ts_code.eq(symbol).all():
        raise ValueError("SYMBOL_FILTER_NOT_HONORED")
    if kind != "dividends":
        dates = pd.to_datetime(f.trade_date, format="%Y%m%d", errors="coerce")
        if dates.isna().any() or not f.trade_date.astype(str).between(start, end).all():
            raise ValueError("INVALID_OR_OUTSIDE_REQUEST_DATES")
        if kind == "limits":
            for col in ("up_limit", "down_limit"):
                f[col] = pd.to_numeric(f[col], errors="coerce")
            if (
                not np.isfinite(f[["up_limit", "down_limit"]]).all().all()
                or not (f.up_limit.gt(f.down_limit) & f.down_limit.gt(0)).all()
            ):
                raise ValueError("INVALID_PRICE_LIMITS")
            if f.trade_date.duplicated().any() or f.empty:
                raise ValueError("EMPTY_OR_DUPLICATE_LIMITS")
        elif kind == "adjustment_histories":
            values = pd.to_numeric(f.adj_factor, errors="coerce")
            if (
                f.empty
                or f.trade_date.duplicated().any()
                or not (np.isfinite(values) & values.gt(0)).all()
            ):
                raise ValueError("INVALID_ADJUSTMENT_HISTORY")
        elif not f.suspend_type.eq("S").all():
            raise ValueError("INVALID_SUSPENSION_TYPE")
    # Dividend query is all-history, not filtered by ann_date here: implemented
    # event timelines are validated before replay, never fed back into features.
    return f


def cache_entry(folder, request):
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    return folder / request["kind"] / (request["symbol"] + "-" + digest[:16] + ".parquet")


def read_verified_cache(path, request):
    marker = path.with_suffix(".json")
    if not path.exists() or not marker.exists():
        return None
    evidence = json.loads(marker.read_text(encoding="utf-8"))
    if (
        evidence["request"] != request
        or evidence["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("CACHE_EVIDENCE_MISMATCH")
    frame = validate_response(
        pd.read_parquet(path), request["kind"], request["symbol"], request["start"], request["end"]
    )
    if len(frame) != evidence["rows"]:
        raise ValueError("CACHE_ROW_COUNT_MISMATCH")
    return frame


def request_plan(preflight):
    return [
        {
            "kind": kind,
            "symbol": symbol,
            "start": preflight["start_date"],
            "end": preflight["end_date"],
        }
        for kind in ("limits", "dividends", "suspensions", "adjustment_histories")
        for symbol in preflight["required_data_symbols"]
    ]


def sync_execution_data(root, provider, *, max_requests=25, interval_seconds=1.3):
    folder = root / "state/baseline-v2-validation"
    preflight = json.loads((folder / "execution-preflight.json").read_text(encoding="utf-8"))
    # Scope is immutable per selected weight run and requested dates.
    directory = folder / "execution-data" / preflight["source_weight_run"]
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.json"
    old = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    if old.get("retry_after_epoch", 0) > time.time():
        return {
            "status": "PROVIDER_COOLDOWN",
            "requests": 0,
            "retry_after_epoch": old["retry_after_epoch"],
        }
    plan = request_plan(preflight)
    result = {
        "status": "RUNNING",
        "source_weight_run": preflight["source_weight_run"],
        "source_panel_sha256": preflight["source_panel_sha256"],
        "requests": 0,
        "cached": 0,
        "complete": 0,
        "required": len(plan),
        "directory": str(directory),
        "real_orders_enabled": False,
        "production_policy_updated": False,
    }
    with sqlite3.connect(directory / "sync-lock.sqlite3", timeout=0) as lock:
        lock.execute("BEGIN IMMEDIATE")
        last_start = 0.0
        for request in plan:
            path = cache_entry(directory, request)
            try:
                cached = read_verified_cache(path, request)
                if cached is not None:
                    result["cached"] += 1
                    result["complete"] += 1
                    continue
                if result["requests"] >= min(600, max(0, max_requests)):
                    result["status"] = "REQUEST_BUDGET_EXHAUSTED"
                    break
                delay = max(1.3, interval_seconds) - (time.monotonic() - last_start)
                if delay > 0:
                    time.sleep(delay)
                last_start = time.monotonic()
                result["requests"] += 1
                kind, symbol = request["kind"], request["symbol"]
                if kind == "limits":
                    frame = provider.stock_limits(symbol, request["start"], request["end"])
                elif kind == "dividends":
                    frame = provider.dividends(symbol)
                elif kind == "suspensions":
                    frame = provider.suspensions(symbol, request["start"], request["end"])
                else:
                    frame = provider.historical_adjustments(
                        symbol, request["start"], request["end"]
                    )
                frame = validate_response(frame, kind, symbol, request["start"], request["end"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix("." + uuid4().hex + ".tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
                write_diagnostic(
                    path.with_suffix(".json"),
                    {
                        "request": request,
                        "rows": len(frame),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "captured_at": datetime.now(UTC).isoformat(),
                        "response_below_endpoint_cap": True,
                    },
                )
                result["complete"] += 1
                write_diagnostic(checkpoint, result)
                if result["requests"] % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "status": "PROGRESS",
                                "requests": result["requests"],
                                "complete": result["complete"],
                                "required": len(plan),
                            }
                        ),
                        flush=True,
                    )
            except Exception as error:  # noqa: BLE001 - sanitize provider failures, stop and cool down
                # No message/traceback from provider: may include credentials.
                result.update(
                    status="STOPPED_ERROR",
                    error_type=type(error).__name__,
                    failed_request=request,
                    retry_after_epoch=time.time() + 1800,
                )
                break
        if result["complete"] == len(plan):
            result["status"] = "DOWNLOAD_COMPLETE_PENDING_AUDIT"
        write_diagnostic(checkpoint, result)
    write_diagnostic(folder / "execution-data-sync.json", result)
    return result


def execution_evidence(root, preflight):
    """Read verified immutable responses, including successful empty responses."""
    directory = (
        root / "state/baseline-v2-validation/execution-data" / preflight["source_weight_run"]
    )
    frames = {kind: [] for kind in LIMITS}
    missing = []
    for request in request_plan(preflight):
        f = read_verified_cache(cache_entry(directory, request), request)
        if f is None:
            missing.append(request)
        else:
            frames[request["kind"]].append(f)
    combined = {
        k: pd.concat(v, ignore_index=True).drop_duplicates()
        if v
        else pd.DataFrame(columns=FIELDS[k])
        for k, v in frames.items()
    }
    return combined, missing


def sync_unexplained_adjustments(root, provider, *, max_requests=5):
    """Recheck exact dates rather than raising tolerance to hide small jumps."""
    folder = root / "state/baseline-v2-validation"
    report = json.loads((folder / "execution-preflight.json").read_text(encoding="utf-8"))
    jumps = pd.read_parquet(Path(report["output_directory"]) / "adjustment-jump-audit.parquet")
    dates = sorted(jumps.loc[~jumps.official_dividend_ex_date_found, "trade_date"].unique())
    directory = folder / "execution-data" / report["source_weight_run"]
    cooldown = directory / "adjustment-cooldown.json"
    if cooldown.exists():
        state = json.loads(cooldown.read_text(encoding="utf-8"))
        if state.get("retry_after_epoch", 0) > time.time():
            return {"status": "ADJUSTMENT_PROVIDER_COOLDOWN", "requests": 0}
    result = {
        "status": "ADJUSTMENT_RECHECK_COMPLETE",
        "requests": 0,
        "dates": dates,
        "real_orders_enabled": False,
        "production_policy_updated": False,
    }
    with sqlite3.connect(directory / "sync-lock.sqlite3", timeout=0) as lock:
        lock.execute("BEGIN IMMEDIATE")
        for date in dates:
            path = directory / "adjustment-dates" / (str(date) + ".parquet")
            if path.exists() and path.with_suffix(".json").exists():
                marker = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if marker["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("ADJUSTMENT_CACHE_HASH_MISMATCH")
                continue
            if result["requests"] >= min(10, max_requests):
                result["status"] = "ADJUSTMENT_REQUEST_BUDGET_EXHAUSTED"
                break
            if result["requests"]:
                time.sleep(1.3)
            result["requests"] += 1
            try:
                f = provider.adjustment_factors(str(date))
            except Exception as error:  # noqa: BLE001 - stop, persist cooldown, sanitize provider error
                write_diagnostic(
                    cooldown,
                    {"error_type": type(error).__name__, "retry_after_epoch": time.time() + 1800},
                )
                raise RuntimeError("ADJUSTMENT_PROVIDER_STOPPED") from None
            if (
                f is None
                or len(f) >= 6000
                or f.empty
                or not {"ts_code", "trade_date", "adj_factor"}.issubset(f.columns)
                or not f.trade_date.astype(str).eq(str(date)).all()
                or f.ts_code.duplicated().any()
            ):
                raise ValueError("INVALID_OR_TRUNCATED_ADJUSTMENT_DATE_RESPONSE")
            values = pd.to_numeric(f.adj_factor, errors="coerce")
            if not (np.isfinite(values) & values.gt(0)).all():
                raise ValueError("INVALID_ADJUSTMENT_VALUES")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix("." + uuid4().hex + ".tmp")
            f.to_parquet(temp, index=False)
            temp.replace(path)
            write_diagnostic(
                path.with_suffix(".json"),
                {
                    "trade_date": str(date),
                    "rows": len(f),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "captured_at": datetime.now(UTC).isoformat(),
                },
            )
    write_diagnostic(folder / "adjustment-recheck.json", result)
    return result
