"""Independent proof for vendor-only adjustment discontinuities, never live edits.

An anomaly may be normalized only with unchanged official ex-reference price,
no implemented corporate action, matching independent raw OHLC, and a stable
independent backward/raw price ratio. Source factors and fitted panels remain frozen.
"""

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

from ..challenger_diagnostics import trading_dates, write_diagnostic
from .baseline_v2_execution_data import execution_evidence

PRICE_FIELDS = ["open", "high", "low", "close"]
PRICE_MATCH_ATOL = 1e-6
INDEPENDENT_FACTOR_RTOL = 1e-7


def verify_no_economic_adjustment(
    previous,
    current,
    af_raw_previous,
    af_raw_current,
    af_adjusted_previous,
    af_adjusted_current,
    *,
    has_action,
):
    """All conditions required; price-reference consistency alone is insufficient."""
    if has_action:
        return False, "IMPLEMENTED_ACTION_EXISTS"
    series = [
        previous,
        current,
        af_raw_previous,
        af_raw_current,
        af_adjusted_previous,
        af_adjusted_current,
    ]
    values = np.array([[float(row[k]) for k in PRICE_FIELDS] for row in series])
    if not (np.isfinite(values).all() and (values > 0).all()):
        return False, "INVALID_OHLC"
    pre_close = float(current["pre_close"])
    if not np.isfinite(pre_close) or not np.isclose(
        pre_close, float(previous["close"]), rtol=0, atol=1e-8
    ):
        return False, "OFFICIAL_REFERENCE_PRICE_ADJUSTED"
    if not np.allclose(values[:2], values[2:4], rtol=0, atol=PRICE_MATCH_ATOL):
        return False, "INDEPENDENT_RAW_PRICE_MISMATCH"
    ratios = values[4:6] / values[2:4]
    center = float(np.median(ratios))
    if not np.allclose(ratios, center, rtol=INDEPENDENT_FACTOR_RTOL, atol=1e-10):
        return False, "INDEPENDENT_FACTOR_NOT_CONSTANT"
    return True, "VERIFIED_NON_ECONOMIC_VENDOR_FACTOR_STEP"


def normalize_verified_steps(factors, details):
    """Remove only independently proven non-economic steps; retain real action ratios."""
    f = factors.copy().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if (
        f.duplicated(["ts_code", "trade_date"]).any()
        or details.duplicated(["symbol", "trade_date"]).any()
    ):
        raise ValueError("DUPLICATE_NORMALIZATION_KEYS")
    if not details.verified.eq(True).all():
        raise ValueError("UNVERIFIED_FACTOR_STEP")
    if not (np.isfinite(f.adj_factor) & f.adj_factor.gt(0)).all():
        raise ValueError("INVALID_NATIVE_FACTORS")
    native_ratio = f.groupby("ts_code").adj_factor.pct_change(fill_method=None) + 1
    keys = {(s, d): i for i, (s, d) in enumerate(zip(f.ts_code, f.trade_date, strict=True))}
    increments = np.ones(len(f))
    audit = []
    for row in details.to_dict("records"):
        key = (row["symbol"], row["trade_date"])
        if key not in keys:
            raise ValueError("VERIFIED_STEP_NOT_IN_FACTOR_SCOPE")
        i = keys[key]
        ratio = native_ratio.iloc[i]
        if (
            not np.isfinite(ratio)
            or not np.isclose(f.adj_factor.iloc[i], row["original_adj_factor"], rtol=0, atol=1e-12)
            or not np.isclose(ratio - 1, row["original_factor_change"], rtol=0, atol=1e-12)
        ):
            raise ValueError("VERIFIED_FACTOR_SOURCE_CHANGED")
        increments[i] = 1 / ratio
        audit.append(
            {
                **row,
                "normalization_increment": float(increments[i]),
                "replacement_economic_factor_ratio": 1.0,
            }
        )
    multipliers = pd.Series(increments).groupby(f.ts_code).cumprod()
    f["native_adj_factor"] = f.adj_factor
    f["verification_multiplier"] = multipliers
    f["adj_factor"] = f.adj_factor * multipliers
    return f, audit


def apply_verified_normalization(root, factors, *, source_weight_run, source_panel_sha256):
    """Evidence-bound opt-in derivation for this research scope; never edits inputs."""
    pointer = root / "state/baseline-v2-validation/adjustment-verification.json"
    if not pointer.exists():
        return factors, []
    report = json.loads(pointer.read_text(encoding="utf-8"))
    if report["status"] != "VERIFIED_VENDOR_ONLY_STEPS":
        return factors, []
    if (
        report["source_weight_run"] != source_weight_run
        or report["source_panel_sha256"] != source_panel_sha256
    ):
        raise ValueError("INDEPENDENT_PROOF_SCOPE_MISMATCH")
    directory = Path(report["output_directory"]).resolve()
    expected_root = (root / "state/baseline-v2-validation/adjustment-verification").resolve()
    if not directory.is_relative_to(expected_root):
        raise ValueError("INDEPENDENT_PROOF_OUTSIDE_RESEARCH_SCOPE")
    for relative, digest in report["source_hashes"].items():
        path = (directory / relative).resolve()
        if (
            not path.is_relative_to(directory)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("INDEPENDENT_PROOF_SOURCE_HASH_MISMATCH")
    detail_path = directory / "verification-details.parquet"
    if hashlib.sha256(detail_path.read_bytes()).hexdigest() != report["details_sha256"]:
        raise ValueError("INDEPENDENT_DETAIL_HASH_MISMATCH")
    source_path = Path(report["source_anomaly_file"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != report["source_anomaly_sha256"]:
        raise ValueError("INDEPENDENT_ANOMALY_SOURCE_HASH_MISMATCH")
    details = pd.read_parquet(detail_path)
    if (
        len(details) != report["anomalies"]
        or len(details) != report["verified"]
        or report["remaining"] != 0
    ):
        raise ValueError("INDEPENDENT_PROOF_COUNT_MISMATCH")
    return normalize_verified_steps(factors, details)


def _read_cache(path):
    marker = path.with_suffix(".json")
    if not path.exists() or not marker.exists():
        return None
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("INDEPENDENT_CACHE_HASH_MISMATCH")
    frame = pd.read_parquet(path)
    if len(frame) != meta["rows"]:
        raise ValueError("INDEPENDENT_CACHE_ROW_MISMATCH")
    return frame


def _save(path, frame, request):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix("." + uuid4().hex + ".tmp")
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


def verify_adjustments(root, store, tushare, alphafeed, *, max_requests=60):
    folder = root / "state/baseline-v2-validation"
    preflight = json.loads((folder / "execution-preflight.json").read_text(encoding="utf-8"))
    source_file = (
        Path(preflight["output_directory"]) / "source-adjustment-jumps-before-normalization.parquet"
    )
    if not source_file.exists():
        source_file = Path(preflight["output_directory"]) / "adjustment-jump-audit.parquet"
    jumps = pd.read_parquet(source_file)
    issues = jumps.loc[~jumps.official_dividend_ex_date_found].copy()
    if issues.empty:
        raise ValueError("NO_UNEXPLAINED_SOURCE_ANOMALIES")
    signature = hashlib.sha256(
        (
            "END_EXCLUSIVE_MILLISECONDS_V2:"
            + issues[["ts_code", "trade_date", "adj_factor"]].to_json()
        ).encode()
    ).hexdigest()
    directory = folder / "adjustment-verification" / signature[:20]
    directory.mkdir(parents=True, exist_ok=True)
    cooldown = directory / "cooldown.json"
    if (
        cooldown.exists()
        and json.loads(cooldown.read_text(encoding="utf-8")).get("retry_after_epoch", 0)
        > time.time()
    ):
        return {"status": "INDEPENDENT_SOURCE_COOLDOWN", "requests": 0}
    calendar = trading_dates(store, preflight["end_date"])
    previous_dates = {d: calendar[calendar.index(d) - 1] for d in issues.trade_date.unique()}
    dates = sorted(set(issues.trade_date) | set(previous_dates.values()))
    symbols = sorted(issues.ts_code.unique())
    start = dates[0]
    end = dates[-1]
    start_ms = int(pd.Timestamp(start, tz="Asia/Shanghai").timestamp() * 1000)
    end_ms = (
        int((pd.Timestamp(end, tz="Asia/Shanghai") + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    )
    plan = [("tushare-daily", d) for d in dates] + [
        (adjust, symbol) for symbol in symbols for adjust in ("none", "backward")
    ]
    requests, completed = 0, 0
    with sqlite3.connect(directory / "sync-lock.sqlite3", timeout=0) as lock:
        lock.execute("BEGIN IMMEDIATE")
        last = 0.0
        for kind, key in plan:
            path = directory / kind / (key + ".parquet")
            if _read_cache(path) is not None:
                completed += 1
                continue
            if requests >= min(100, max_requests):
                return {
                    "status": "INDEPENDENT_REQUEST_BUDGET_EXHAUSTED",
                    "requests": requests,
                    "completed": completed,
                    "required": len(plan),
                }
            if (delay := 1.3 - (time.monotonic() - last)) > 0:
                time.sleep(delay)
            last = time.monotonic()
            requests += 1
            try:
                if kind == "tushare-daily":
                    f = tushare.daily_bars(key)
                    required = {"ts_code", "trade_date", "pre_close", *PRICE_FIELDS}
                    if f is None or f.empty or len(f) >= 6000 or not required.issubset(f.columns):
                        raise ValueError("INVALID_DAILY_SCHEMA_OR_CAP")
                    if not f.trade_date.astype(str).eq(key).all() or f.ts_code.duplicated().any():
                        raise ValueError("DAILY_FILTER_NOT_HONORED")
                    request = {"source": "tushare", "trade_date": key}
                else:
                    f = alphafeed.klines.get(
                        key,
                        period="1d",
                        adjust=kind,
                        count=30,
                        start_time=start_ms,
                        end_time=end_ms,
                        to_dataframe=True,
                    )
                    if (
                        f is None
                        or f.empty
                        or len(f) >= 30
                        or not {"symbol", "trade_date", *PRICE_FIELDS}.issubset(f.columns)
                    ):
                        raise ValueError("INVALID_INDEPENDENT_SCHEMA_OR_CAP")
                    f = f.copy()
                    f["trade_date"] = pd.to_datetime(f.trade_date).dt.strftime("%Y%m%d")
                    if (
                        not f.symbol.eq(key).all()
                        or f.trade_date.duplicated().any()
                        or not f.trade_date.between(start, end).all()
                    ):
                        raise ValueError("INDEPENDENT_FILTER_NOT_HONORED")
                    request = {
                        "source": "alphafeed",
                        "symbol": key,
                        "adjust": kind,
                        "start": start,
                        "end": end,
                    }
                _save(path, f, request)
                completed += 1
            except Exception as error:  # noqa: BLE001 - persist cooldown and never expose credentials
                status = {
                    "status": "STOPPED_INDEPENDENT_SOURCE_ERROR",
                    "error_type": type(error).__name__,
                    "requests": requests,
                    "retry_after_epoch": time.time() + 1800,
                }
                write_diagnostic(cooldown, status)
                return status
            if requests % 10 == 0:
                print(
                    json.dumps(
                        {
                            "status": "INDEPENDENT_PROGRESS",
                            "completed": completed,
                            "required": len(plan),
                        }
                    ),
                    flush=True,
                )
    official, missing = execution_evidence(root, preflight)
    if missing:
        raise ValueError("OFFICIAL_EXECUTION_SOURCE_INCOMPLETE")
    dividends = official["dividends"]
    action_keys = set(
        zip(
            dividends.loc[dividends.div_proc.eq("实施"), "ts_code"],
            dividends.loc[dividends.div_proc.eq("实施"), "ex_date"].astype(str),
        )
    )
    daily = {
        d: _read_cache(directory / "tushare-daily" / (d + ".parquet")).set_index("ts_code")
        for d in dates
    }
    outcomes = []
    for event in issues.to_dict("records"):
        symbol, date = event["ts_code"], event["trade_date"]
        previous = previous_dates[date]
        raw = _read_cache(directory / "none" / (symbol + ".parquet")).set_index("trade_date")
        adjusted = _read_cache(directory / "backward" / (symbol + ".parquet")).set_index(
            "trade_date"
        )
        try:
            valid, reason = verify_no_economic_adjustment(
                daily[previous].loc[symbol],
                daily[date].loc[symbol],
                raw.loc[previous],
                raw.loc[date],
                adjusted.loc[previous],
                adjusted.loc[date],
                has_action=(symbol, date) in action_keys,
            )
        except KeyError:
            valid, reason = False, "MISSING_REQUIRED_DATE_OR_SYMBOL"
        outcomes.append(
            {
                "symbol": symbol,
                "trade_date": date,
                "original_adj_factor": event["adj_factor"],
                "original_factor_change": event["adjustment_change"],
                "verified": bool(valid),
                "reason": reason,
            }
        )
    details = pd.DataFrame(outcomes)
    details_path = directory / "verification-details.parquet"
    if details_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(details_path), details)
    else:
        details.to_parquet(details_path, index=False)
    source_hashes = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob("*/*.parquet")
    }
    result = {
        "status": "VERIFIED_VENDOR_ONLY_STEPS"
        if details.verified.all()
        else "BLOCKED_INDEPENDENT_MISMATCH",
        "source_weight_run": preflight["source_weight_run"],
        "source_panel_sha256": preflight["source_panel_sha256"],
        "source_anomaly_file": str(source_file),
        "source_anomaly_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "output_directory": str(directory),
        "anomalies": len(details),
        "symbols": len(symbols),
        "verified": int(details.verified.sum()),
        "remaining": int((~details.verified).sum()),
        "price_match_atol": PRICE_MATCH_ATOL,
        "independent_factor_rtol": INDEPENDENT_FACTOR_RTOL,
        "source_hashes": source_hashes,
        "details_sha256": hashlib.sha256(
            (directory / "verification-details.parquet").read_bytes()
        ).hexdigest(),
        "requests": requests,
        "completed_queries": completed,
        "source_factors_unchanged": True,
        "production_policy_updated": False,
        "real_orders_enabled": False,
        "warning": "Confirms no economic adjustment at these steps, not the vendor's internal root cause.",
    }
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(folder / "adjustment-verification.json", result)
    return {k: v for k, v in result.items() if k != "source_hashes"}
