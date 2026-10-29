"""Whole research-union adjustment evidence. No selected-stock-only validation."""

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..challenger_diagnostics import write_diagnostic
from .baseline_v2_execution_data import read_verified_cache, validate_response
from .baseline_v2_release import digest, read_json, save_stage
from .security_identity import SecurityIdentity


def union_scope(root, manifest):
    frozen = read_json(Path(manifest["replay_directory"]) / "frozen-protocol.json")
    pool_path = (
        Path(frozen["scoring"]["identity_audit"]["output_directory"])
        / "identity-top1000-preview.parquet"
    )
    sample = read_json(root / "state/baseline-v2-validation/sample-audit.json")
    sample_path = Path(sample["output_directory"]) / "sample-exclusions.parquet"
    daily = pd.read_parquet(pool_path, columns=["security_id"])
    monthly = pd.read_parquet(sample_path, columns=["symbol", "first_cached_price_date"])
    identity = SecurityIdentity(frozen["scoring"]["coverage_audit"]["identity_policy"])
    symbols = set(daily.security_id) | {identity.stable(s) for s in monthly.symbol}
    native = set(symbols)
    for row in identity.changes:
        if row["old"] in symbols:
            native.add(row["new"])
    return (
        identity,
        sorted(symbols),
        sorted(native),
        {str(p): digest(p) for p in (pool_path, sample_path)},
    )


def available_dividend_caches(root):
    selected = {}
    folders = [
        root / "state/baseline-v2-validation/execution-data",
        root / "state/baseline-v2-validation/daily-inputs/execution-extensions",
        root / "state/baseline-v2-release/source-cache",
    ]
    for folder in folders:
        for marker in sorted(folder.glob("**/dividends/*.json")):
            meta = read_json(marker)
            request = meta.get("request", {})
            if request.get("kind") != "dividends":
                continue
            frame = read_verified_cache(marker.with_suffix(".parquet"), request)
            if frame is None:
                continue
            symbol = request["symbol"]
            captured = meta.get("captured_at", "")
            if symbol not in selected or captured > selected[symbol][0]:
                selected[symbol] = (captured, marker.with_suffix(".parquet"))
    return selected


def scan_union(root, store, manifest):
    identity, symbols, native, scope_hashes = union_scope(root, manifest)
    filters = [
        ("ts_code", "in", native),
        ("trade_date", ">=", "20181201"),
        ("trade_date", "<=", "20251231"),
    ]
    # Native active-code selection does not splice inactive alias back histories.
    daily = identity.market_rows(
        store.read(
            "daily", columns=["ts_code", "trade_date", "close", "pre_close"], filters=filters
        ),
        "trade_date",
        ["close", "pre_close"],
    )
    adj = identity.market_rows(
        store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters),
        "trade_date",
        ["adj_factor"],
    )
    joined = daily.merge(
        adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one"
    ).sort_values(["ts_code", "trade_date"])
    joined["native_factor_ratio"] = (
        joined.groupby("ts_code").adj_factor.pct_change(fill_method=None) + 1
    )
    joined["previous_close"] = joined.groupby("ts_code").close.shift()
    joined["reference_ratio"] = joined.previous_close / joined.pre_close
    changes = joined.loc[joined.native_factor_ratio.sub(1).abs().gt(0.0001)].copy()
    changes["reference_agrees"] = np.isclose(
        changes.native_factor_ratio, changes.reference_ratio, rtol=0.0001, atol=0.0001
    )
    known = available_dividend_caches(root)
    covered = {identity.stable(s) for s in known}
    missing = sorted(set(native) - set(known))
    directory = Path(manifest["directory"])
    joined.to_parquet(directory / "full-union-native-prices.parquet", index=False)
    changes.to_parquet(directory / "full-union-factor-steps.parquet", index=False)
    anomalies = changes.loc[~changes.reference_agrees].copy()
    anomalies.to_parquet(directory / "full-union-reference-mismatches.parquet", index=False)
    action_audit = audit_cached_actions(changes, known, identity, directory)
    plan = {
        "start": "20181201",
        "end": "20251231",
        "required_native_symbols": native,
        "missing_full_history_dividends": missing,
        "scope_hashes": scope_hashes,
        "identity_policy": identity.changes,
        "manifest_sha256": digest(directory / "manifest.json"),
        "real_orders_enabled": False,
    }
    write_diagnostic(directory / "source-request-plan.json", plan)
    result = {
        "status": "BLOCKED_FULL_UNION_ACTION_OR_ANOMALY_EVIDENCE",
        "daily_plus_training_union": len(symbols),
        "native_symbols": len(native),
        "native_price_rows": len(joined),
        "factor_steps": len(changes),
        "reference_mismatches": len(anomalies),
        "verified_action_response_symbols": len(set(symbols) & covered),
        "missing_action_responses": len(missing),
        "native_missing_factor_rows": int(joined.adj_factor.isna().sum()),
        "cached_action_alignment": action_audit,
        "meaning": "Reference agreement is same-provider cross-check, not independent proof or full action economic validation.",
        "source_window": ["20181201", "20251231"],
        "production_changed": False,
    }
    return save_stage(
        manifest,
        "full_union_economics",
        result,
        [
            directory / "full-union-factor-steps.parquet",
            directory / "full-union-reference-mismatches.parquet",
            directory / "source-request-plan.json",
            directory / "full-union-action-alignment.parquet",
            directory / "full-union-native-prices.parquet",
        ],
    )


def audit_cached_actions(changes, caches, identity, directory):
    """Classify factor steps using cached implemented events, never repair a factor.

    Missing events, differentiated distributions and unhandled rights offerings
    remain unresolved; same-provider numerical agreement is not independent proof.
    """
    events = []
    for symbol, (_, path) in caches.items():
        frame = pd.read_parquet(path)
        frame = frame.loc[frame.div_proc.eq("实施") & frame.ex_date.notna()].copy()
        frame["ts_code"] = identity.stable(symbol)
        events.append(frame[["ts_code", "ex_date", "cash_div_tax", "stk_div"]])
    action = (
        pd.concat(events, ignore_index=True).drop_duplicates()
        if events
        else pd.DataFrame(columns=["ts_code", "ex_date", "cash_div_tax", "stk_div"])
    )
    action["ex_date"] = action.ex_date.astype(str)
    # Preserve every conflicting class as ambiguous, never arbitrarily pick one.
    grouped = action.groupby(["ts_code", "ex_date"], sort=False)
    counts = grouped.size().rename("distinct_action_classes")
    representative = grouped[["cash_div_tax", "stk_div"]].first().join(counts).reset_index()
    audit = changes.merge(
        representative,
        how="left",
        left_on=["ts_code", "trade_date"],
        right_on=["ts_code", "ex_date"],
        validate="many_to_one",
    )
    cash = pd.to_numeric(audit.cash_div_tax, errors="coerce")
    stock = pd.to_numeric(audit.stk_div, errors="coerce")
    audit["simple_action_reference"] = (audit.previous_close - cash) / (1 + stock)
    valid = (
        audit.distinct_action_classes.eq(1)
        & cash.ge(0)
        & stock.ge(0)
        & audit.simple_action_reference.gt(0)
    )
    agrees = valid & audit.simple_action_reference.sub(audit.pre_close).abs().le(0.00500001)
    audit["action_alignment"] = "NO_IMPLEMENTED_EVENT_IN_CACHE_UNRESOLVED"
    audit.loc[audit.distinct_action_classes.gt(1), "action_alignment"] = (
        "DIFFERENTIATED_ACTION_CLASSES_REVIEW_REQUIRED"
    )
    audit.loc[valid & ~agrees, "action_alignment"] = (
        "SIMPLE_ACTION_REFERENCE_DISAGREES_REVIEW_REQUIRED"
    )
    audit.loc[agrees, "action_alignment"] = "SIMPLE_ACTION_REFERENCE_ALIGNS_SAME_PROVIDER_ONLY"
    path = directory / "full-union-action-alignment.parquet"
    audit.to_parquet(path, index=False)
    return {
        "factor_step_classification": {
            str(k): int(v) for k, v in audit.action_alignment.value_counts().items()
        },
        "raw_source_unchanged": True,
        "independent_economic_validation_passed": False,
    }


def sync_union_dividends(root, provider, manifest, *, max_requests=32):
    """One bounded serial batch, cache every response, no retries on provider failure."""
    plan = read_json(Path(manifest["directory"]) / "source-request-plan.json")
    known = available_dividend_caches(root)
    pending = sorted(set(plan["required_native_symbols"]) - set(known))
    directory = root / "state/baseline-v2-release/source-cache"
    directory.mkdir(parents=True, exist_ok=True)
    cooldown = directory / "cooldown.json"
    if cooldown.exists() and read_json(cooldown)["retry_after"] > time.time():
        return {"status": "PROVIDER_COOLDOWN", "requests": 0}
    result = {
        "status": "BOUNDED_BATCH_COMPLETE",
        "requests": 0,
        "pending_before": len(pending),
        "real_orders_enabled": False,
    }
    import sqlite3

    with sqlite3.connect(directory / "sync-lock.sqlite3", timeout=0) as lock:
        lock.execute("BEGIN IMMEDIATE")
        for symbol in pending[: max(0, min(32, max_requests))]:
            time.sleep(1.3)
            request = {
                "kind": "dividends",
                "symbol": symbol,
                "start": plan["start"],
                "end": plan["end"],
            }
            try:
                frame = validate_response(
                    provider.dividends(symbol), "dividends", symbol, plan["start"], plan["end"]
                )
            except Exception as error:  # noqa: BLE001 - no provider text may leak keys
                result.update(status="STOPPED_PROVIDER_ERROR", error_type=type(error).__name__)
                write_diagnostic(
                    cooldown,
                    {"retry_after": time.time() + 1800, "error_type": type(error).__name__},
                )
                break
            result["requests"] += 1
            path = directory / "dividends" / (symbol + ".parquet")
            path.parent.mkdir(exist_ok=True)
            frame.to_parquet(path, index=False)
            write_diagnostic(
                path.with_suffix(".json"),
                {
                    "request": request,
                    "rows": len(frame),
                    "sha256": digest(path),
                    "captured_at": datetime.now(UTC).isoformat(),
                },
            )
    result["pending_after"] = len(pending) - result["requests"]
    write_diagnostic(directory / "progress.json", result)
    return result
