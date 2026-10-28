"""Data readiness for paired continuous replay of the frozen weight experiment.

Top-ten equal baskets deliberately match the earlier signal proxy. They are an
ablation protocol, NOT the full production V2 constructor or release evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from ..challenger_diagnostics import trading_dates, write_diagnostic
from .baseline_v2_adjustment_verification import apply_verified_normalization
from .baseline_v2_execution_data import execution_evidence
from .replay_actions import official_replay_actions


def paired_targets(panel, report):
    names = list(report["diagnostic_coefficients_not_released"])
    coefficients = np.array([report["diagnostic_coefficients_not_released"][n] for n in names])
    if not np.isfinite(coefficients).all():
        raise ValueError("NONFINITE_FROZEN_COEFFICIENTS")
    p = report["protocol"]
    test = panel.loc[
        panel.date.between(pd.Timestamp(p["test_start"]), pd.Timestamp(p["test_end"]))
    ].copy()
    # Intercept is constant across a cross section and does not affect rank.
    test["learned_score"] = test[names].to_numpy(float) @ coefficients
    rows = []
    for date, frame in test.groupby("date", sort=True):
        for variant, column in (("fixed", "score"), ("learned", "learned_score")):
            selected = frame.sort_values([column, "symbol"], ascending=[False, True]).head(10)
            if len(selected) != 10:
                raise ValueError("INSUFFICIENT_PAIRED_BASKET")
            for symbol in selected.symbol:
                rows.append(
                    {"date": date, "symbol": symbol, "target_weight": 0.1, "variant": variant}
                )
    return pd.DataFrame(rows)


def execution_preflight(root, store):
    folder = root / "state/baseline-v2-validation"
    source = folder / "weight-validation.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    evidence = root / "state/baseline-v2-validation/runs" / report["run_id"]
    panel_path = evidence / "monthly-panel.parquet"
    if hashlib.sha256(panel_path.read_bytes()).hexdigest() != report["panel_sha256"]:
        raise ValueError("FROZEN_PANEL_HASH_MISMATCH")
    targets = paired_targets(pd.read_parquet(panel_path), report)
    output = (
        folder
        / "execution-preflights"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    output.mkdir(parents=True)
    targets.to_parquet(output / "paired-monthly-targets.parquet", index=False)
    calendar = trading_dates(store, report["protocol"]["test_end"])
    # Replay includes cash-only first signal session; first possible fill is next session.
    start = targets.date.min().strftime("%Y%m%d")
    end = report["protocol"]["test_end"]
    dates = [d for d in calendar if start <= d <= end]
    symbols = sorted(targets.symbol.unique())
    filters = [("ts_code", "in", symbols), ("trade_date", ">=", start), ("trade_date", "<=", end)]
    prices = store.read(
        "daily", columns=["ts_code", "trade_date", "open", "close"], filters=filters
    )
    limits = store.read("stock_limits", filters=filters)
    scope = {
        "source_weight_run": report["run_id"],
        "start_date": start,
        "end_date": end,
        "required_data_symbols": symbols,
    }
    official, missing_requests = execution_evidence(root, scope)
    for kind, frame in official.items():
        frame.to_parquet(output / ("official-" + kind + ".parquet"), index=False)
    if not official["limits"].empty:
        limits = pd.concat([limits, official["limits"]], ignore_index=True).drop_duplicates(
            ["trade_date", "ts_code"], keep="last"
        )
    required = pd.MultiIndex.from_product(
        [dates, symbols], names=["trade_date", "ts_code"]
    ).to_frame(index=False)
    checked = required.merge(
        prices, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
    )
    checked = checked.merge(limits, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    checked["price_ok"] = (
        np.isfinite(checked.open)
        & checked.open.gt(0)
        & np.isfinite(checked.close)
        & checked.close.gt(0)
    )
    checked["limits_ok"] = (
        np.isfinite(checked.up_limit)
        & checked.up_limit.gt(0)
        & np.isfinite(checked.down_limit)
        & checked.down_limit.gt(0)
        & checked.up_limit.gt(checked.down_limit)
    )
    suspension = official["suspensions"]
    full = suspension.loc[
        suspension.suspend_type.eq("S")
        & (suspension.suspend_timing.isna() | suspension.suspend_timing.eq(""))
    ]
    proven = set(zip(full.trade_date.astype(str), full.ts_code))
    checked["verified_full_session_suspension"] = [
        (d, s) in proven for d, s in zip(checked.trade_date, checked.ts_code, strict=True)
    ]
    listing = pd.to_datetime(
        store.read("instruments").set_index("ts_code").list_date, format="%Y%m%d", errors="coerce"
    )
    checked["not_yet_listed"] = pd.to_datetime(checked.trade_date).lt(checked.ts_code.map(listing))
    exempt = checked.verified_full_session_suspension | checked.not_yet_listed
    checked.to_parquet(output / "market-data-readiness.parquet", index=False)
    dividend = store.read("dividend")
    # Presence of events is NOT evidence of complete history (nor is no event proof
    # of no dividend). A separate complete bounded source query is required.
    gaps = [
        "HISTORICAL_FEE_SCHEDULE_NOT_FROZEN",
        "PRODUCTION_CONSTRUCTOR_AND_DAILY_SIGNAL_PROTOCOL_NOT_REPLAYED",
    ]
    if any(r["kind"] == "dividends" for r in missing_requests):
        gaps.append("CORPORATE_ACTION_HISTORY_COMPLETENESS_NOT_ATTESTED")
    if any(r["kind"] == "suspensions" for r in missing_requests):
        gaps.append("SUSPENSION_QUERY_COVERAGE_INCOMPLETE")
    if any(r["kind"] == "adjustment_histories" for r in missing_requests):
        gaps.append("OFFICIAL_ADJUSTMENT_HISTORY_QUERY_COVERAGE_INCOMPLETE")
    adjustment_keys = set(
        zip(
            official["adjustment_histories"].trade_date.astype(str),
            official["adjustment_histories"].ts_code,
        )
    )
    checked["official_adjustment_ok"] = [
        (d, s) in adjustment_keys for d, s in zip(checked.trade_date, checked.ts_code, strict=True)
    ]
    adjustment_gaps = int((~checked.official_adjustment_ok & ~exempt).sum())
    if adjustment_gaps:
        gaps.append("OFFICIAL_ADJUSTMENT_VALUES_MISSING")
    checked.to_parquet(output / "market-data-readiness.parquet", index=False)
    if (~checked.price_ok & ~exempt).any():
        gaps.append("PRICE_GAPS_REQUIRE_VERIFIED_SUSPENSION_OR_BACKFILL")
    if (~checked.limits_ok & ~exempt).any():
        gaps.append("OFFICIAL_HISTORICAL_PRICE_LIMITS_MISSING")
    action_error = None
    actions = pd.DataFrame()
    if not official["dividends"].empty:
        raw = official["dividends"]
        relevant = raw.record_date.astype(str).between(start, end)
        try:
            actions = official_replay_actions(raw.loc[relevant])
            actions.to_parquet(output / "replay-actions.parquet", index=False)
        except ValueError as error:
            action_error = str(error)
            gaps.append("INVALID_CORPORATE_ACTION_TIMELINES_OR_AMOUNTS")
            errors = []
            for event in raw.loc[relevant & raw.div_proc.eq("实施")].to_dict("records"):
                try:
                    official_replay_actions(pd.DataFrame([event]))
                except ValueError as event_error:
                    errors.append(
                        {
                            "symbol": event["ts_code"],
                            "end_date": event["end_date"],
                            "record_date": event["record_date"],
                            "error": str(event_error),
                        }
                    )
            write_diagnostic(
                output / "action-errors.json", {"errors": errors, "batch_error": action_error}
            )
    factors = store.read("adj_factor", filters=filters).sort_values(["ts_code", "trade_date"])
    correction_rows = 0
    repair_dir = folder / "execution-data" / report["run_id"] / "adjustment-dates"
    repairs = []
    for path in sorted(repair_dir.glob("*.parquet")):
        marker = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if marker["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("ADJUSTMENT_REPAIR_HASH_MISMATCH")
        frame = pd.read_parquet(path)
        if not frame.trade_date.eq(path.stem).all():
            raise ValueError("ADJUSTMENT_REPAIR_DATE_MISMATCH")
        repairs.append(frame.loc[frame.ts_code.isin(symbols)])
    # A single-symbol complete series is the consistent basis. Never splice
    # cross-sectional spot checks over it (the endpoint representations can differ).
    if not official["adjustment_histories"].empty:
        repairs.append(official["adjustment_histories"])
    if repairs:
        revision = pd.concat(repairs, ignore_index=True).drop_duplicates(
            ["ts_code", "trade_date"], keep="last"
        )
        comparison = factors.merge(
            revision,
            on=["ts_code", "trade_date"],
            suffixes=("_cached", "_official"),
            validate="one_to_one",
        )
        comparison["changed"] = ~np.isclose(
            comparison.adj_factor_cached, comparison.adj_factor_official, rtol=1e-8, atol=1e-8
        )
        comparison.to_parquet(output / "adjustment-recheck-differences.parquet", index=False)
        correction_rows = int(comparison.changed.sum())
        factors = (
            pd.concat([factors, *repairs], ignore_index=True)
            .drop_duplicates(["ts_code", "trade_date"], keep="last")
            .sort_values(["ts_code", "trade_date"])
        )
    factors["adjustment_change"] = factors.groupby("ts_code").adj_factor.pct_change(
        fill_method=None
    )
    jumps = factors.loc[factors.adjustment_change.abs().gt(0.0001)].copy()
    explained = (
        set(zip(actions.symbol, actions.ex_date.dt.strftime("%Y%m%d")))
        if not actions.empty
        else set()
    )
    jumps["official_dividend_ex_date_found"] = [
        (s, d) in explained for s, d in zip(jumps.ts_code, jumps.trade_date, strict=True)
    ]
    source_unexplained = int((~jumps.official_dividend_ex_date_found).sum())
    jumps.to_parquet(output / "source-adjustment-jumps-before-normalization.parquet", index=False)
    factors, normalizations = apply_verified_normalization(
        root,
        factors,
        source_weight_run=report["run_id"],
        source_panel_sha256=report["panel_sha256"],
    )
    if normalizations:
        factors.to_parquet(output / "derived-adjustment-factors.parquet", index=False)
        write_diagnostic(
            output / "verified-factor-normalization.json",
            {
                "source_factors_unchanged": True,
                "events": normalizations,
                "warning": "Isolated evidence-backed research derivation, not a production input replacement.",
            },
        )
        factors["adjustment_change"] = factors.groupby("ts_code").adj_factor.pct_change(
            fill_method=None
        )
        jumps = factors.loc[factors.adjustment_change.abs().gt(0.0001)].copy()
        jumps["official_dividend_ex_date_found"] = [
            (s, d) in explained for s, d in zip(jumps.ts_code, jumps.trade_date, strict=True)
        ]
    jumps.to_parquet(output / "adjustment-jump-audit.parquet", index=False)
    if not jumps.official_dividend_ex_date_found.all():
        gaps.append("ADJUSTMENT_JUMPS_NOT_EXPLAINED_BY_VALIDATED_ACTIONS")
    data_gaps = [
        g
        for g in gaps
        if g
        not in {
            "HISTORICAL_FEE_SCHEDULE_NOT_FROZEN",
            "PRODUCTION_CONSTRUCTOR_AND_DAILY_SIGNAL_PROTOCOL_NOT_REPLAYED",
        }
    ]
    result = {
        "status": "BLOCKED_EXECUTION_DATA_NOT_READY"
        if data_gaps
        else "DATA_READY_REPLAY_NOT_COMPLETED",
        "data_status": "READY_FOR_ISOLATED_REPLAY_DATA" if not data_gaps else "DATA_GAPS_REMAIN",
        "data_gate_failures": data_gaps,
        "market_and_action_downloads_complete": not any(
            r["kind"] in {"limits", "dividends", "suspensions"} for r in missing_requests
        ),
        "factor_adjustment_consistency_pass": bool(jumps.official_dividend_ex_date_found.all()),
        "source_weight_run": report["run_id"],
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_panel_sha256": report["panel_sha256"],
        "output_directory": str(output),
        "protocol": "MONTHLY_TOP10_EQUAL_SIGNAL_ABLATION_NOT_PRODUCTION_V2",
        "start_date": start,
        "end_date": end,
        "sessions": len(dates),
        "symbols": len(symbols),
        "signal_months": int(targets.date.nunique()),
        "target_rows": len(targets),
        "conservative_full_union_grid": True,
        "missing_price_rows": int((~checked.price_ok).sum()),
        "missing_limit_rows": int((~checked.limits_ok).sum()),
        "known_dividend_event_symbols": len(set(symbols) & set(dividend.ts_code)),
        "official_dividend_event_symbols": int(official["dividends"].ts_code.nunique()),
        "official_query_responses_missing": len(missing_requests),
        "full_session_suspension_rows": int(checked.verified_full_session_suspension.sum()),
        "before_listing_rows": int(checked.not_yet_listed.sum()),
        "unexplained_price_gap_rows": int((~checked.price_ok & ~exempt).sum()),
        "unexplained_limit_gap_rows": int((~checked.limits_ok & ~exempt).sum()),
        "validated_action_events": len(actions),
        "unexplained_adjustment_value_gap_rows": adjustment_gaps,
        "adjustment_jump_rows": len(jumps),
        "source_unexplained_adjustment_jump_rows": source_unexplained,
        "independently_verified_vendor_steps_normalized": len(normalizations),
        "source_factors_not_overwritten": True,
        "official_adjustment_recheck_dates": len(list(repair_dir.glob("*.parquet"))),
        "official_adjustment_history_symbols": int(
            official["adjustment_histories"].ts_code.nunique()
        ),
        "official_adjustment_changed_rows": correction_rows,
        "fitted_panel_not_rebuilt_after_adjustment_recheck": bool(correction_rows),
        "unexplained_adjustment_jump_rows": int((~jumps.official_dividend_ex_date_found).sum()),
        "action_error": action_error,
        "corporate_action_coverage_basis": "ALL_HISTORY_PER_SYMBOL_BELOW_CAP_NOT_ORIGINAL_VINTAGE_ATTESTATION",
        "downloaded_evidence_source": str(folder / "execution-data" / report["run_id"]),
        "required_data_symbols": symbols,
        "gate_failures": gaps,
        "replay_executed": False,
        "real_orders_enabled": False,
        "production_policy_updated": False,
    }
    write_diagnostic(output / "report.json", result)
    write_diagnostic(folder / "execution-preflight.json", result)
    return {k: v for k, v in result.items() if k != "required_data_symbols"}
