"""Evidence-bound V2 release workbench. Never mutates live strategies or accounts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from ..challenger_diagnostics import write_diagnostic
from ..config import BacktestConfig
from ..costs import execution_price, transaction_fee
from ..models import Side
from .baseline_v2_industry import industry_as_of
from .baseline_v2_weights import block_lower_bound
from .replay_attribution import stock_wealth_change
from .security_identity import SecurityIdentity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(manifest):
    for path, checksum in manifest["input_hashes"].items():
        if digest(path) != checksum:
            raise ValueError("RELEASE_INPUT_HASH_MISMATCH:" + Path(path).name)


def release_decision(protocol, reports, *, evidence_valid=True):
    """Unknown, stale, absent or partial reports are failures, not implicit approvals."""
    if protocol.get("real_orders_enabled") is not False:
        raise ValueError("REAL_TRADING_NOT_IN_RELEASE_SCOPE")
    blocked = [
        name
        for name in protocol["required_stages"]
        if reports.get(name, {}).get("status") != "PASS"
    ]
    forward = reports.get("forward_observation", {})
    if (
        "forward_observation" in protocol["required_stages"]
        and "forward_observation" not in blocked
        and (
            forward.get("mature_prospective_days", 0) < protocol["forward_review_days"]
            or forward.get("backfilled_days_counted") is not False
        )
    ):
        blocked.append("forward_observation")
    if not evidence_valid:
        blocked.append("evidence_integrity")
    # Activation creates the observations; requiring those same observations
    # before activation would make an independent candidate impossible to start.
    # All historical/data/execution gates remain mandatory, not merely a subset.
    observation_required = [s for s in protocol["required_stages"] if s != "forward_observation"]
    if "observation_required_stages" in protocol and set(
        protocol["observation_required_stages"]
    ) != set(observation_required):
        raise ValueError("OBSERVATION_MUST_RETAIN_ALL_NON_FORWARD_HARD_GATES")
    observation_blocked = [
        s for s in observation_required if reports.get(s, {}).get("status") != "PASS"
    ]
    if not evidence_valid:
        observation_blocked.append("evidence_integrity")
    return {
        "status": "ELIGIBLE_FOR_INDEPENDENT_PAPER_RELEASE"
        if not blocked
        else "ELIGIBLE_FOR_PAPER_OBSERVATION_NOT_PROMOTION"
        if not observation_blocked
        else "BLOCKED_NOT_FINAL_V2",
        "blocked_stages": blocked,
        "observation_blocked_stages": observation_blocked,
        "observation_activation_authorized": not observation_blocked,
        "promotion_authorized": not blocked,
        "activation_is_not_account_creation_or_order_submission": True,
        # Legacy consumers retain the stricter promotion meaning.
        "paper_release_authorized": not blocked,
        "real_orders_enabled": False,
        "replace_original_baseline": False,
    }


def freeze_release(root):
    base = root / "state/baseline-v2-validation/daily-inputs"
    report_path, audit_path = base / "daily-replay.json", base / "daily-replay-audit.json"
    report, audit = read_json(report_path), read_json(audit_path)
    if not report["two_year_window_complete"] or audit["source_report_sha256"] != digest(
        report_path
    ):
        raise ValueError("COMPLETE_CURRENT_DAILY_REPLAY_AUDIT_REQUIRED")
    source = Path(report["output_directory"])
    frozen = read_json(source / "frozen-protocol.json")
    paths = [
        report_path,
        audit_path,
        source / "frozen-protocol.json",
        source / "execution-bars.parquet",
        source / "execution-actions.parquet",
        root / "configs/baseline_v2_release.yaml",
        *[
            source / f"{v}-{s}.parquet"
            for v in ("fixed", "learned")
            for s in ("equity", "fills", "reconstructed-holdings")
        ],
        *[source / f"{v}.sqlite3" for v in ("fixed", "learned")],
        *map(Path, frozen["input_hashes"]),
    ]
    paths += list((root / "src/moneymore/research").glob("baseline_v2*.py"))
    paths += [
        root / "src/moneymore/portfolio_constructor.py",
        root / "src/moneymore/execution_replay.py",
        root / "src/moneymore/research/integer_execution.py",
        root / "src/moneymore/research/replay_actions.py",
    ]
    directory = (
        root
        / "state/baseline-v2-release/runs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    directory.mkdir(parents=True)
    manifest = {
        "version": "baseline_v2_release_manifest_v1",
        "directory": str(directory),
        "replay_directory": str(source),
        "frozen_at": datetime.now(UTC).isoformat(),
        "input_hashes": {str(p): digest(p) for p in sorted(set(paths))},
        "protocol": yaml.safe_load(
            (root / "configs/baseline_v2_release.yaml").read_text(encoding="utf-8")
        ),
        "research_only": True,
        "real_orders_enabled": False,
    }
    verify_manifest(manifest)
    write_diagnostic(directory / "manifest.json", manifest)
    write_diagnostic(
        root / "state/baseline-v2-release/latest.json",
        {"directory": str(directory), "manifest_sha256": digest(directory / "manifest.json")},
    )
    return manifest


def save_stage(manifest, name, result, artifacts=()):
    directory = Path(manifest["directory"])
    result = {
        **result,
        "manifest_sha256": digest(directory / "manifest.json"),
        "artifact_hashes": {str(p): digest(p) for p in artifacts},
    }
    write_diagnostic(directory / (name + ".json"), result)
    return result


def actual_stock_attribution(
    equity, fills, holdings, corporate_ledger, rights, actions, bars, industry
):
    """Exact economic PnL. Industry tags are contemporaneous, never today's labels."""
    marks = bars.set_index(["date", "symbol"])["close"]
    hold_groups = {str(d): f for d, f in holdings.groupby("trade_date")}
    action_index = actions.set_index("action_key")
    previous_values, previous_equity = {}, float(equity.equity.iloc[0])
    rows, checks = [], []
    for day in equity.itertuples():
        date, key = pd.Timestamp(day.trade_date), str(day.trade_date)
        held = hold_groups.get(key, holdings.iloc[:0])
        for position in held.itertuples():
            if not np.isclose(
                position.economic_quantity * marks.loc[(date, position.symbol)],
                position.market_value,
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError("RECONSTRUCTED_HOLDING_MARK_MISMATCH")
        values = dict(zip(held.symbol, held.market_value, strict=True))
        for right in rights.itertuples():
            action = action_index.loc[right.action_key]
            if right.cash_per_share > 0 and action.ex_date <= date < pd.Timestamp(
                right.cash_pay_date
            ):
                values[right.symbol] = (
                    values.get(right.symbol, 0) + right.entitled_quantity * right.cash_per_share
                )
        trades = fills.loc[fills.trade_date.astype(str).eq(key)]
        ledger = corporate_ledger.loc[corporate_ledger.trade_date.astype(str).eq(key)]
        pnl = stock_wealth_change(previous_values, values, trades, ledger)
        expected = float(day.equity) - previous_equity
        error = sum(pnl.values()) - expected
        if not np.isfinite(error) or abs(error) > 0.01:
            raise ValueError("STOCK_PNL_DOES_NOT_RECONCILE_TO_ACCOUNT")
        labels = industry_as_of(industry, date)
        for symbol in sorted(pnl):
            fee = float(trades.loc[trades.symbol.eq(symbol), "fee"].sum())
            rows.append(
                {
                    "trade_date": key,
                    "symbol": symbol,
                    "net_pnl": pnl[symbol],
                    "explicit_fee": fee,
                    "pnl_before_explicit_fees": pnl[symbol] + fee,
                    "industry": labels.l1_name.get(symbol, "UNKNOWN_NOT_FILLED"),
                    "board": "GROWTH_STAR"
                    if symbol.startswith(("300", "301", "302", "688", "689"))
                    else "MAIN_BOARD",
                    "end_economic_value": values.get(symbol, 0),
                }
            )
        checks.append({"trade_date": key, "pnl_error": error, "account_pnl": expected})
        previous_values, previous_equity = values, float(day.equity)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def run_attribution(root, store, manifest):
    verify_manifest(manifest)
    directory, source = Path(manifest["directory"]), Path(manifest["replay_directory"])
    frozen = read_json(source / "frozen-protocol.json")
    industry = SecurityIdentity(
        frozen["scoring"]["coverage_audit"]["identity_policy"]
    ).industry_rows(store.read("baseline_v2_industry_membership"))
    bars, actions = (
        pd.read_parquet(source / "execution-bars.parquet"),
        pd.read_parquet(source / "execution-actions.parquet"),
    )
    reports = []
    for variant in ("fixed", "learned"):
        equity = pd.read_parquet(source / f"{variant}-equity.parquet")
        fills = pd.read_parquet(source / f"{variant}-fills.parquet")
        holdings = pd.read_parquet(source / f"{variant}-reconstructed-holdings.parquet")
        with sqlite3.connect(
            (source / f"{variant}.sqlite3").as_uri() + "?mode=ro", uri=True
        ) as conn:
            ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", conn)
            rights = pd.read_sql_query("SELECT * FROM corporate_entitlements", conn)
        rows, checks = actual_stock_attribution(
            equity, fills, holdings, ledger, rights, actions, bars, industry
        )
        rows.to_parquet(directory / f"{variant}-stock-pnl.parquet", index=False)
        checks.to_parquet(directory / f"{variant}-pnl-checks.parquet", index=False)
        rows["year"] = rows.trade_date.str[:4]
        annual = []
        previous = 1_000_000.0
        for year, days in equity.groupby(equity.trade_date.astype(str).str[:4]):
            sample = rows.loc[rows.year.eq(year)]
            annual.append(
                {
                    "year": year,
                    "start_equity": previous,
                    "net_return": float(sample.net_pnl.sum() / previous),
                    "fees": float(sample.explicit_fee.sum()),
                    "unknown_industry_net_pnl": float(
                        sample.loc[sample.industry.eq("UNKNOWN_NOT_FILLED"), "net_pnl"].sum()
                    ),
                    "board_return_contributions": (
                        sample.groupby("board").net_pnl.sum() / previous
                    ).to_dict(),
                    "industry_return_contributions": (
                        sample.groupby("industry").net_pnl.sum() / previous
                    ).to_dict(),
                    "top_stock_pnl": sample.groupby("symbol")
                    .net_pnl.sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .to_dict(),
                    "bottom_stock_pnl": sample.groupby("symbol")
                    .net_pnl.sum()
                    .sort_values()
                    .head(10)
                    .to_dict(),
                }
            )
            previous = float(days.equity.iloc[-1])
        reports.append(
            {
                "variant": variant,
                "days": len(checks),
                "max_pnl_error": float(checks.pnl_error.abs().max()),
                "annual": annual,
            }
        )
    result = {
        "status": "ACCOUNTING_PASS_STYLE_ATTRIBUTION_PENDING_ECONOMIC_INPUTS",
        "runs": reports,
        "scope": "EXACT_STOCK_INDUSTRY_BOARD_ACCOUNTING_NOT_CAUSAL_STYLE_ALPHA",
        "style_gate": "WAIT_FULL_UNION_ECONOMICS",
        "production_changed": False,
    }
    return save_stage(
        manifest,
        "attribution",
        result,
        [
            directory / f"{v}-{s}.parquet"
            for v in ("fixed", "learned")
            for s in ("stock-pnl", "pnl-checks")
        ],
    )


def paired_return_robustness(fixed, learned, protocol):
    paired = fixed[["trade_date", "daily_return"]].merge(
        learned[["trade_date", "daily_return"]],
        on="trade_date",
        suffixes=("_fixed", "_learned"),
        validate="one_to_one",
    )
    if len(paired) != len(fixed) or len(paired) != len(learned):
        raise ValueError("PAIRED_RETURN_CALENDAR_MISMATCH")
    for col in ("daily_return_fixed", "daily_return_learned"):
        if not np.isfinite(paired[col]).all() or paired[col].le(-1).any():
            raise ValueError("INVALID_PAIRED_RETURNS")
    paired["month"] = paired.trade_date.astype(str).str[:6]
    monthly = paired.groupby("month")[["daily_return_fixed", "daily_return_learned"]].agg(
        lambda x: float(np.log1p(x).sum())
    )
    # Jan31 is the all-cash initial day, not an independent month of invested evidence.
    monthly = monthly.loc[monthly.abs().sum(axis=1).gt(1e-15)]
    improvement = monthly.daily_return_learned - monthly.daily_return_fixed
    lower = block_lower_bound(
        improvement, protocol["bootstrap_samples"], protocol["seed"], protocol["comparisons"]
    )
    rolling = {
        str(n): {
            "windows": int(improvement.rolling(n).sum().notna().sum()),
            "positive_fraction": float(improvement.rolling(n).sum().dropna().gt(0).mean()),
            "minimum_log_advantage": float(improvement.rolling(n).sum().min()),
        }
        for n in (3, 6, 12)
    }
    return {
        "status": "PASS"
        if len(monthly) >= protocol["minimum_test_months"] and lower is not None and lower > 0
        else "FAIL",
        "invested_months": len(monthly),
        "monthly_log_return_improvement_ci_low": lower,
        "leave_one_month_out_minimum_log_advantage": float((improvement.sum() - improvement).min()),
        "rolling": rolling,
        "scope": "CONTINUOUS_NET_RETURNS_NOT_RANK_IC_REVALIDATION",
        "bootstrap": "ORIGINAL_CIRCULAR_3_MONTH_BLOCK_MULTIPLE_COMPARISON_CORRECTION",
    }


def run_robustness(manifest):
    verify_manifest(manifest)
    source = Path(manifest["replay_directory"])
    result = paired_return_robustness(
        *[pd.read_parquet(source / f"{v}-equity.parquet") for v in ("fixed", "learned")],
        manifest["protocol"],
    )
    weights = read_json(source / "frozen-protocol.json")["scoring"]["weight_report"]
    signal_failures = [x for x in weights["gate_failures"] if "ROBUST" in x]
    result["continuous_return_status"] = result["status"]
    result["unresolved_original_signal_gates"] = signal_failures
    if signal_failures:
        result["status"] = "FAIL_SIGNAL_ROBUSTNESS_NOT_OVERRIDDEN_BY_ACCOUNT_RETURN"
    return save_stage(manifest, "robustness", result)


def run_financial_inventory(root, store, manifest):
    path = store.curated / "fina_indicator_research_v2.parquet"
    frame = pd.read_parquet(
        path,
        columns=[
            "ts_code",
            "ann_date",
            "end_date",
            "update_flag",
            "_payload_capture",
            "roe",
            "q_sales_yoy",
        ],
    )
    capture = pd.to_datetime(frame._payload_capture.str[:8], format="%Y%m%d", errors="coerce")
    ann = pd.to_datetime(frame.ann_date, format="%Y%m%d", errors="coerce")
    directory = Path(manifest["directory"])
    frame.to_parquet(directory / "financial-observed-version-snapshot.parquet", index=False)
    result = {
        "status": "BLOCKED_ORIGINAL_VINTAGES_NOT_ATTESTED",
        "rows": len(frame),
        "symbols": int(frame.ts_code.nunique()),
        "rows_captured_after_original_announcement": int(capture.gt(ann).sum()),
        "update_flag_counts": frame.update_flag.fillna("UNKNOWN").value_counts().to_dict(),
        "source_sha256": digest(path),
        "new_snapshot_sha256": digest(directory / "financial-observed-version-snapshot.parquet"),
        "snapshot_observed_at": datetime.now(UTC).isoformat(),
        "meaning": "Later capture does not prove wrong values, but ann_date/update_flag do not attest historical field-version availability.",
        "required_evidence": "Original as-published statement/indicator values with publication and revision timestamps, or equivalent independently verified point-in-time archive.",
        "future_snapshot_is_not_historical_attestation": True,
        "production_changed": False,
    }
    return save_stage(
        manifest,
        "financial_vintages",
        result,
        [path, directory / "financial-observed-version-snapshot.parquet"],
    )


def run_cost_screen(manifest):
    """A diagnostic screen only. Explicitly cannot satisfy full execution stress gate."""
    source = Path(manifest["replay_directory"])
    frozen = read_json(source / "frozen-protocol.json")["protocol"]
    base_config = {k: frozen[k] for k in BacktestConfig.model_fields}
    bars = pd.read_parquet(source / "execution-bars.parquet").set_index(["date", "symbol"])
    results = []
    for variant in ("fixed", "learned"):
        fills = pd.read_parquet(source / f"{variant}-fills.parquet")
        equity = pd.read_parquet(source / f"{variant}-equity.parquet")
        for bps, multiplier in [(5, 1), (10, 1), (10, 2)]:
            config = BacktestConfig.model_validate(
                {
                    **base_config,
                    "slippage_bps": bps,
                    "commission_rate": base_config["commission_rate"] * multiplier,
                    "minimum_commission": base_config["minimum_commission"] * multiplier,
                }
            )
            cash_delta = 0.0
            for fill in fills.itertuples():
                side = Side(fill.side)
                price = execution_price(
                    float(bars.loc[(pd.Timestamp(fill.trade_date), fill.symbol), "open"]), side, bps
                )
                sign = -1 if side == Side.BUY else 1
                cash_delta += sign * fill.quantity * (price - fill.price) - (
                    transaction_fee(price * fill.quantity, side, config) - fill.fee
                )
            results.append(
                {
                    "variant": variant,
                    "slippage_bps": bps,
                    "commission_multiplier": multiplier,
                    "extra_cost_fixed_fill_path": -cash_delta,
                    "counterfactual_ending_equity": float(equity.equity.iloc[-1] + cash_delta),
                }
            )
    result = {
        "status": "SCREEN_ONLY_FULL_ENDOGENOUS_REPLAY_REQUIRED",
        "runs": results,
        "limitations": [
            "NO_REPLANNING_OR_CASH_FEASIBILITY_CLAIM",
            "NO_DEPTH_OR_CAPACITY_CONCLUSION",
        ],
        "production_changed": False,
    }
    return save_stage(manifest, "execution_stress_screen", result)


def publish_status(manifest):
    directory = Path(manifest["directory"])
    reports = {
        name: read_json(directory / f"{name}.json")
        for name in manifest["protocol"]["required_stages"]
        if (directory / f"{name}.json").exists()
    }
    for name, report in list(reports.items()):
        try:
            if report.get("manifest_sha256") != digest(directory / "manifest.json"):
                raise ValueError("STALE_STAGE_MANIFEST")
            if any(
                digest(p) != checksum for p, checksum in report.get("artifact_hashes", {}).items()
            ):
                raise ValueError("STAGE_ARTIFACT_CHANGED")
        except (ValueError, FileNotFoundError):
            reports[name] = {"status": "INVALID_OR_STALE_STAGE_EVIDENCE"}
    try:
        verify_manifest(manifest)
        valid = True
    except (ValueError, FileNotFoundError):
        valid = False
    result = {
        **release_decision(manifest["protocol"], reports, evidence_valid=valid),
        "directory": str(directory),
        "stage_statuses": {
            name: reports.get(name, {}).get("status", "NOT_COMPLETED")
            for name in manifest["protocol"]["required_stages"]
        },
    }
    write_diagnostic(directory / "release-status.json", result)
    write_diagnostic(directory.parents[1] / "status.json", result)
    return result
