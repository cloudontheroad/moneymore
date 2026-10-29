"""Prospective paper contract and parity checks. No broker order submission."""

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ..challenger_diagnostics import write_diagnostic
from ..config import BacktestConfig
from .baseline_v2_daily_readiness import classify_held_inputs, construct_daily_targets
from .baseline_v2_release import digest, read_json, save_stage, verify_manifest


def shadow_targets(
    scores, quantities, prices, signal_date, execution_date, policy, universe_members
):
    classification = classify_held_inputs(
        quantities, scores.symbol, universe_members=universe_members
    )
    if classification["input_faults"]:
        return {
            "status": "HOLD_AND_ALERT",
            "target_weights": {},
            "clear_previous_plan": True,
            **classification,
        }
    _, weights, _ = construct_daily_targets(
        scores,
        quantities,
        prices,
        signal_date,
        execution_date,
        policy,
        universe_members=universe_members,
    )
    return {
        "status": "SHADOW_TARGET_ONLY",
        "target_weights": weights,
        "real_orders_enabled": False,
        **classification,
    }


def run_runtime_audit(root, manifest):
    verify_manifest(manifest)
    source, directory = Path(manifest["replay_directory"]), Path(manifest["directory"])
    frozen = read_json(source / "frozen-protocol.json")
    config = BacktestConfig.from_yaml(root / "configs/default.yaml")
    risks = []
    for variant in ("fixed", "learned"):
        equity = pd.read_parquet(source / f"{variant}-equity.parquet")
        drawdown = equity.equity / equity.equity.cummax() - 1
        risks.append(
            {
                "variant": variant,
                "close_warning_days": int(drawdown.le(-config.max_drawdown * 2 / 3).sum()),
                "close_limit_days": int(drawdown.le(-config.max_drawdown).sum()),
                "caveat": "Close-based audit flags non-equivalence; not a replay of open-time sticky account risk states.",
            }
        )
    contract = {
        "schema": "factor_v2_provisional_paper_contract_v1",
        "manifest_sha256": digest(directory / "manifest.json"),
        "factor_policy": frozen["scoring"]["policy"],
        "native_model": "qlib.contrib.model.linear.LinearModel",
        "coefficients": frozen["scoring"]["weight_report"]["diagnostic_coefficients_not_released"],
        "score_semantics": "ONCE_NORMALIZED_DIRECTION_ADJUSTED_FEATURES_DOT_COEFFICIENTS_NOT_EXPECTED_RETURN",
        "complete_pool_required": True,
        "prior_plan_on_fault": "CLEAR_HOLD_AND_ALERT",
        "identity_policy": frozen["scoring"]["coverage_audit"]["identity_policy"],
        "reference_execution": frozen["protocol"],
        "platform_risk": config.model_dump(),
        "source_baseline_not_replaced": True,
        "orders_enabled": False,
        "paper_activation_authorized": False,
    }
    write_diagnostic(directory / "provisional-paper-contract.json", contract)
    parity = {
        "status": "BLOCKED_RESEARCH_RUNTIME_CONTRACT_DIFFERENCES",
        "observations": risks,
        "differences": [
            "REPLAY_NO_ADDITIONAL_DRAWDOWN_STOP_RUNTIME_STICKY_RISK_STATES",
            "OLD_RUNTIME_SCORE_MISSING_GUARD_DOES_NOT_DISTINGUISH_UNIVERSE_EXIT",
            "OLD_RUNTIME_CURRENT_UNIVERSE_NOT_FULL_DATED_IDENTITY_RESEARCH_ADAPTER",
        ],
        "new_shadow_target_adapter": "REUSES_FORMAL_RESEARCH_CONSTRUCTOR_AND_EXPLICIT_EXIT_CLASSIFIER",
        "required": "New version must run the same risk/data/identity/fee contract in both replay and paper; never disable platform safeguards to match returns.",
        "production_changed": False,
    }
    save_stage(
        manifest,
        "parity",
        parity,
        [directory / "provisional-paper-contract.json", root / "configs/default.yaml"],
    )
    forward = {
        "status": "WAITING_FOR_ELIGIBLE_FROZEN_VERSION_AND_REAL_OBSERVATIONS",
        "mature_prospective_days": 0,
        "backfilled_days_counted": False,
        "observation_days": manifest["protocol"]["forward_observation_days"],
        "review_days": manifest["protocol"]["forward_review_days"],
        "orders_enabled": False,
        "contract_sha256": digest(directory / "provisional-paper-contract.json"),
        "candidate_not_registered_or_resumed": True,
    }
    save_stage(manifest, "forward_observation", forward)
    return parity


def record_prospective_observation(directory, payload, *, now=None):
    """Append once to a current-day journal; historical reconstruction never qualifies."""
    local = (now or datetime.now(UTC)).astimezone(ZoneInfo("Asia/Shanghai"))
    if payload["trade_date"] != local.strftime("%Y%m%d") or payload.get("backfilled") is not False:
        raise ValueError("ONLY_REAL_CURRENT_DAY_OBSERVATIONS_ALLOWED")
    contract = directory / "provisional-paper-contract.json"
    if payload.get("contract_sha256") != digest(contract):
        raise ValueError("OBSERVATION_CONTRACT_CHANGED")
    if not read_json(contract).get("paper_activation_authorized"):
        raise ValueError("PROVISIONAL_CONTRACT_NOT_ELIGIBLE_FOR_OBSERVATION_ACCOUNT")
    path = directory / "prospective-journal" / (payload["trade_date"] + ".json")
    if path.exists():
        existing = read_json(path)
        if existing["payload"] != payload:
            raise ValueError("CANNOT_REWRITE_FIRST_CREATED_PROSPECTIVE_OBSERVATION")
        return existing
    record = {
        "payload": payload,
        "first_created_at": local.isoformat(),
        "historical_backfill": False,
    }
    write_diagnostic(path, record)
    return record
