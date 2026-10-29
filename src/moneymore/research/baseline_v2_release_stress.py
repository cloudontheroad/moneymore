"""Collect endogenous stress replays, independently rechecking daily ledgers."""

from pathlib import Path

import pandas as pd

from ..config import BacktestConfig
from .baseline_v2_continuous_replay import audit_fill_rules, audit_ledger
from .baseline_v2_release import read_json, save_stage


def collect_execution_stress(root, manifest):
    base = root / "state/baseline-v2-validation/daily-inputs/stress-replays"
    reference = read_json(Path(manifest["replay_directory"]) / "frozen-protocol.json")
    required = ["slippage_5bps", "slippage_10bps", "combined_10bps_double_commission_tax20"]
    rows, missing, artifacts = [], [], []
    for scenario in required + ["platform_risk_hold", "platform_risk_reductions"]:
        path = base / scenario / "latest.json"
        if not path.exists():
            missing.append(scenario)
            continue
        report = read_json(path)
        source = Path(report["output_directory"])
        frozen_path = source / "frozen-protocol.json"
        frozen = read_json(frozen_path)
        if frozen["scoring"] != reference["scoring"]:
            raise ValueError("STRESS_SCORING_CONTRACT_NOT_SAME_AS_REFERENCE")
        if report.get("stress_scenario") != scenario or not report["two_year_window_complete"]:
            raise ValueError("STRESS_SCENARIO_OR_FULL_WINDOW_MISMATCH")
        config = BacktestConfig.model_validate(
            {k: frozen["protocol"][k] for k in BacktestConfig.model_fields}
        )
        bars_path, actions_path = (
            source / "execution-bars.parquet",
            source / "execution-actions.parquet",
        )
        bars, actions = pd.read_parquet(bars_path), pd.read_parquet(actions_path)
        artifacts += [path, frozen_path, bars_path, actions_path]
        if {r["variant"] for r in report["runs"]} != {"fixed", "learned"}:
            raise ValueError("PAIRED_STRESS_ACCOUNTS_REQUIRED")
        for run in report["runs"]:
            variant = run["variant"]
            equity_path, fills_path = (
                source / f"{variant}-equity.parquet",
                source / f"{variant}-fills.parquet",
            )
            db = source / f"{variant}.sqlite3"
            equity, fills = pd.read_parquet(equity_path), pd.read_parquet(fills_path)
            if len(equity) != 464 or equity.trade_date.astype(str).duplicated().any():
                raise ValueError("FULL_UNIQUE_STRESS_LEDGER_REQUIRED")
            proof, _, _ = audit_ledger(db, equity, bars, actions, config.initial_cash)
            rules = audit_fill_rules(fills.to_dict("records"), bars, config)
            if rules["status"] != "PASS" or not run["reconciliation"]["matched"]:
                raise ValueError("STRESS_EXECUTION_RULE_OR_RECONCILIATION_FAILURE")
            artifacts += [equity_path, fills_path, db]
            rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "days": len(equity),
                    "ledger_max_error": float(proof.equity_error.abs().max()),
                    "fill_rules": rules,
                    "performance": run["performance"],
                    "input_hold_days": run["input_hold_days"],
                    "blocked_solver_plans": run["blocked_solver_plans"],
                    "risk_blocked_days": run.get("platform_risk_blocked_days", 0),
                    "dividend_treatment": "IMMEDIATE_20_PERCENT_STRESS"
                    if scenario.startswith("combined_")
                    else "GROSS_NOT_HOLDING_PERIOD_TAX",
                    "risk_treatment": scenario
                    if scenario.startswith("platform_risk_")
                    else "NO_ADDITIONAL_DRAWDOWN_STOP",
                }
            )
    return save_stage(
        manifest,
        "execution_stress",
        {
            "status": "PARTIAL_ENDOGENOUS_STRESS_CAPACITY_AND_RUNTIME_REMAIN",
            "missing_scenarios": missing,
            "runs": rows,
            "remaining": [
                "HISTORICAL_VOLUME_PARTICIPATION_AND_DEPTH_CAPACITY",
                "VERSIONED_PAPER_RUNTIME_PARITY_AND_RECOVERY_ACCEPTANCE",
            ],
            "frozen_fill_repricing_not_used_as_execution_proof": True,
            "real_orders_enabled": False,
            "production_changed": False,
        },
        artifacts,
    )
