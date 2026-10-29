"""Publish the user-authorized style-repair forward PAPER experiment only."""

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from moneymore.baseline_v2_candidate import prepare_factor_v2
from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.qlib_candidate_observer import candidate_catalog
from moneymore.research.baseline_v2_release import digest, read_json
from moneymore.research.style_v2_experiment import SCOPE


def main():
    root = Path(__file__).resolve().parents[1]
    local = datetime.now(ZoneInfo("Asia/Shanghai"))
    date = local.strftime("%Y%m%d")
    audit = read_json(root / "state/baseline-v2-style-release/latest.json")
    if audit["as_of"] != date or audit["status"] != "CURRENT_POOL_STYLE_AUDIT_NOT_PROMOTION":
        raise ValueError("TODAYS_STYLE_AUDIT_REQUIRED")
    audit_dir = Path(audit["directory"])
    weights_path = Path(audit["source_weights"])
    if digest(weights_path) != audit["source_weights_sha256"]:
        raise ValueError("STYLE_AUDIT_WEIGHT_SOURCE_CHANGED")
    weights = read_json(weights_path)
    suite = read_json(root / "state/baseline-v2-release/corrected-execution-suite.json")
    if suite["status"] != "COMPLETED_RESEARCH_NOT_PUBLISHED":
        raise ValueError("CORRECTED_EXECUTION_SUITE_NOT_COMPLETED")
    tag = date
    registry = root / "state/factor-baseline-candidates" / (tag + ".json")
    if registry.exists() or any(r["candidate_tag"] == tag for r in candidate_catalog(root)):
        raise ValueError("NEW_STYLE_V2_TAG_ALREADY_USED")
    evidence_dir = audit_dir / "publication"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    artifacts = [weights_path, audit_dir / "report.json", Path(audit["overlay_path"])]
    artifacts += list(audit_dir.glob("*.parquet"))
    evidence = {
        "scope": SCOPE,
        "version": "baseline_style_v2_" + date,
        "real_orders_enabled": False,
        "automatic_promotion": False,
        "coefficients": weights["diagnostic_coefficients_not_released"],
        "source_selection": "ORIGINAL_2023_VALIDATION_RANK_IC_ONLY_ALPHA_GRID_UNCHANGED",
        "not_selected_by_october_returns_or_board_fraction": True,
        "known_weight_research_failures": weights["gate_failures"],
        "vendor_financial_original_pit_claimed": False,
        "style_audit": audit,
        "execution_suite": suite,
        "artifact_hashes": {str(p): digest(p) for p in artifacts},
    }
    evidence_path = evidence_dir / "experiment-contract.json"
    write_diagnostic(evidence_path, evidence)
    policy = yaml.safe_load((root / "configs/baseline_v2.yaml").read_text(encoding="utf-8"))
    policy.update(
        version=evidence["version"],
        activation_date=date,
        candidate_tag=tag,
        display_name="因子基线 V2｜风格修正版（独立模拟）",
    )
    policy["experimental_linear_model"] = {
        "native_model": "qlib.contrib.model.linear.LinearModel",
        "scope": SCOPE,
        "coefficients": evidence["coefficients"],
        "intercept": 0.0,
        "evidence_path": str(evidence_path),
        "evidence_sha256": digest(evidence_path),
    }
    policy["input_overlay"] = {
        "path": audit["overlay_path"],
        "sha256": audit["overlay_sha256"],
        "identity_policy": audit["identity_policy"],
    }
    candidate = {
        "candidate_tag": tag,
        "account_id": "qlib_candidate_" + tag,
        "display_name": policy["display_name"],
        "portfolio_variant": "factor_baseline_v2",
        "strategy_family": "factor_baseline",
        "activation_date": date,
        "created_at": datetime.now(UTC).isoformat(),
        "factor_policy": policy,
        "real_orders_enabled": False,
        "promotion_eligible": False,
        "research_gate_scope": "STYLE_REPAIR_SCORING_CONTRACT_ONLY_NOT_PROMOTION",
        "known_weight_research_failures": weights["gate_failures"],
        "style_release_evidence": {
            "path": str(evidence_path),
            "sha256": digest(evidence_path),
            "scope": SCOPE,
        },
        "model_retrained": True,
        "validation_hold": False,
        "publication_state": "ACTIVE_FORWARD_PAPER_AWAITING_FIRST_OPEN",
        "publication_basis": "USER_AUTHORIZED_FOUR_STYLE_REPAIR_DEPENDENCIES_20261029",
        "style_audit": audit,
    }
    store = ParquetStore(root / "data")
    config = BacktestConfig.from_yaml(root / "configs/default.yaml")
    broker = PaperBroker(root / "state/paper_orders.sqlite3")
    # Build a real dated close plan before making this account visible to the scheduler.
    result = prepare_factor_v2(
        root=root,
        store=store,
        broker=broker,
        config=config,
        candidate=candidate,
        trade_date=date,
        signal_date=date,
    )
    plan = read_json(root / "state/qlib-portfolio-plans" / tag / (date + ".json"))
    if (
        plan["signal_date"] != date
        or plan["real_orders_enabled"]
        or not broker.reconcile(candidate["account_id"]).matched
    ):
        raise ValueError("STYLE_V2_FIRST_CLOSE_PLAN_OR_RECONCILIATION_FAILED")
    if broker.fills(candidate["account_id"]):
        raise ValueError("NEW_STYLE_V2_MUST_NOT_BACKFILL_PREACTIVATION_FILLS")
    write_diagnostic(registry, candidate)
    release = {
        "status": candidate["publication_state"],
        "candidate_tag": tag,
        "account_id": candidate["account_id"],
        "signal_date": date,
        "real_orders_enabled": False,
        "old_baseline_replaced": False,
        "registry": str(registry),
        "experiment_contract": str(evidence_path),
        "report_path": str(result.report_path),
        "selected": plan["target_weights"],
        "daily_workflow": "EXISTING_CANDIDATE_CLOSE_QUEUE_AND_OPEN_EXECUTION",
        "first_live_paper_fill_verified": False,
    }
    write_diagnostic(root / "state/baseline-v2-style-release/publication.json", release)
    print(json.dumps(release, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
