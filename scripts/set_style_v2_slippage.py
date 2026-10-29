"""Prospective new-V2-only cost change; preserve old evidence and all fills."""

import argparse
import json
import shutil
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from moneymore.baseline_v2_candidate import prepare_factor_v2
from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.research.style_v2_experiment import candidate_execution_config, validate_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bps", type=float, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    local = datetime.now(ZoneInfo("Asia/Shanghai"))
    date = local.strftime("%Y%m%d")
    if time(9, 15) <= local.time() < time(15, 10):
        raise ValueError("COST_CHANGE_REQUIRES_OUTSIDE_EXECUTION_SESSION")
    registry = root / "state/factor-baseline-candidates/20261029.json"
    candidate = json.loads(registry.read_text(encoding="utf-8"))
    if candidate["account_id"] != "qlib_candidate_20261029":
        raise ValueError("UNEXPECTED_COST_CHANGE_ACCOUNT")
    validate_experiment(root, candidate["factor_policy"])
    broker = PaperBroker(root / "state/paper_orders.sqlite3")
    if broker.fills(candidate["account_id"]) or any(o["account_id"] == candidate["account_id"] and o["status"] == "PENDING" for o in broker.orders()):
        raise ValueError("PRE_FIRST_FILL_MIGRATION_ONLY")
    source_policy = candidate["factor_policy"]
    candidate["execution_policy"] = {
        "version": "style_v2_slippage_10bps_v1_20261029",
        "slippage_bps": args.bps, "real_orders_enabled": False,
        "effective_signal_date": date,
        "changed_at": datetime.now(UTC).isoformat(),
        "basis": "USER_AUTHORIZED_CONSERVATIVE_COST_MODEL_NOT_MEASURED_SLIPPAGE",
        "commission_and_taxes_separate": True,
    }
    config = BacktestConfig.from_yaml(root / "configs/default.yaml")
    effective = candidate_execution_config(config, candidate)
    archive = root / "state/baseline-v2-style-release/execution-policy-changes" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive.mkdir(parents=True, exist_ok=False)
    paths = {"registry.json": registry, "plan.json": root / "state/qlib-portfolio-plans/20261029" / (date + ".json"), "report.json": root / "state/qlib-candidate-shadow/20261029" / (date + ".json"), "publication.json": root / "state/baseline-v2-style-release/publication.json"}
    for name, path in paths.items():
        if path.exists():
            shutil.copy2(path, archive / name)
    result = prepare_factor_v2(root=root, store=ParquetStore(root / "data"), broker=broker,
                              config=config, candidate=candidate, trade_date=date, signal_date=date)
    if source_policy != candidate["factor_policy"]:
        raise ValueError("COST_CHANGE_MUST_NOT_CHANGE_MODEL")
    write_diagnostic(registry, candidate)
    publication = json.loads(paths["publication.json"].read_text(encoding="utf-8"))
    publication.update(execution_slippage_bps=effective.slippage_bps, execution_policy_version=candidate["execution_policy"]["version"])
    write_diagnostic(paths["publication.json"], publication)
    outcome = {"status": "NEW_V2_COST_POLICY_UPDATED", "account_id": candidate["account_id"], "new_slippage_bps": effective.slippage_bps, "shared_default_slippage_bps": config.slippage_bps, "old_evidence_archive": str(archive), "model_changed": False, "historical_fills_changed": False, "real_orders_enabled": False, "report_path": str(result.report_path)}
    write_diagnostic(archive / "change.json", outcome)
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    main()
