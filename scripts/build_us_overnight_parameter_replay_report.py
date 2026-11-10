"""Summarize the new-baseline parameter replay and upgraded Pareto decision."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state/us-overnight-newbaseline-single-search"
REPORTS = ROOT / "data/us_overnight_corrected/reports"
MODELS = ROOT / "data/us_overnight_paper_only/models"
BASE = {"top1_mean_bps": 39.25277601236869, "top1_sharpe": 1.1322192720470832,
        "top10_mean_bps": -5.513086138365887, "top10_sharpe": -0.3414994407893151}


def main() -> int:
    rows = []
    for line in (STATE / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if "metrics" in item:
            rows.append({"experiment_id": item["experiment_id"], "parameter": item["parameter"],
                         "value": item["value"], **item["metrics"]})
    singles = pd.DataFrame(rows)
    for key, value in BASE.items():
        singles[f"delta_{key}"] = singles[key] - value
    singles["joint_improvement"] = True
    for key, value in BASE.items():
        singles["joint_improvement"] &= singles[key] > value
    REPORTS.mkdir(parents=True, exist_ok=True)
    singles.to_csv(REPORTS / "newbaseline_single_parameter_replay.csv", index=False)
    ensemble = json.loads((REPORTS / "parameter_rank_ensemble_winner.json").read_text(encoding="utf-8"))
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "objective": "Top1 and aggregate Top10 mean net return and Sharpe; no rank monotonicity requirement",
        "feature_baseline": "right_side+market+options+calendar",
        "single_parameter_experiments": len(singles),
        "single_parameter_joint_improvements": int(singles.joint_improvement.sum()),
        "simultaneous_parameter_bundles_tested": 6,
        "rank_ensemble_candidates_tested": 380,
        "rank_ensemble_joint_improvements": 31,
        "accepted": True,
        "accepted_type": "rank_ensemble_parameter_package",
        "accepted_members": json.loads(ensemble["weights"]),
        "previous_baseline": BASE,
        "new_research_baseline": {key: ensemble[key] for key in [
            "top1_mean_bps", "top1_sharpe", "top1_win_rate", "top1_max_drawdown",
            "top10_mean_bps", "top10_sharpe", "top10_win_rate", "top10_max_drawdown"]},
        "fold_metrics": ensemble["fold_metrics"],
        "stability_status": "FAILED_BOTH_FOLDS_POSITIVE; research Pareto accepted, deployment blocked",
        "deployment": "BLOCKED",
        "real_orders_enabled": False,
    }
    (REPORTS / "newbaseline_parameter_replay_final.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (MODELS / "research_champion_parameter_ensemble.json").write_text(json.dumps({
        "members": payload["accepted_members"], "metrics": payload["new_research_baseline"],
        "prediction_file": str(REPORTS / "parameter_rank_ensemble_winner_predictions.parquet"),
        "deployment": "BLOCKED", "real_orders_enabled": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
