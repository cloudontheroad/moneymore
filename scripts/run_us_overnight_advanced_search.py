"""Resumable multi-period, one-variable, Pareto LightGBM search (paper only)."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAINER = ROOT / "scripts" / "train_us_overnight_lightgbm.py"
MODELS = ROOT / "data" / "us_overnight_corrected" / "models" / "full_search"
PANEL = ROOT / "data" / "us_overnight_corrected" / "features" / "historical_panel_61_lag1.parquet"
STATE = ROOT / "state" / "us-overnight-corrected-search"
LEDGER = STATE / "ledger.jsonl"
CHAMPION = STATE / "champion.json"
FRONTIER = STATE / "pareto_frontier.json"

BASE = {
    "num_leaves": 31,
    "min_child_samples": 300,
    "colsample_bytree": 0.75,
    "reg_lambda": 3.0,
    "reg_alpha": 0.0,
    "min_split_gain": 0.0,
    "max_depth": -1,
    "learning_rate": 0.025,
    "n_estimators": 350,
    "huber_alpha": 0.9,
    "subsample": 0.8,
    "subsample_freq": 0,
    "weight_half_life": 126,
    "rolling_train_sessions": 504,
    "target_transform": "raw",
    "min_child_weight": 0.001,
    "path_smooth": 0.0,
    "feature_fraction_bynode": 1.0,
    "extra_trees": False,
    "max_bin": 255,
    "min_data_in_bin": 3,
    "boosting_type": "gbdt",
    "data_sample_strategy": "bagging",
    "drop_rate": 0.1,
    "skip_drop": 0.5,
    "max_drop": 50,
    "top_rate": 0.2,
    "other_rate": 0.1,
    "top1_weight": 1.0,
    "profit_early_stopping_rounds": 0,
    "validation_sessions": 252,
    "retrain_every": 21,
}

PHASES = [
    ("num_leaves", [15, 21, 23, 25, 27]),
    ("min_child_samples", [250, 275, 300, 400]),
    ("colsample_bytree", [0.70, 0.75, 0.80]),
    ("reg_lambda", [2.0, 3.0, 4.0, 5.0, 2.5]),
    ("reg_alpha", [0.1, 0.5, 1.0]),
    ("min_split_gain", [0.01, 0.05, 0.1]),
    ("max_depth", [8, 6, 10]),
    ("learning_rate", [0.02, 0.03, 0.015]),
    ("n_estimators", [450, 250, 550]),
    ("huber_alpha", [0.85, 0.8, 0.95]),
    ("weight_half_life", [189, 252, 63]),
    ("rolling_train_sessions", [630, 756, 378]),
    ("subsample_freq", [1]),
    ("subsample", [0.9, 0.7, 0.85]),
    ("target_transform", ["daily_mad"]),
    ("path_smooth", [1.0, 5.0, 10.0, 20.0, 50.0]),
    ("min_child_weight", [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]),
    ("feature_fraction_bynode", [0.9, 0.8, 0.7, 0.6]),
    ("extra_trees", [True]),
    ("max_bin", [63, 127, 511, 1023]),
    ("min_data_in_bin", [10, 30, 100, 300]),
    ("profit_early_stopping_rounds", [20, 40, 80]),
    ("reg_lambda", [10.0, 30.0, 100.0, 300.0]),
    ("reg_alpha", [1.0, 3.0, 10.0, 30.0, 100.0]),
    ("data_sample_strategy", ["goss"]),
    ("boosting_type", ["dart"]),
    ("top1_weight", [1.5, 2.0, 3.0, 5.0]),
]

CLI = {
    "num_leaves": "--num-leaves",
    "min_child_samples": "--min-child-samples",
    "colsample_bytree": "--colsample-bytree",
    "reg_lambda": "--reg-lambda",
    "reg_alpha": "--reg-alpha",
    "min_split_gain": "--min-split-gain",
    "max_depth": "--max-depth",
    "learning_rate": "--learning-rate",
    "n_estimators": "--n-estimators",
    "huber_alpha": "--huber-alpha",
    "subsample": "--subsample",
    "subsample_freq": "--subsample-freq",
    "weight_half_life": "--weight-half-life",
    "rolling_train_sessions": "--rolling-train-sessions",
    "target_transform": "--target-transform",
    "min_child_weight": "--min-child-weight",
    "path_smooth": "--path-smooth",
    "feature_fraction_bynode": "--feature-fraction-bynode",
    "extra_trees": "--extra-trees",
    "max_bin": "--max-bin",
    "min_data_in_bin": "--min-data-in-bin",
    "boosting_type": "--boosting-type",
    "data_sample_strategy": "--data-sample-strategy",
    "drop_rate": "--drop-rate",
    "skip_drop": "--skip-drop",
    "max_drop": "--max-drop",
    "top_rate": "--top-rate",
    "other_rate": "--other-rate",
    "top1_weight": "--top1-weight",
    "profit_early_stopping_rounds": "--profit-early-stopping-rounds",
    "validation_sessions": "--validation-sessions",
    "retrain_every": "--retrain-every",
}


def metrics(payload: dict) -> dict[str, float | int]:
    top1 = payload["validation_rank_slot_metrics"][0]
    return {
        "violations": int(payload["validation_monotonic_violations"]),
        "top1_mean_bps": float(top1["mean_net_bps"]),
        "top1_sharpe": float(top1["sharpe"]),
        "top1_drawdown": float(top1["max_drawdown"]),
        "top1_cumulative": float(top1["cumulative_return"]),
        "top10_mean_bps": float(payload["validation_top_n_mean_net"] * 10_000),
        "top10_sharpe": float(payload["validation_top_n_sharpe"]),
        "top10_drawdown": float(payload["validation_top_n_max_drawdown"]),
        "median_fold_top1_bps": float(payload["validation_median_fold_top1_mean_bps"]),
        "worst_fold_top1_bps": float(payload["validation_worst_fold_top1_mean_bps"]),
        "positive_top1_folds": int(payload["validation_positive_top1_folds"]),
    }


def command(parameters: dict, prefix: str) -> list[str]:
    result = [
        str(PYTHON), str(TRAINER), "--output-prefix", prefix,
        "--panel-path", str(PANEL), "--model-dir", str(MODELS),
    ]
    for name, value in parameters.items():
        if isinstance(value, bool):
            if value:
                result.append(CLI[name])
        else:
            result.extend([CLI[name], str(value)])
    return result


def run(parameters: dict, prefix: str) -> dict:
    subprocess.run(
        command(parameters, prefix),
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return json.loads((MODELS / f"{prefix}_ranker.json").read_text(encoding="utf-8"))


def dominates(left: dict, right: dict) -> bool:
    maximize = (
        "top1_mean_bps",
        "top1_sharpe",
        "top1_drawdown",
        "top10_mean_bps",
        "top10_sharpe",
        "median_fold_top1_bps",
        "worst_fold_top1_bps",
        "positive_top1_folds",
    )
    no_worse = all(left[key] >= right[key] for key in maximize) and left["violations"] <= right["violations"]
    strictly_better = any(left[key] > right[key] for key in maximize) or left["violations"] < right["violations"]
    return no_worse and strictly_better


def decision(candidate: dict, incumbent: dict) -> tuple[bool, float, str]:
    if candidate["violations"] > incumbent["violations"]:
        return False, -999.0, "排序违例增加"
    if candidate["positive_top1_folds"] < incumbent["positive_top1_folds"]:
        return False, -999.0, "正收益分段减少"
    if candidate["top1_mean_bps"] < incumbent["top1_mean_bps"] - 2.0:
        return False, -999.0, "Top1均值下降超过2bp"
    if candidate["top1_sharpe"] < incumbent["top1_sharpe"] - 0.15:
        return False, -999.0, "Top1 Sharpe下降超过0.15"
    if candidate["top1_drawdown"] < incumbent["top1_drawdown"] - 0.05:
        return False, -999.0, "Top1最大回撤恶化超过5个百分点"
    if candidate["top10_mean_bps"] < incumbent["top10_mean_bps"] - 2.0:
        return False, -999.0, "Top10均值下降超过2bp"
    scales = {
        "top1_mean_bps": max(abs(incumbent["top1_mean_bps"]), 10.0),
        "top1_sharpe": max(abs(incumbent["top1_sharpe"]), 0.5),
        "top10_mean_bps": max(abs(incumbent["top10_mean_bps"]), 5.0),
        "median_fold_top1_bps": max(abs(incumbent["median_fold_top1_bps"]), 10.0),
        "worst_fold_top1_bps": max(abs(incumbent["worst_fold_top1_bps"]), 20.0),
    }
    delta = (
        0.30 * (candidate["top1_mean_bps"] - incumbent["top1_mean_bps"]) / scales["top1_mean_bps"]
        + 0.25 * (candidate["top1_sharpe"] - incumbent["top1_sharpe"]) / scales["top1_sharpe"]
        + 0.15 * (candidate["top10_mean_bps"] - incumbent["top10_mean_bps"]) / scales["top10_mean_bps"]
        + 0.15 * (candidate["median_fold_top1_bps"] - incumbent["median_fold_top1_bps"]) / scales["median_fold_top1_bps"]
        + 0.10 * (candidate["worst_fold_top1_bps"] - incumbent["worst_fold_top1_bps"]) / scales["worst_fold_top1_bps"]
        + 0.05 * (candidate["top1_drawdown"] - incumbent["top1_drawdown"]) / 0.10
        + 0.25 * (incumbent["violations"] - candidate["violations"])
    )
    return delta > 0.05, float(delta), "综合收益改善" if delta > 0.05 else "综合改善不足"


def append(record: dict) -> None:
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def promote(prefix: str) -> None:
    for suffix in ("ranker.joblib", "ranker.json", "feature_importance.csv", "walk_forward_daily.csv", "rank_slot_metrics.csv"):
        shutil.copy2(MODELS / f"{prefix}_{suffix}", MODELS / f"lightgbm_{suffix}")


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    completed = {}
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[row["experiment_id"]] = row

    if "baseline" not in completed:
        payload = run(BASE, "advanced_baseline")
        row = {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "experiment_id": "baseline",
            "phase": "baseline",
            "value": "current_champion",
            "parameters": BASE,
            "metrics": metrics(payload),
            "adopted": True,
            "reason": "多分段协议基线",
            "score_delta": 0.0,
            "real_orders_enabled": False,
        }
        append(row)
        completed["baseline"] = row

    champion_parameters = completed["baseline"]["parameters"].copy()
    champion_metrics = completed["baseline"]["metrics"].copy()
    if CHAMPION.exists():
        saved = json.loads(CHAMPION.read_text(encoding="utf-8"))
        champion_parameters = saved["parameters"]
        champion_metrics = saved["metrics"]

    for phase, values in PHASES:
        phase_base = champion_parameters.copy()
        phase_metrics = champion_metrics.copy()
        candidates = []
        for value in values:
            tag = str(value).lower().replace(".", "p")
            experiment_id = f"{phase}-{tag}"
            prefix = f"advanced_{experiment_id}"
            parameters = phase_base | {phase: value}
            if experiment_id in completed:
                row = completed[experiment_id]
            else:
                try:
                    payload = run(parameters, prefix)
                    candidate_metrics = metrics(payload)
                    adopted, score, reason = decision(candidate_metrics, phase_metrics)
                    row = {
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "experiment_id": experiment_id,
                        "phase": phase,
                        "value": value,
                        "parameters": parameters,
                        "metrics": candidate_metrics,
                        "eligible": adopted,
                        "reason": reason,
                        "score_delta": score,
                        "prefix": prefix,
                        "real_orders_enabled": False,
                    }
                except subprocess.CalledProcessError as exc:
                    row = {
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "experiment_id": experiment_id,
                        "phase": phase,
                        "value": value,
                        "parameters": parameters,
                        "eligible": False,
                        "reason": f"训练失败:{exc.returncode}",
                        "score_delta": -999.0,
                        "real_orders_enabled": False,
                    }
                append(row)
                completed[experiment_id] = row
            if "metrics" in row:
                candidates.append(row)
        eligible = [row for row in candidates if row.get("eligible")]
        if eligible:
            winner = max(eligible, key=lambda row: row["score_delta"])
            champion_parameters = winner["parameters"]
            champion_metrics = winner["metrics"]
            promote(winner["prefix"])
            winner_id = winner["experiment_id"]
            phase_decision = "PROMOTED"
        else:
            winner_id = "incumbent"
            phase_decision = "RETAINED"
        CHAMPION.write_text(
            json.dumps(
                {
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                    "last_phase": phase,
                    "decision": phase_decision,
                    "winner": winner_id,
                    "parameters": champion_parameters,
                    "metrics": champion_metrics,
                    "real_orders_enabled": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        all_rows = [row for row in completed.values() if "metrics" in row]
        frontier = [
            row for row in all_rows
            if not any(dominates(other["metrics"], row["metrics"]) for other in all_rows if other is not row)
        ]
        FRONTIER.write_text(json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"phase": phase, "decision": phase_decision, "winner": winner_id, "metrics": champion_metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
