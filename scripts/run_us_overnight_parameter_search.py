"""Sequential one-variable LightGBM search with a resumable paper-only ledger."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAINER = ROOT / "scripts" / "train_us_overnight_lightgbm.py"
MODELS = ROOT / "data" / "us_overnight_paper_only" / "models"
STATE = ROOT / "state" / "us-overnight-parameter-search"
LEDGER = STATE / "ledger.jsonl"
CHAMPION = STATE / "champion.json"

PARAMETERS = {
    "num_leaves": 23,
    "min_child_samples": 300,
    "colsample_bytree": 0.75,
    "reg_lambda": 3.0,
    "reg_alpha": 0.25,
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
}

PHASES = [
    ("reg_lambda", [4.0, 5.0, 2.5]),
    ("reg_alpha", [0.5, 1.0, 0.1]),
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
}


def _metrics(payload: dict, prefix: str) -> dict[str, float | int]:
    top1 = payload["validation_rank_slot_metrics"][0]
    daily = pd.read_csv(MODELS / f"{prefix}_walk_forward_daily.csv")
    midpoint = len(daily) // 2
    return {
        "violations": int(payload["validation_monotonic_violations"]),
        "top1_mean_bps": float(top1["mean_net_bps"]),
        "top1_sharpe": float(top1["sharpe"]),
        "top1_max_drawdown": float(top1["max_drawdown"]),
        "top1_cumulative_return": float(top1["cumulative_return"]),
        "top10_mean_bps": float(payload["validation_top_n_mean_net"] * 10_000),
        "top10_win_rate": float(payload["validation_top_n_win_rate"]),
        "top10_sharpe": float(payload["validation_top_n_sharpe"]),
        "top10_max_drawdown": float(payload["validation_top_n_max_drawdown"]),
        "first_half_top10_bps": float(daily.iloc[:midpoint]["top10_mean_net"].mean() * 10_000),
        "second_half_top10_bps": float(daily.iloc[midpoint:]["top10_mean_net"].mean() * 10_000),
    }


def _quality(metrics: dict[str, float | int], base: dict[str, float | int]) -> tuple[bool, float, str]:
    if metrics["violations"] > base["violations"]:
        return False, -1e9, "more_monotonic_violations"
    if metrics["top1_mean_bps"] < 0.9 * base["top1_mean_bps"]:
        return False, -1e9, "top1_mean_degraded_over_10pct"
    if metrics["top1_sharpe"] < 0.9 * base["top1_sharpe"]:
        return False, -1e9, "top1_sharpe_degraded_over_10pct"
    if metrics["top1_max_drawdown"] < base["top1_max_drawdown"] - 0.05:
        return False, -1e9, "top1_drawdown_worse_over_5pp"
    if metrics["top10_mean_bps"] < 0.8 * base["top10_mean_bps"]:
        return False, -1e9, "top10_mean_degraded_over_20pct"
    ratios = (
        0.35 * metrics["top1_mean_bps"] / base["top1_mean_bps"]
        + 0.35 * metrics["top1_sharpe"] / base["top1_sharpe"]
        + 0.15 * metrics["top10_mean_bps"] / base["top10_mean_bps"]
        + 0.15 * metrics["top10_sharpe"] / base["top10_sharpe"]
    )
    score = float(ratios - 0.25 * (metrics["violations"] - base["violations"]))
    return score > 1.0, score, "composite_improved" if score > 1.0 else "no_composite_improvement"


def _run(parameters: dict, prefix: str) -> dict:
    command = [str(PYTHON), str(TRAINER), "--output-prefix", prefix]
    for name, value in parameters.items():
        command.extend([CLI[name], str(value)])
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return json.loads((MODELS / f"{prefix}_ranker.json").read_text(encoding="utf-8"))


def _promote(prefix: str) -> None:
    for suffix in (
        "ranker.joblib",
        "ranker.json",
        "feature_importance.csv",
        "walk_forward_daily.csv",
        "rank_slot_metrics.csv",
    ):
        shutil.copy2(MODELS / f"{prefix}_{suffix}", MODELS / f"lightgbm_{suffix}")


def _append(record: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    champion_parameters = PARAMETERS.copy()
    if CHAMPION.exists():
        champion_parameters.update(json.loads(CHAMPION.read_text(encoding="utf-8"))["parameters"])
    completed = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed.add(row["experiment_id"])

    baseline_payload = json.loads((MODELS / "lightgbm_ranker.json").read_text(encoding="utf-8"))
    baseline_metrics = _metrics(baseline_payload, "lightgbm")
    for parameter, values in PHASES:
        if parameter == "subsample" and champion_parameters["subsample_freq"] == 0:
            continue
        phase_base_parameters = champion_parameters.copy()
        phase_base_metrics = baseline_metrics.copy()
        candidates = []
        for value in values:
            experiment_id = f"{parameter}-{str(value).replace('.', 'p')}"
            prefix = f"search_{experiment_id}"
            parameters = phase_base_parameters | {parameter: value}
            if experiment_id not in completed:
                payload = _run(parameters, prefix)
                metrics = _metrics(payload, prefix)
                eligible, score, reason = _quality(metrics, phase_base_metrics)
                _append(
                    {
                        "completed_at_utc": datetime.now(UTC).isoformat(),
                        "experiment_id": experiment_id,
                        "parameter": parameter,
                        "value": value,
                        "parameters": parameters,
                        "metrics": metrics,
                        "eligible": eligible,
                        "score": score,
                        "reason": reason,
                        "real_orders_enabled": False,
                    }
                )
            else:
                payload = json.loads((MODELS / f"{prefix}_ranker.json").read_text(encoding="utf-8"))
                metrics = _metrics(payload, prefix)
                eligible, score, reason = _quality(metrics, phase_base_metrics)
            candidates.append((eligible, score, value, prefix, metrics, reason))
        eligible_candidates = [row for row in candidates if row[0]]
        if eligible_candidates:
            winner = max(eligible_candidates, key=lambda row: row[1])
            _, score, value, prefix, metrics, reason = winner
            champion_parameters[parameter] = value
            baseline_metrics = metrics
            _promote(prefix)
            decision = "PROMOTED"
        else:
            score, value, prefix, metrics, reason = 1.0, phase_base_parameters[parameter], "lightgbm", phase_base_metrics, "incumbent_retained"
            decision = "RETAINED"
        CHAMPION.write_text(
            json.dumps(
                {
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                    "last_parameter": parameter,
                    "decision": decision,
                    "parameters": champion_parameters,
                    "metrics": baseline_metrics,
                    "score": score,
                    "reason": reason,
                    "real_orders_enabled": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"parameter": parameter, "decision": decision, "value": value, "metrics": metrics}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
