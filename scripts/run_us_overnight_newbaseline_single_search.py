"""Replay every historical one-variable experiment on the bundle champion.

All candidates use the same corrected panel, four-module feature bundle and
validation protocol. Results form a candidate pool; no greedy promotion occurs
inside this script. Bundle-aware parameter combination happens afterwards.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/Scripts/python.exe"
TRAINER = ROOT / "scripts/train_us_overnight_lightgbm.py"
PANEL = ROOT / "data/us_overnight_corrected/features/historical_panel_bundle_modules.parquet"
FEATURES = ROOT / "data/us_overnight_corrected/features/feature_columns_bundle_right_market_options_calendar.json"
MODELS = ROOT / "data/us_overnight_paper_only/models/newbaseline_single_search"
STATE = ROOT / "state/us-overnight-newbaseline-single-search"
LEDGER = STATE / "ledger.jsonl"

BASE = {
    "num_leaves": 23, "min_child_samples": 300, "colsample_bytree": 0.75,
    "reg_lambda": 3.0, "reg_alpha": 0.25, "min_split_gain": 0.0,
    "max_depth": -1, "learning_rate": 0.025, "n_estimators": 350,
    "huber_alpha": 0.9, "subsample": 0.8, "subsample_freq": 0,
    "weight_half_life": 126, "rolling_train_sessions": 504,
    "target_transform": "raw", "min_child_weight": 50.0, "path_smooth": 0.0,
    "feature_fraction_bynode": 1.0, "extra_trees": False, "max_bin": 255,
    "min_data_in_bin": 3, "boosting_type": "gbdt", "data_sample_strategy": "bagging",
    "top1_weight": 1.0, "profit_early_stopping_rounds": 0,
}

GRID = {
    "num_leaves": [15, 21, 25, 27, 31],
    "min_child_samples": [250, 275, 400],
    "colsample_bytree": [0.70, 0.80],
    "reg_lambda": [2.0, 2.5, 4.0, 5.0, 10.0, 30.0, 100.0, 300.0],
    "reg_alpha": [0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0],
    "min_split_gain": [0.01, 0.05, 0.1],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.015, 0.02, 0.03],
    "n_estimators": [250, 450, 550],
    "huber_alpha": [0.8, 0.85, 0.95],
    "weight_half_life": [63, 189, 252],
    "rolling_train_sessions": [378, 630, 756],
    "subsample_freq": [1],
    "target_transform": ["daily_mad"],
    "path_smooth": [1.0, 5.0, 10.0, 20.0, 50.0],
    "min_child_weight": [0.1, 1.0, 5.0, 10.0, 25.0],
    "feature_fraction_bynode": [0.9, 0.8, 0.7, 0.6],
    "extra_trees": [True],
    "max_bin": [63, 127, 511, 1023],
    "min_data_in_bin": [10, 30, 100, 300],
    "profit_early_stopping_rounds": [20, 40, 80],
    "data_sample_strategy": ["goss"],
    "boosting_type": ["dart"],
    "top1_weight": [1.5, 2.0, 3.0, 5.0],
}

CLI = {
    "num_leaves": "--num-leaves", "min_child_samples": "--min-child-samples",
    "colsample_bytree": "--colsample-bytree", "reg_lambda": "--reg-lambda",
    "reg_alpha": "--reg-alpha", "min_split_gain": "--min-split-gain",
    "max_depth": "--max-depth", "learning_rate": "--learning-rate",
    "n_estimators": "--n-estimators", "huber_alpha": "--huber-alpha",
    "subsample": "--subsample", "subsample_freq": "--subsample-freq",
    "weight_half_life": "--weight-half-life", "rolling_train_sessions": "--rolling-train-sessions",
    "target_transform": "--target-transform", "min_child_weight": "--min-child-weight",
    "path_smooth": "--path-smooth", "feature_fraction_bynode": "--feature-fraction-bynode",
    "extra_trees": "--extra-trees", "max_bin": "--max-bin", "min_data_in_bin": "--min-data-in-bin",
    "boosting_type": "--boosting-type", "data_sample_strategy": "--data-sample-strategy",
    "top1_weight": "--top1-weight", "profit_early_stopping_rounds": "--profit-early-stopping-rounds",
}


def tag(value: object) -> str:
    return str(value).lower().replace(".", "p")


def command(parameters: dict, prefix: str) -> list[str]:
    result = [str(PYTHON), str(TRAINER), "--panel-path", str(PANEL), "--feature-columns-json", str(FEATURES),
              "--model-dir", str(MODELS), "--output-prefix", prefix, "--validation-sessions", "126", "--retrain-every", "5"]
    for name, value in parameters.items():
        if isinstance(value, bool):
            if value:
                result.append(CLI[name])
        else:
            result.extend([CLI[name], str(value)])
    return result


def summarize(payload: dict) -> dict:
    top1 = payload["validation_rank_slot_metrics"][0]
    return {
        "top1_mean_bps": top1["mean_net_bps"], "top1_sharpe": top1["sharpe"],
        "top1_win_rate": top1["win_rate"], "top1_max_drawdown": top1["max_drawdown"],
        "top10_mean_bps": payload["validation_top_n_mean_net"] * 10_000,
        "top10_sharpe": payload["validation_top_n_sharpe"],
        "top10_win_rate": payload["validation_top_n_win_rate"],
        "top10_max_drawdown": payload["validation_top_n_max_drawdown"],
    }


def run_one(item: tuple[str, object]) -> dict:
    parameter, value = item
    experiment_id = f"{parameter}-{tag(value)}"
    prefix = f"single_{experiment_id}"
    output = MODELS / f"{prefix}_ranker.json"
    parameters = BASE | {parameter: value}
    if not output.exists():
        subprocess.run(command(parameters, prefix), cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {"completed_at_utc": datetime.now(UTC).isoformat(), "experiment_id": experiment_id,
            "parameter": parameter, "value": value, "parameters": parameters, "metrics": summarize(payload), "prefix": prefix}


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True); STATE.mkdir(parents=True, exist_ok=True)
    items = [(parameter, value) for parameter, values in GRID.items() for value in values]
    completed = set()
    if LEDGER.exists():
        completed = {json.loads(line)["experiment_id"] for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [item for item in items if f"{item[0]}-{tag(item[1])}" not in completed]
    print(json.dumps({"total": len(items), "completed": len(completed), "pending": len(pending)}), flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_one, item): item for item in pending}
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"completed_at_utc": datetime.now(UTC).isoformat(), "experiment_id": f"{item[0]}-{tag(item[1])}",
                       "parameter": item[0], "value": item[1], "error": repr(exc)}
            with LEDGER.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"finished": len(completed) + index, "total": len(items), "experiment": row["experiment_id"],
                              "metrics": row.get("metrics"), "error": row.get("error")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
