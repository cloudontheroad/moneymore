"""Sequential corrected-data factor-group ablation using the parameter champion."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/Scripts/python.exe"
TRAINER = ROOT / "scripts/train_us_overnight_lightgbm.py"
PANEL = ROOT / "data/us_overnight_corrected/features/historical_panel_factor_groups_lag1.parquet"
GROUPS_PATH = ROOT / "data/us_overnight_corrected/features/factor_groups.json"
PARAM_CHAMPION = ROOT / "state/us-overnight-corrected-search/champion.json"
STATE = ROOT / "state/us-overnight-v2-factor-search"
MODELS = ROOT / "data/us_overnight_corrected/models/v2_factor_search"
LEDGER = STATE / "ledger.jsonl"

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
    "drop_rate": "--drop-rate", "skip_drop": "--skip-drop", "max_drop": "--max-drop",
    "top_rate": "--top-rate", "other_rate": "--other-rate", "top1_weight": "--top1-weight",
    "profit_early_stopping_rounds": "--profit-early-stopping-rounds",
    "validation_sessions": "--validation-sessions", "retrain_every": "--retrain-every",
}


def metric(payload: dict, prefix: str) -> dict[str, float | int]:
    slots = pd.DataFrame(payload["validation_rank_slot_metrics"])
    daily = pd.read_csv(MODELS / f"{prefix}_walk_forward_daily.csv")
    top10 = daily["top10_mean_net"]
    top1 = slots.iloc[0]
    return {
        "violations": int(payload["validation_monotonic_violations"]),
        "top1_mean_bps": float(top1["mean_net_bps"]),
        "top1_sharpe": float(top1["sharpe"]),
        "top1_annual_volatility": float(top1["mean_net_bps"] / 10_000 * 252 / top1["sharpe"]) if top1["sharpe"] else 0.0,
        "top1_cumulative": float(top1["cumulative_return"]),
        "top1_drawdown": float(top1["max_drawdown"]),
        "top10_mean_bps": float(top10.mean() * 10_000),
        "top10_sharpe": float(top10.mean() / top10.std(ddof=1) * np.sqrt(252)),
        "top10_volatility": float(top10.std(ddof=1) * np.sqrt(252)),
        "top10_cumulative": float((1 + top10).prod() - 1),
        "top10_drawdown": float(((1 + top10).cumprod() / (1 + top10).cumprod().cummax() - 1).min()),
        "top10_win_rate": float((top10 > 0).mean()),
    }


def run(parameters: dict, features: list[str], prefix: str) -> dict:
    feature_file = STATE / f"{prefix}_features.json"
    feature_file.write_text(json.dumps(features), encoding="utf-8")
    command = [str(PYTHON), str(TRAINER), "--panel-path", str(PANEL), "--model-dir", str(MODELS),
               "--feature-columns-json", str(feature_file), "--output-prefix", prefix]
    for name, value in parameters.items():
        if name not in CLI:
            continue
        if isinstance(value, bool):
            if value:
                command.append(CLI[name])
        else:
            command.extend([CLI[name], str(value)])
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads((MODELS / f"{prefix}_ranker.json").read_text(encoding="utf-8"))


def acceptable(candidate: dict, base: dict) -> tuple[bool, float]:
    # Strict cumulative Pareto gate: an improvement cannot compensate for a
    # deterioration in another portfolio objective.
    hard = (
        candidate["violations"] <= base["violations"]
        and candidate["top1_mean_bps"] >= base["top1_mean_bps"]
        and candidate["top1_sharpe"] >= base["top1_sharpe"]
        and candidate["top1_drawdown"] >= base["top1_drawdown"]
        and candidate["top10_mean_bps"] >= base["top10_mean_bps"]
        and candidate["top10_sharpe"] >= base["top10_sharpe"]
        and candidate["top10_drawdown"] >= base["top10_drawdown"]
    )
    score = (
        0.40 * (candidate["top1_mean_bps"] - base["top1_mean_bps"]) / 5
        + 0.25 * (candidate["top1_sharpe"] - base["top1_sharpe"]) / 0.5
        + 0.20 * (candidate["top10_mean_bps"] - base["top10_mean_bps"]) / 5
        + 0.15 * (candidate["top10_sharpe"] - base["top10_sharpe"]) / 0.5
        + 0.25 * (base["violations"] - candidate["violations"])
    )
    return bool(hard and score > 0.05), float(score)


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    groups = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    parameters = json.loads(PARAM_CHAMPION.read_text(encoding="utf-8"))["parameters"]
    base = groups["base"]
    candidates = [
        ("trend_first_turn", [x for x in groups["right_side_trend"] if x.startswith(("first_up_", "first_close_above_"))]),
        ("trend_persistence", [x for x in groups["right_side_trend"] if x.startswith(("trend_efficiency_", "up_day_ratio_", "ma_slope_", "trend_stack_"))]),
        ("trend_breakout", [x for x in groups["right_side_trend"] if x.startswith(("breakout_", "near_high_"))]),
        ("trend_pullback", [x for x in groups["right_side_trend"] if x.startswith("pullback_reversal_")]),
        ("trend_all", groups["right_side_trend"]),
        ("overnight_w5", [x for x in groups["overnight_history"] if x.endswith("_5")]),
        ("overnight_w20", [x for x in groups["overnight_history"] if x.endswith("_20")]),
        ("overnight_w60", [x for x in groups["overnight_history"] if x.endswith("_60")]),
        ("overnight_all", groups["overnight_history"]),
        ("market_spy", [x for x in groups["market_regime"] if x.endswith("_SPY")]),
        ("market_vol", [x for x in groups["market_regime"] if "vol20" in x or "VIX" in x]),
        ("market_all", groups["market_regime"]),
        ("sector_1d", [x for x in groups["sector_regime"] if "ret1" in x]),
        ("sector_5d", [x for x in groups["sector_regime"] if "ret5" in x]),
        ("sector_all", groups["sector_regime"]),
        ("calendar", groups["calendar"]),
    ]
    payload = run(parameters, base, "factor_baseline")
    champion_metrics = metric(payload, "factor_baseline")
    champion_features = base.copy()
    adopted = 0
    records = []
    for name, additions in candidates:
        features = list(dict.fromkeys([*champion_features, *additions]))
        payload = run(parameters, features, f"factor_{name}")
        metrics = metric(payload, f"factor_{name}")
        keep, score = acceptable(metrics, champion_metrics)
        if keep:
            champion_features, champion_metrics = features, metrics
            adopted += 1
        row = {"completed_at_utc": datetime.now(UTC).isoformat(), "experiment": name,
               "factor_count": len(features), "added": additions, "adopted": keep,
               "score_delta": score, "metrics": metrics}
        records.append(row)
        with LEDGER.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"experiment": name, "adopted": keep, "metrics": metrics}), flush=True)
    summary = {"experiments": len(candidates) + 1, "adopted": adopted, "parameters": parameters,
               "features": champion_features, "metrics": champion_metrics, "records": records}
    (STATE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
