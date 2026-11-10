"""Search anchored rank blends using the existing primary adoption metrics."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/us_overnight_paper_only/models/anchor_verification"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"
BASE_MODELS = {
    "champion": MODEL_DIR / "anchored_champion_252_selected_predictions.parquet",
    "mcw1": MODEL_DIR / "anchored_mcw1_252_selected_predictions.parquet",
    "half63": MODEL_DIR / "anchored_half63_252_selected_predictions.parquet",
    "top1w5": MODEL_DIR / "anchored_top1w5_252_selected_predictions.parquet",
}


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in compositions(total - first, parts - 1):
            yield (first, *remainder)


def load() -> pd.DataFrame:
    parts = []
    for name, path in MODELS.items():
        frame = pd.read_parquet(path)[
            ["timestamp", "symbol", "label_next_open_net", "rank"]
        ].rename(columns={"rank": name, "label_next_open_net": f"label_{name}"})
        parts.append(frame)
    merged = parts[0]
    for frame in parts[1:]:
        merged = merged.merge(frame, on=["timestamp", "symbol"], how="outer")
    labels = [column for column in merged if column.startswith("label_")]
    merged["label"] = merged[labels].bfill(axis=1).iloc[:, 0]
    return merged.drop(columns=labels)


def select(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    score = sum(weight * frame[name].fillna(11.0) for name, weight in weights.items())
    result = frame[["timestamp", "symbol", "label"]].copy()
    result["score"] = score
    result["rank"] = result.groupby("timestamp")["score"].rank(method="first")
    return result.loc[result["rank"] <= 10]


def stats(series: pd.Series) -> dict[str, float]:
    equity = (1 + series).cumprod()
    return {
        "mean_bps": float(series.mean() * 10_000),
        "sharpe": float(series.mean() / series.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "cumulative_return": float(equity.iloc[-1] - 1),
    }


def evaluate(selected: pd.DataFrame) -> dict[str, float]:
    top1 = stats(selected.loc[selected["rank"] == 1].set_index("timestamp")["label"])
    top10 = stats(selected.groupby("timestamp")["label"].mean())
    return {f"top1_{key}": value for key, value in top1.items()} | {
        f"top10_{key}": value for key, value in top10.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-target-transforms", action="store_true")
    parser.add_argument("--include-daily-zscore", action="store_true")
    parser.add_argument("--investable", action="store_true")
    parser.add_argument("--corrected-open2open", action="store_true")
    args = parser.parse_args()
    global MODELS
    MODELS = dict(BASE_MODELS)
    baseline_weights = {"champion": 1.0}
    report_stem = "anchored_parameter_ensemble_search"
    if args.include_target_transforms:
        MODELS.update({
            "daily_centered": MODEL_DIR / "anchored_target_daily_centered_252_selected_predictions.parquet",
            "daily_rank": MODEL_DIR / "anchored_target_daily_rank_252_selected_predictions.parquet",
        })
        baseline_weights = {"champion": 0.9, "mcw1": 0.1}
        report_stem = "anchored_target_transform_ensemble_search"
    if args.include_daily_zscore:
        MODELS.update({
            "daily_centered": MODEL_DIR / "anchored_target_daily_centered_252_selected_predictions.parquet",
            "daily_rank": MODEL_DIR / "anchored_target_daily_rank_252_selected_predictions.parquet",
            "daily_zscore": MODEL_DIR / "anchored_target_daily_zscore_252_selected_predictions.parquet",
        })
        baseline_weights = {
            "champion": 0.3,
            "half63": 0.2,
            "top1w5": 0.2,
            "daily_centered": 0.3,
        }
        report_stem = "anchored_daily_zscore_ensemble_search"
    incumbent_weights = None
    if args.investable:
        clean_dir = ROOT / "data/us_overnight_paper_only/models/investable_120d_20m_5"
        MODELS = {
            "champion": clean_dir / "clean_champion_selected_predictions.parquet",
            "half63": clean_dir / "clean_half63_selected_predictions.parquet",
            "top1w5": clean_dir / "clean_top1w5_selected_predictions.parquet",
            "daily_centered": clean_dir / "clean_daily_centered_selected_predictions.parquet",
        }
        incumbent_weights = {
            "champion": 0.3,
            "half63": 0.2,
            "top1w5": 0.2,
            "daily_centered": 0.3,
        }
        baseline_weights = incumbent_weights
        report_stem = "investable_120d_20m_5_ensemble_search"
    if args.corrected_open2open:
        corrected_dir = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected"
        MODELS = {
            "champion": corrected_dir / "corrected_champion_selected_predictions.parquet",
            "half63": corrected_dir / "corrected_half63_selected_predictions.parquet",
            "top1w5": corrected_dir / "corrected_top1w5_selected_predictions.parquet",
            "daily_centered": corrected_dir / "corrected_daily_centered_selected_predictions.parquet",
        }
        incumbent_weights = {
            "champion": 0.3,
            "half63": 0.2,
            "top1w5": 0.2,
            "daily_centered": 0.3,
        }
        baseline_weights = incumbent_weights
        report_stem = "investable_corrected_open2open_ensemble_search"
    frame = load()
    base = evaluate(select(frame, baseline_weights))
    rows = []
    names = list(MODELS)
    for units in compositions(10, len(names)):
        if sum(value > 0 for value in units) < 2:
            continue
        weights = {name: value / 10 for name, value in zip(names, units) if value}
        metrics = evaluate(select(frame, weights))
        primary_improvement = all([
            metrics["top1_mean_bps"] > base["top1_mean_bps"],
            metrics["top1_sharpe"] > base["top1_sharpe"],
            metrics["top10_mean_bps"] > base["top10_mean_bps"],
            metrics["top10_sharpe"] > base["top10_sharpe"],
        ])
        rows.append({"weights": json.dumps(weights), **metrics,
                     "primary_improvement": primary_improvement})
    result = pd.DataFrame(rows)
    result["balanced_delta"] = (
        (result.top1_mean_bps - base["top1_mean_bps"]) / 25
        + (result.top1_sharpe - base["top1_sharpe"])
        + (result.top10_mean_bps - base["top10_mean_bps"]) / 10
        + (result.top10_sharpe - base["top10_sharpe"])
    )
    result = result.sort_values(["primary_improvement", "balanced_delta"], ascending=False)
    winner = result.iloc[0].to_dict()
    weights = json.loads(winner["weights"])
    selected = select(frame, weights)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT_DIR / f"{report_stem}.csv", index=False)
    selected.to_parquet(REPORT_DIR / f"{report_stem}_winner.parquet", index=False)
    report = {
        "tested": int(len(result)),
        "primary_improvements": int(result.primary_improvement.sum()),
        "baseline": base,
        "winner": winner,
        "decision": "accept" if winner["primary_improvement"] else "reject",
        "reference_metrics_are_not_adoption_gates": True,
    }
    if incumbent_weights is not None:
        report["previous_champion_fixed_weights"] = incumbent_weights
        report["previous_champion_on_investable_panel"] = evaluate(
            select(frame, incumbent_weights)
        )
    (REPORT_DIR / f"{report_stem}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
