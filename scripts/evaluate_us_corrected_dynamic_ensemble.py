"""Prequential audit of corrected open-to-open ensemble weights.

At every rebalance the candidate blend is selected only from the trailing
calibration window.  The chosen weights are then frozen for the following
holding block.  This prevents the full-year ensemble search from selecting
and evaluating weights on the same returns.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"
OUT_JSON = REPORT_DIR / "corrected_open2open_dynamic_ensemble.json"
OUT_CSV = REPORT_DIR / "corrected_open2open_dynamic_ensemble_blocks.csv"
OUT_PREDICTIONS = REPORT_DIR / "corrected_open2open_dynamic_ensemble_predictions.parquet"

MODELS = {
    "champion": MODEL_DIR / "corrected_champion_selected_predictions.parquet",
    "half63": MODEL_DIR / "corrected_half63_selected_predictions.parquet",
    "top1w5": MODEL_DIR / "corrected_top1w5_selected_predictions.parquet",
    "daily_centered": MODEL_DIR / "corrected_daily_centered_selected_predictions.parquet",
}
INCUMBENT = {"champion": 0.3, "half63": 0.2, "top1w5": 0.2, "daily_centered": 0.3}
RETROSPECTIVE = {"champion": 0.1, "half63": 0.1, "top1w5": 0.7, "daily_centered": 0.1}
CALIBRATION_DAYS = 63
HOLD_DAYS = 21


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first, *rest)


def load() -> pd.DataFrame:
    parts = []
    for name, path in MODELS.items():
        frame = pd.read_parquet(path)[["timestamp", "symbol", "label_next_open_net", "rank"]]
        frame = frame.rename(columns={"rank": name, "label_next_open_net": f"label_{name}"})
        parts.append(frame)
    merged = parts[0]
    for frame in parts[1:]:
        merged = merged.merge(frame, on=["timestamp", "symbol"], how="outer")
    labels = [column for column in merged if column.startswith("label_")]
    merged["label"] = merged[labels].bfill(axis=1).iloc[:, 0]
    return merged.drop(columns=labels).sort_values(["timestamp", "symbol"])


def select(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    score = sum(weight * frame[name].fillna(11.0) for name, weight in weights.items())
    out = frame[["timestamp", "symbol", "label"]].copy()
    out["score"] = score
    out["rank"] = out.groupby("timestamp")["score"].rank(method="first")
    return out[out["rank"] <= 10]


def stats(series: pd.Series) -> dict[str, float]:
    series = series.dropna().sort_index()
    equity = (1 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "sessions": int(len(series)),
        "mean_bps": float(series.mean() * 10_000),
        "win_rate": float((series > 0).mean()),
        "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "cumulative_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def returns(selected: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    top1 = selected[selected["rank"] == 1].set_index("timestamp")["label"]
    top10 = selected.groupby("timestamp")["label"].mean()
    return top1, top10


def calibration_score(top1: pd.Series, top10: pd.Series) -> float:
    """Equal-weight the two declared goals using scale-free Sharpe."""
    return stats(top1)["sharpe"] + stats(top10)["sharpe"]


def evaluate_fixed(frame: pd.DataFrame, dates: list, weights: dict[str, float]) -> dict:
    chosen = select(frame[frame.timestamp.isin(dates)], weights)
    top1, top10 = returns(chosen)
    return {"top1": stats(top1), "top10": stats(top10)}


def main() -> int:
    frame = load()
    dates = list(pd.Index(frame.timestamp.unique()).sort_values())
    candidates = []
    names = list(MODELS)
    for units in compositions(10, len(names)):
        if sum(value > 0 for value in units) < 2:
            continue
        candidates.append({name: value / 10 for name, value in zip(names, units) if value})

    chosen_parts = []
    blocks = []
    start = CALIBRATION_DAYS
    while start < len(dates):
        calibration_dates = dates[start - CALIBRATION_DAYS:start]
        hold_dates = dates[start:min(start + HOLD_DAYS, len(dates))]
        calibration = frame[frame.timestamp.isin(calibration_dates)]
        scored = []
        for weights in candidates:
            top1, top10 = returns(select(calibration, weights))
            scored.append((calibration_score(top1, top10), weights))
        score, weights = max(scored, key=lambda item: item[0])
        held = select(frame[frame.timestamp.isin(hold_dates)], weights)
        held["block"] = len(blocks) + 1
        chosen_parts.append(held)
        top1, top10 = returns(held)
        blocks.append({
            "block": len(blocks) + 1,
            "calibration_start": str(pd.Timestamp(calibration_dates[0]).date()),
            "calibration_end": str(pd.Timestamp(calibration_dates[-1]).date()),
            "hold_start": str(pd.Timestamp(hold_dates[0]).date()),
            "hold_end": str(pd.Timestamp(hold_dates[-1]).date()),
            "weights": json.dumps(weights, sort_keys=True),
            "calibration_score": score,
            "top1_mean_bps": stats(top1)["mean_bps"],
            "top10_mean_bps": stats(top10)["mean_bps"],
        })
        start += HOLD_DAYS

    dynamic = pd.concat(chosen_parts, ignore_index=True)
    top1, top10 = returns(dynamic)
    test_dates = dates[CALIBRATION_DAYS:]
    initial = frame[frame.timestamp.isin(dates[:CALIBRATION_DAYS])]
    frozen_scores = []
    for weights in candidates:
        frozen_top1, frozen_top10 = returns(select(initial, weights))
        frozen_scores.append((calibration_score(frozen_top1, frozen_top10), weights))
    initial_score, initial_weights = max(frozen_scores, key=lambda item: item[0])
    report = {
        "method": "trailing-63-session selection, frozen 21 sessions",
        "selection_objective": "top1 Sharpe + top10 Sharpe on past data only",
        "candidate_count": len(candidates),
        "test_start": str(pd.Timestamp(test_dates[0]).date()),
        "test_end": str(pd.Timestamp(test_dates[-1]).date()),
        "dynamic": {"top1": stats(top1), "top10": stats(top10)},
        "initial_63_frozen": {
            "weights": initial_weights,
            "calibration_score": initial_score,
            **evaluate_fixed(frame, test_dates, initial_weights),
        },
        "equal_weight": evaluate_fixed(frame, test_dates, {name: 0.25 for name in names}),
        "fixed_incumbent": evaluate_fixed(frame, test_dates, INCUMBENT),
        "fixed_retrospective_winner": evaluate_fixed(frame, test_dates, RETROSPECTIVE),
        "blocks": blocks,
        "decision_rule": "adopt only if dynamic improves both declared goals out of sample",
    }
    dynamic.to_parquet(OUT_PREDICTIONS, index=False)
    pd.DataFrame(blocks).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "dynamic", "initial_63_frozen", "equal_weight", "fixed_incumbent",
        "fixed_retrospective_winner")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
