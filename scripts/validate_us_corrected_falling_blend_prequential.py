"""Prequential validation of the main/ablation blend weight."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"
CALIBRATION = 63
HOLD = 21
WEIGHTS = [value / 10 for value in range(11)]


def load() -> pd.DataFrame:
    main = pd.read_parquet(
        MODEL_DIR / "corrected_cs_varying_clip10_selected_predictions.parquet"
    )[["timestamp", "symbol", "label_next_open_net", "rank"]].rename(
        columns={"rank": "main_rank", "label_next_open_net": "main_label"}
    )
    ablation = pd.read_parquet(
        MODEL_DIR / "corrected_cs_varying_clip10_no_falling_selected_predictions.parquet"
    )[["timestamp", "symbol", "label_next_open_net", "rank"]].rename(
        columns={"rank": "ablation_rank", "label_next_open_net": "ablation_label"}
    )
    merged = main.merge(ablation, on=["timestamp", "symbol"], how="outer")
    merged["label"] = merged.main_label.fillna(merged.ablation_label)
    return merged


def select(frame: pd.DataFrame, main_weight: float) -> pd.DataFrame:
    out = frame[["timestamp", "symbol", "label"]].copy()
    out["score"] = (
        main_weight * frame.main_rank.fillna(11)
        + (1 - main_weight) * frame.ablation_rank.fillna(11)
    )
    out["rank"] = out.groupby("timestamp")["score"].rank(method="first")
    return out[out["rank"] <= 10]


def returns(selected: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    top1 = selected[selected["rank"] == 1].set_index("timestamp")["label"]
    top10 = selected.groupby("timestamp")["label"].mean()
    return top1, top10


def stats(series: pd.Series) -> dict:
    series = series.sort_index()
    equity = (1 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "sessions": int(len(series)),
        "mean_bps": float(series.mean() * 10_000),
        "win_rate": float((series > 0).mean()),
        "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "cumulative_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "without_best5_mean_bps": float(series.nsmallest(max(1, len(series) - 5)).mean() * 10_000),
    }


def evaluate(selected: pd.DataFrame) -> dict:
    top1, top10 = returns(selected)
    return {"top1": stats(top1), "top10": stats(top10)}


def utility(frame: pd.DataFrame, weight: float) -> float:
    top1, top10 = returns(select(frame, weight))
    return stats(top1)["sharpe"] + stats(top10)["sharpe"]


def fixed(frame: pd.DataFrame, dates: list, weight: float) -> dict:
    return evaluate(select(frame[frame.timestamp.isin(dates)], weight))


def main() -> int:
    frame = load()
    dates = list(pd.Index(frame.timestamp.unique()).sort_values())
    calibration_dates = dates[:CALIBRATION]
    test_dates = dates[CALIBRATION:]
    calibration_frame = frame[frame.timestamp.isin(calibration_dates)]
    frozen_weight = max(WEIGHTS, key=lambda weight: utility(calibration_frame, weight))
    calibration_126_dates = dates[:126]
    test_126_dates = dates[126:]
    calibration_126 = frame[frame.timestamp.isin(calibration_126_dates)]
    frozen_126_weight = max(WEIGHTS, key=lambda weight: utility(calibration_126, weight))

    rolling_parts, blocks = [], []
    for start in range(CALIBRATION, len(dates), HOLD):
        history_dates = dates[start - CALIBRATION:start]
        hold_dates = dates[start:min(start + HOLD, len(dates))]
        history = frame[frame.timestamp.isin(history_dates)]
        weight = max(WEIGHTS, key=lambda value: utility(history, value))
        held = select(frame[frame.timestamp.isin(hold_dates)], weight)
        held["selected_weight"] = weight
        rolling_parts.append(held)
        block_metrics = evaluate(held)
        blocks.append({
            "hold_start": str(pd.Timestamp(hold_dates[0]).date()),
            "hold_end": str(pd.Timestamp(hold_dates[-1]).date()),
            "selected_main_weight": weight,
            "top1_mean_bps": block_metrics["top1"]["mean_bps"],
            "top10_mean_bps": block_metrics["top10"]["mean_bps"],
        })
    rolling = pd.concat(rolling_parts, ignore_index=True)
    report = {
        "protocol": "first 63 sessions calibrate; next 189 sessions are untouched test",
        "frozen_selected_main_weight": frozen_weight,
        "frozen_selected": fixed(frame, test_dates, frozen_weight),
        "rolling_63_21": evaluate(rolling),
        "fixed_main_only": fixed(frame, test_dates, 1.0),
        "fixed_retrospective_90_10_reference": fixed(frame, test_dates, 0.9),
        "frozen_126_selected_main_weight": frozen_126_weight,
        "frozen_126_test": fixed(frame, test_126_dates, frozen_126_weight),
        "rolling_blocks": blocks,
        "decision_rule": "retain blend only if past-only selection improves both Top1 and Top10 versus main-only",
        "decision": "reject 90/10 as validated baseline; keep main-only",
        "reason": (
            "The 63-session calibration selects main-only; rolling selection damages Top10. "
            "The 126-session calibration selects 90/10 but the following 126 sessions are negative."
        ),
    }
    pd.DataFrame(blocks).to_csv(
        REPORT_DIR / "corrected_falling_blend_prequential_blocks.csv", index=False
    )
    rolling.to_parquet(
        REPORT_DIR / "corrected_falling_blend_prequential_predictions.parquet", index=False
    )
    (REPORT_DIR / "corrected_falling_blend_prequential.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "frozen_selected_main_weight", "frozen_selected", "rolling_63_21",
        "fixed_main_only", "fixed_retrospective_90_10_reference",
        "frozen_126_selected_main_weight", "frozen_126_test")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
