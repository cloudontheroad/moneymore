"""Search one variable: exponential half-life of the isotonic calibrator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from evaluate_us_corrected_return_calibrator import (
    HOLD,
    LOOKBACK,
    PRED,
    REPORT_DIR,
    add_normalized_score,
    evaluate_baseline,
    stats,
)

HALF_LIVES = [21, 42, 63, 126]


def run(frame: pd.DataFrame, dates: list, half_life: int) -> tuple[dict, pd.DataFrame]:
    pieces = []
    for start in range(LOOKBACK, len(dates), HOLD):
        history_dates = dates[start - LOOKBACK:start]
        hold_dates = dates[start:min(start + HOLD, len(dates))]
        history = frame[frame.timestamp.isin(history_dates)].copy()
        hold = frame[frame.timestamp.isin(hold_dates)].copy()
        date_age = {date: len(history_dates) - 1 - i for i, date in enumerate(history_dates)}
        weights = history.timestamp.map(lambda date: 0.5 ** (date_age[date] / half_life))
        calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        calibrator.fit(
            history.score_z.to_numpy(),
            history.label_next_open_net.to_numpy(),
            sample_weight=weights.to_numpy(),
        )
        hold["expected_net_return"] = calibrator.predict(hold.score_z.to_numpy())
        hold["trade"] = hold.expected_net_return > 0
        pieces.append(hold)
    calibrated = pd.concat(pieces, ignore_index=True)
    test_dates = dates[LOOKBACK:]
    top1_rows = calibrated[calibrated["rank"] == 1].set_index("timestamp")
    top1 = top1_rows.label_next_open_net.where(top1_rows.trade, 0).reindex(test_dates, fill_value=0)
    traded = calibrated[calibrated.trade]
    top10 = traded.groupby("timestamp").label_next_open_net.mean().reindex(test_dates, fill_value=0)
    counts = traded.groupby("timestamp").size().reindex(test_dates, fill_value=0)
    return {
        "half_life": half_life,
        "top1": stats(top1),
        "top10": stats(top10),
        "average_positions": float(counts.mean()),
        "zero_position_sessions": int((counts == 0).sum()),
    }, calibrated


def main() -> int:
    frame = add_normalized_score(pd.read_parquet(PRED))
    dates = list(pd.Index(frame.timestamp.unique()).sort_values())
    baseline = evaluate_baseline(frame, dates[LOOKBACK:])
    experiments, predictions = [], {}
    for half_life in HALF_LIVES:
        result, pred = run(frame, dates, half_life)
        experiments.append(result)
        predictions[half_life] = pred
    rows = []
    for result in experiments:
        rows.append({
            "half_life": result["half_life"],
            "average_positions": result["average_positions"],
            "zero_position_sessions": result["zero_position_sessions"],
            **{f"top1_{key}": value for key, value in result["top1"].items()},
            **{f"top10_{key}": value for key, value in result["top10"].items()},
        })
    table = pd.DataFrame(rows)
    table["joint_sharpe"] = table.top1_sharpe + table.top10_sharpe
    winner_row = table.sort_values("joint_sharpe", ascending=False).iloc[0]
    winner_half_life = int(winner_row.half_life)
    winner = next(item for item in experiments if item["half_life"] == winner_half_life)
    improved = all([
        winner["top1"]["mean_bps"] > baseline["top1"]["mean_bps"],
        winner["top1"]["sharpe"] > baseline["top1"]["sharpe"],
        winner["top10"]["mean_bps"] > baseline["top10"]["mean_bps"],
        winner["top10"]["sharpe"] > baseline["top10"]["sharpe"],
    ])
    report = {
        "variable": "isotonic calibration exponential half-life",
        "baseline": baseline,
        "winner": winner,
        "decision": "accept" if improved else "reject all",
        "experiments": experiments,
    }
    table.to_csv(REPORT_DIR / "corrected_calibrator_decay_search.csv", index=False)
    predictions[winner_half_life].to_parquet(
        REPORT_DIR / "corrected_calibrator_decay_winner_predictions.parquet", index=False
    )
    (REPORT_DIR / "corrected_calibrator_decay_search.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
