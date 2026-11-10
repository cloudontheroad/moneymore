"""Point-in-time expected-return calibration for the accepted stock ranker.

The stock model and ordering remain frozen.  An isotonic calibrator maps each
candidate's within-day standardized model score to expected net return using
only the preceding 126 sessions.  Calibration is frozen for the next 21
sessions.  Candidates with non-positive expected net return are held as cash.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected/corrected_cs_varying_clip10_selected_predictions.parquet"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"
LOOKBACK = 126
HOLD = 21


def add_normalized_score(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby("timestamp")["score"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    out["score_z"] = ((out.score - mean) / std).fillna(0)
    return out


def stats(series: pd.Series) -> dict:
    series = series.sort_index()
    equity = (1 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "sessions": int(len(series)),
        "active_sessions": int((series != 0).sum()),
        "mean_bps": float(series.mean() * 10_000),
        "win_rate_all_sessions": float((series > 0).mean()),
        "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "cumulative_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "without_best5_mean_bps": float(series.nsmallest(max(1, len(series) - 5)).mean() * 10_000),
    }


def evaluate_baseline(frame: pd.DataFrame, dates: list) -> dict:
    test = frame[frame.timestamp.isin(dates)]
    top1 = test[test["rank"] == 1].set_index("timestamp").label_next_open_net.reindex(dates)
    top10 = test.groupby("timestamp").label_next_open_net.mean().reindex(dates)
    return {"top1": stats(top1), "top10": stats(top10)}


def main() -> int:
    frame = add_normalized_score(pd.read_parquet(PRED))
    dates = list(pd.Index(frame.timestamp.unique()).sort_values())
    predictions, blocks = [], []
    for start in range(LOOKBACK, len(dates), HOLD):
        history_dates = dates[start - LOOKBACK:start]
        hold_dates = dates[start:min(start + HOLD, len(dates))]
        history = frame[frame.timestamp.isin(history_dates)]
        hold = frame[frame.timestamp.isin(hold_dates)].copy()
        calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
        calibrator.fit(history.score_z.to_numpy(), history.label_next_open_net.to_numpy())
        hold["expected_net_return"] = calibrator.predict(hold.score_z.to_numpy())
        hold["trade"] = hold.expected_net_return > 0
        hold["block"] = len(blocks) + 1
        predictions.append(hold)
        blocks.append({
            "block": len(blocks) + 1,
            "history_start": str(pd.Timestamp(history_dates[0]).date()),
            "history_end": str(pd.Timestamp(history_dates[-1]).date()),
            "hold_start": str(pd.Timestamp(hold_dates[0]).date()),
            "hold_end": str(pd.Timestamp(hold_dates[-1]).date()),
            "predicted_positive_rate": float(hold.trade.mean()),
            "expected_return_min_bps": float(hold.expected_net_return.min() * 10_000),
            "expected_return_max_bps": float(hold.expected_net_return.max() * 10_000),
        })
    calibrated = pd.concat(predictions, ignore_index=True)
    test_dates = dates[LOOKBACK:]

    top1_rows = calibrated[calibrated["rank"] == 1].set_index("timestamp")
    top1 = top1_rows.label_next_open_net.where(top1_rows.trade, 0).reindex(test_dates, fill_value=0)

    traded = calibrated[calibrated.trade]
    top10 = traded.groupby("timestamp").label_next_open_net.mean().reindex(test_dates, fill_value=0)
    traded_count = traded.groupby("timestamp").size().reindex(test_dates, fill_value=0)
    report = {
        "protocol": "trailing 126 sessions calibrate; frozen 21 sessions",
        "calibrator": "sklearn IsotonicRegression on within-day Top10 score z-score",
        "trade_rule": "expected net return > 0 (equivalent to expected gross return > 20 bps)",
        "baseline": evaluate_baseline(frame, test_dates),
        "calibrated": {"top1": stats(top1), "top10": stats(top10)},
        "average_positions_when_active": float(traded_count[traded_count > 0].mean()),
        "zero_position_sessions": int((traded_count == 0).sum()),
        "blocks": blocks,
    }
    base = report["baseline"]
    candidate = report["calibrated"]
    improved = all([
        candidate["top1"]["mean_bps"] > base["top1"]["mean_bps"],
        candidate["top1"]["sharpe"] > base["top1"]["sharpe"],
        candidate["top10"]["mean_bps"] > base["top10"]["mean_bps"],
        candidate["top10"]["sharpe"] > base["top10"]["sharpe"],
    ])
    report["decision"] = "accept" if improved else "reject"
    calibrated.to_parquet(REPORT_DIR / "corrected_return_calibrator_predictions.parquet", index=False)
    pd.DataFrame(blocks).to_csv(REPORT_DIR / "corrected_return_calibrator_blocks.csv", index=False)
    (REPORT_DIR / "corrected_return_calibrator.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
