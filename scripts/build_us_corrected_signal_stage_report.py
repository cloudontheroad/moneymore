"""Consolidate the corrected cross-sectional signal experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"

MODELS = {
    "old_daily_centered": "corrected_daily_centered_ranker.json",
    "cross_section_varying": "corrected_cs_varying_ranker.json",
    "clip_5pct": "corrected_cs_varying_clip05_ranker.json",
    "clip_10pct": "corrected_cs_varying_clip10_ranker.json",
    "clip_15pct": "corrected_cs_varying_clip15_ranker.json",
}


def main() -> int:
    rows = []
    for experiment, filename in MODELS.items():
        payload = json.loads((MODEL_DIR / filename).read_text(encoding="utf-8"))
        top1 = payload["validation_rank_slot_metrics"][0]
        rows.append({
            "experiment": experiment,
            "trained_features": payload["trained_feature_count"],
            "target_clip_abs": payload.get("target_clip_abs"),
            "top1_mean_bps": top1["mean_net_bps"],
            "top1_win_rate": top1["win_rate"],
            "top1_sharpe": top1["sharpe"],
            "top1_cumulative_return": top1["cumulative_return"],
            "top1_max_drawdown": top1["max_drawdown"],
            "top10_mean_bps": payload["validation_top_n_mean_net"] * 10_000,
            "top10_win_rate": payload["validation_top_n_win_rate"],
            "top10_sharpe": payload["validation_top_n_sharpe"],
            "top10_max_drawdown": payload["validation_top_n_max_drawdown"],
            "trimmed_mean_bps": payload["validation_5pct_trimmed_mean_net"] * 10_000,
            "without_best5_mean_bps": payload["validation_mean_without_best_5_nights"] * 10_000,
            "positive_top1_folds": payload["validation_positive_top1_folds"],
            "worst_top1_fold_mean_bps": payload["validation_worst_fold_top1_mean_bps"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT_DIR / "corrected_signal_stage_experiments.csv", index=False)
    records = json.loads(frame.to_json(orient="records"))
    winner = next(row for row in records if row["experiment"] == "clip_10pct")
    report = {
        "tested": len(frame),
        "adopted_package": [
            "remove cross-section-constant predictors",
            "clip daily-centered training target to +/-10%",
        ],
        "winner": winner,
        "decision": "research baseline accepted; live deployment remains blocked",
        "reason": (
            "Best joint Top1/Top10 result. Five-percent clipping sacrifices Top1; "
            "fifteen-percent clipping is dominated by ten-percent clipping."
        ),
        "remaining_risks": [
            "top1 maximum drawdown remains high",
            "one of four 63-session Top1 folds is negative",
            "mean without the best five Top10 nights is approximately zero",
        ],
        "experiments": records,
    }
    (REPORT_DIR / "corrected_signal_stage_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
