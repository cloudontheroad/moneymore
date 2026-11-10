"""Blend the accepted signal with its falling-deceleration ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"
MAIN = MODEL_DIR / "corrected_cs_varying_clip10_selected_predictions.parquet"
ABLATION = MODEL_DIR / "corrected_cs_varying_clip10_no_falling_selected_predictions.parquet"


def stats(series: pd.Series) -> dict:
    series = series.sort_index()
    equity = (1 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "mean_bps": float(series.mean() * 10_000),
        "win_rate": float((series > 0).mean()),
        "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "cumulative_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "without_best5_mean_bps": float(series.nsmallest(len(series) - 5).mean() * 10_000),
    }


def evaluate(selected: pd.DataFrame) -> dict:
    top1 = selected[selected["rank"] == 1].set_index("timestamp")["label"]
    top10 = selected.groupby("timestamp")["label"].mean()
    dates = list(top10.index.sort_values())
    folds = []
    for number, fold_dates in enumerate(np.array_split(dates, 4), 1):
        folds.append({
            "fold": number,
            "top1_mean_bps": float(top1.reindex(fold_dates).mean() * 10_000),
            "top10_mean_bps": float(top10.reindex(fold_dates).mean() * 10_000),
        })
    return {"top1": stats(top1), "top10": stats(top10), "folds": folds}


def main() -> int:
    main = pd.read_parquet(MAIN)[["timestamp", "symbol", "label_next_open_net", "rank"]]
    main = main.rename(columns={"rank": "main_rank", "label_next_open_net": "main_label"})
    ablation = pd.read_parquet(ABLATION)[["timestamp", "symbol", "label_next_open_net", "rank"]]
    ablation = ablation.rename(columns={"rank": "ablation_rank", "label_next_open_net": "ablation_label"})
    merged = main.merge(ablation, on=["timestamp", "symbol"], how="outer")
    merged["label"] = merged.main_label.fillna(merged.ablation_label)
    rows, selections = [], {}
    for units in range(11):
        weight = units / 10
        score = (
            weight * merged.main_rank.fillna(11)
            + (1 - weight) * merged.ablation_rank.fillna(11)
        )
        selected = merged[["timestamp", "symbol", "label"]].copy()
        selected["score"] = score
        selected["rank"] = selected.groupby("timestamp")["score"].rank(method="first")
        selected = selected[selected["rank"] <= 10]
        metrics = evaluate(selected)
        rows.append({
            "main_weight": weight,
            **{f"top1_{key}": value for key, value in metrics["top1"].items()},
            **{f"top10_{key}": value for key, value in metrics["top10"].items()},
        })
        selections[weight] = (selected, metrics)
    result = pd.DataFrame(rows)
    baseline = result[result.main_weight == 1.0].iloc[0]
    result["primary_improvement"] = (
        (result.top1_mean_bps > baseline.top1_mean_bps)
        & (result.top1_sharpe > baseline.top1_sharpe)
        & (result.top10_mean_bps > baseline.top10_mean_bps)
        & (result.top10_sharpe > baseline.top10_sharpe)
    )
    winner = result[result.primary_improvement].sort_values(
        ["top1_sharpe", "top10_sharpe"], ascending=False
    ).iloc[0]
    weight = float(winner.main_weight)
    selected, winner_metrics = selections[weight]
    result.to_csv(REPORT_DIR / "corrected_falling_blend_search.csv", index=False)
    selected.to_parquet(REPORT_DIR / "corrected_falling_blend_winner.parquet", index=False)
    report = {
        "tested": len(result),
        "baseline_main_weight": 1.0,
        "winner_main_weight": weight,
        "winner_ablation_weight": 1 - weight,
        "baseline": result[result.main_weight == 1.0].iloc[0].to_dict(),
        "winner": winner.to_dict(),
        "winner_folds": winner_metrics["folds"],
        "decision": "accept as research baseline" if winner.primary_improvement else "reject",
        "live_deployment": False,
    }
    (REPORT_DIR / "corrected_falling_blend_search.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
