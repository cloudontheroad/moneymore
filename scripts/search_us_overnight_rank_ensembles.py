"""Search bundle-aware rank ensembles from saved walk-forward candidates."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data/us_overnight_paper_only/models"
SINGLE_DIR = BASE_DIR / "newbaseline_single_search"
OUT = ROOT / "data/us_overnight_corrected/reports"
MODELS = {
    "champion": BASE_DIR / "bundle_right_market_options_calendar_selected_predictions.parquet",
    "mcw1": SINGLE_DIR / "single_min_child_weight-1p0_selected_predictions.parquet",
    "half63": SINGLE_DIR / "single_weight_half_life-63_selected_predictions.parquet",
    "path1": SINGLE_DIR / "single_path_smooth-1p0_selected_predictions.parquet",
    "leaves25": SINGLE_DIR / "single_num_leaves-25_selected_predictions.parquet",
    "top1w5": SINGLE_DIR / "single_top1_weight-5p0_selected_predictions.parquet",
    "mcs275": SINGLE_DIR / "single_min_child_samples-275_selected_predictions.parquet",
}
BASE = {"top1_mean_bps": 39.25277601236869, "top1_sharpe": 1.1322192720470832,
        "top10_mean_bps": -5.513086138365887, "top10_sharpe": -0.3414994407893151}


def load() -> pd.DataFrame:
    parts = []
    for name, path in MODELS.items():
        frame = pd.read_parquet(path)[["timestamp", "symbol", "label_next_open_net", "rank"]].copy()
        frame = frame.rename(columns={"rank": name, "label_next_open_net": f"label_{name}"})
        parts.append(frame)
    merged = parts[0]
    for frame in parts[1:]:
        merged = merged.merge(frame, on=["timestamp", "symbol"], how="outer")
    labels = [column for column in merged if column.startswith("label_")]
    merged["label_next_open_net"] = merged[labels].bfill(axis=1).iloc[:, 0]
    return merged.drop(columns=labels)


def select(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    score = 0.0
    for name, weight in weights.items():
        score = score + weight * frame[name].fillna(11.0)
    work = frame[["timestamp", "symbol", "label_next_open_net"]].copy()
    work["ensemble_rank_score"] = score
    work["rank"] = work.groupby("timestamp")["ensemble_rank_score"].rank(method="first")
    return work.loc[work["rank"] <= 10].copy()


def evaluate(frame: pd.DataFrame, weights: dict[str, float]) -> dict:
    selected = select(frame, weights)
    daily = selected.groupby("timestamp")["label_next_open_net"].mean()
    top1 = selected.loc[selected["rank"] == 1].set_index("timestamp")["label_next_open_net"]
    def stats(series: pd.Series, prefix: str) -> dict:
        equity = (1 + series).cumprod()
        return {f"{prefix}_mean_bps": float(series.mean() * 10_000),
                f"{prefix}_sharpe": float(series.mean() / series.std(ddof=1) * np.sqrt(252)),
                f"{prefix}_win_rate": float((series > 0).mean()),
                f"{prefix}_max_drawdown": float((equity / equity.cummax() - 1).min())}
    return {**stats(top1, "top1"), **stats(daily, "top10")}


def main() -> int:
    frame = load(); rows = []
    names = list(MODELS)
    for left, right in itertools.combinations(names, 2):
        for left_weight in (0.25, 0.5, 0.75):
            weights = {left: left_weight, right: 1 - left_weight}
            rows.append({"members": "+".join(weights), "weights": json.dumps(weights), **evaluate(frame, weights)})
    for members in itertools.combinations(names, 3):
        weights = {name: 1 / 3 for name in members}
        rows.append({"members": "+".join(members), "weights": json.dumps(weights), **evaluate(frame, weights)})
    # Fine-grained stabilised blend: incumbent plus the three complementary
    # Top1 specialists, on a predeclared 0.1 simplex grid.
    focus = ["champion", "mcw1", "half63", "top1w5"]
    for units in itertools.product(range(11), repeat=4):
        if sum(units) != 10 or sum(value > 0 for value in units) < 2:
            continue
        weights = {name: value / 10 for name, value in zip(focus, units) if value > 0}
        rows.append({"members": "+".join(weights), "weights": json.dumps(weights), **evaluate(frame, weights)})
    result = pd.DataFrame(rows)
    result["joint_improvement"] = (
        (result.top1_mean_bps > BASE["top1_mean_bps"]) & (result.top1_sharpe > BASE["top1_sharpe"])
        & (result.top10_mean_bps > BASE["top10_mean_bps"]) & (result.top10_sharpe > BASE["top10_sharpe"])
    )
    result["balanced_delta"] = (
        (result.top1_mean_bps - BASE["top1_mean_bps"]) / 40
        + (result.top1_sharpe - BASE["top1_sharpe"])
        + (result.top10_mean_bps - BASE["top10_mean_bps"]) / 20
        + (result.top10_sharpe - BASE["top10_sharpe"])
    )
    result["both_folds_positive"] = False
    result["worst_fold_top1_bps"] = np.nan
    result["worst_fold_top10_bps"] = np.nan
    for index in result.index[result["joint_improvement"]]:
        weights = json.loads(result.at[index, "weights"])
        candidate = select(frame, weights)
        dates = sorted(candidate["timestamp"].unique())
        top1_means, top10_means = [], []
        for fold_dates in (dates[:63], dates[63:126]):
            fold = candidate.loc[candidate["timestamp"].isin(fold_dates)]
            top1_means.append(float(fold.loc[fold["rank"] == 1, "label_next_open_net"].mean() * 10_000))
            top10_means.append(float(fold.groupby("timestamp")["label_next_open_net"].mean().mean() * 10_000))
        result.at[index, "worst_fold_top1_bps"] = min(top1_means)
        result.at[index, "worst_fold_top10_bps"] = min(top10_means)
        result.at[index, "both_folds_positive"] = min(top1_means) > 0 and min(top10_means) > 0
    result = result.sort_values(["both_folds_positive", "joint_improvement", "balanced_delta"], ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "parameter_rank_ensemble_search.csv", index=False)
    winner = result.iloc[0].to_dict()
    winner_weights = json.loads(winner["weights"])
    winner_selected = select(frame, winner_weights)
    winner_selected.to_parquet(OUT / "parameter_rank_ensemble_winner_predictions.parquet", index=False)
    dates = sorted(winner_selected["timestamp"].unique())
    fold_rows = []
    for fold_number, fold_dates in enumerate((dates[:63], dates[63:126]), 1):
        fold_frame = winner_selected.loc[winner_selected["timestamp"].isin(fold_dates)]
        daily = fold_frame.groupby("timestamp")["label_next_open_net"].mean()
        top1 = fold_frame.loc[fold_frame["rank"] == 1].set_index("timestamp")["label_next_open_net"]
        fold_rows.append({"fold": fold_number, "sessions": len(fold_dates),
                          "top1_mean_bps": float(top1.mean() * 10_000),
                          "top1_sharpe": float(top1.mean() / top1.std(ddof=1) * np.sqrt(252)),
                          "top10_mean_bps": float(daily.mean() * 10_000),
                          "top10_sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))})
    winner["fold_metrics"] = fold_rows
    (OUT / "parameter_rank_ensemble_winner.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    print(json.dumps({"tested": len(result), "joint_improvements": int(result.joint_improvement.sum()),
                      "fold_robust_improvements": int(result.both_folds_positive.sum()), "winner": winner}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
