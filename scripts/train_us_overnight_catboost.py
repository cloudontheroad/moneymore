"""Qlib CatBoost Alpha360 rolling walk-forward baseline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_open_to_open/features/alpha360/panel.parquet"
FEATURES = ROOT / "data/us_overnight_open_to_open/features/alpha360/features.json"
OUT = ROOT / "state/us-overnight-open2open-v1/models"


def metrics(returns: pd.Series) -> dict[str, float | int]:
    returns = returns.dropna().astype(float)
    equity = (1 + returns).cumprod()
    return {
        "sessions": len(returns),
        "mean_net_bps": float(returns.mean() * 10_000),
        "win_rate": float((returns > 0).mean()),
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "cumulative_return": float(equity.iloc[-1] - 1),
    }


def target(frame: pd.DataFrame) -> pd.Series:
    grouped = frame.groupby("timestamp")["label_next_open_net"]
    return (grouped.rank(pct=True, method="average") - 0.5) * 3.46


def model() -> CatBoostRegressor:
    # Microsoft Qlib's published CatBoost Alpha360 configuration.
    return CatBoostRegressor(
        loss_function="RMSE", learning_rate=0.0421, subsample=0.8789,
        depth=6, num_leaves=100, grow_policy="Lossguide",
        bootstrap_type="Poisson", iterations=1000,
        task_type="GPU", devices="0", random_seed=42,
        verbose=False, allow_writing_files=False,
    )


def main() -> int:
    features = json.loads(FEATURES.read_text(encoding="utf-8"))
    panel = pd.read_parquet(PANEL).dropna(subset=["label_next_open_net"]).copy()
    panel = panel.loc[panel["label_next_open_net"].abs() <= 0.5]
    dates = sorted(panel["timestamp"].unique())
    validation_dates = dates[-126:]
    predictions: list[pd.DataFrame] = []
    fitted: CatBoostRegressor | None = None
    for offset, date in enumerate(validation_dates):
        if fitted is None or offset % 5 == 0:
            position = dates.index(date)
            train_dates = dates[max(0, position - 504):position]
            inner_valid = set(train_dates[-63:])
            train = panel.loc[panel["timestamp"].isin(train_dates)]
            fit_mask = ~train["timestamp"].isin(inner_valid)
            fitted = model()
            fitted.fit(
                train.loc[fit_mask, features], target(train).loc[fit_mask],
                eval_set=(train.loc[~fit_mask, features], target(train).loc[~fit_mask]),
                early_stopping_rounds=50, use_best_model=True, verbose=False,
            )
        day = panel.loc[panel["timestamp"] == date, [
            "timestamp", "symbol", "label_next_open_net", *features
        ]].copy()
        day["score"] = fitted.predict(day[features])
        predictions.append(day[["timestamp", "symbol", "label_next_open_net", "score"]])
    result = pd.concat(predictions, ignore_index=True)
    result["rank"] = result.groupby("timestamp")["score"].rank(method="first", ascending=False)
    selected = result.loc[result["rank"] <= 10].copy()
    slots = []
    for rank, group in selected.groupby("rank"):
        slots.append({"rank": int(rank), **metrics(group.sort_values("timestamp")["label_next_open_net"])})
    slot_metrics = pd.DataFrame(slots).sort_values("rank")
    top10 = selected.groupby("timestamp")["label_next_open_net"].mean().sort_index()
    universe = result.groupby("timestamp")["label_next_open_net"].mean().sort_index()
    folds = []
    top1 = selected.loc[selected["rank"] == 1].set_index("timestamp")["label_next_open_net"]
    for number, start in enumerate((0, 63), 1):
        fold_dates = list(top10.index[start:start + 63])
        folds.append({
            "fold": number,
            "top1": metrics(top1.reindex(fold_dates)),
            "top10": metrics(top10.reindex(fold_dates)),
        })
    means = slot_metrics["mean_net_bps"].to_numpy(float)
    summary = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model": "Microsoft Qlib CatBoost Alpha360 configuration",
        "source": "examples/benchmarks/CatBoost/workflow_config_catboost_Alpha360.yaml",
        "features": len(features), "effective_features": 300,
        "target": "Qlib CSRankNorm of open(t)-to-open(t+1) net return",
        "rolling_train_sessions": 504, "inner_validation_sessions": 63,
        "validation_sessions": 126, "retrain_every": 5,
        "top10": metrics(top10),
        "relative_edge_bps": float((top10 - universe).mean() * 10_000),
        "rank_slots": slot_metrics.to_dict("records"),
        "monotonic_violations": int((means[:-1] - means[1:] <= 0).sum()),
        "folds": folds,
        "real_orders_enabled": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qlib_alpha360_catboost.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    slot_metrics.to_csv(OUT / "qlib_alpha360_catboost_rank_slots.csv", index=False)
    selected.to_parquet(OUT / "qlib_alpha360_catboost_predictions.parquet", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
