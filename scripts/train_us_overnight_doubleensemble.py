"""Qlib's official DoubleEnsemble implementation on the Alpha360 panel."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.contrib.model.double_ensemble import DEnsembleModel

from train_us_overnight_catboost import metrics, target

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_open_to_open/features/alpha360/panel.parquet"
FEATURES = ROOT / "data/us_overnight_open_to_open/features/alpha360/features.json"
OUT = ROOT / "state/us-overnight-open2open-v1/models"


def qlib_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = pd.concat(
        [frame[features].reset_index(drop=True), target(frame).reset_index(drop=True)], axis=1
    )
    values.columns = pd.MultiIndex.from_tuples(
        [("feature", name) for name in features] + [("label", "LABEL0")]
    )
    return values


class RollingDataset:
    def __init__(self, train: pd.DataFrame, valid: pd.DataFrame):
        self.frames = {"train": train, "valid": valid}

    def prepare(self, segments, **kwargs):
        if isinstance(segments, list):
            return tuple(self.frames[name] for name in segments)
        return self.frames[segments]


def new_model() -> DEnsembleModel:
    # Exact public Qlib DoubleEnsemble configuration.  The same model is
    # applied to Alpha360; only the dataset handler differs from the yaml.
    return DEnsembleModel(
        base_model="gbm", loss="mse", num_models=3,
        enable_sr=True, enable_fs=True, alpha1=1, alpha2=1,
        bins_sr=10, bins_fs=5, decay=0.5,
        sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
        sub_weights=[1, 1, 1], epochs=28,
        colsample_bytree=0.8879, learning_rate=0.2,
        subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
        max_depth=8, num_leaves=210, num_threads=20, verbosity=-1,
        seed=42,
    )


def predict(model: DEnsembleModel, frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    for submodel, columns, weight in zip(model.ensemble, model.sub_features, model.sub_weights):
        result += submodel.predict(frame.loc[:, columns]) * weight
    return result / np.sum(model.sub_weights)


def main() -> int:
    features = json.loads(FEATURES.read_text(encoding="utf-8"))
    panel = pd.read_parquet(PANEL).dropna(subset=["label_next_open_net"]).copy()
    panel = panel.loc[panel["label_next_open_net"].abs() <= 0.5]
    dates = sorted(panel["timestamp"].unique())
    validation_dates = dates[-126:]
    predictions: list[pd.DataFrame] = []
    fitted: DEnsembleModel | None = None
    for offset, date in enumerate(validation_dates):
        if fitted is None or offset % 5 == 0:
            position = dates.index(date)
            train_dates = dates[max(0, position - 504):position]
            inner_valid = set(train_dates[-63:])
            history = panel.loc[panel["timestamp"].isin(train_dates)]
            fit = history.loc[~history["timestamp"].isin(inner_valid)]
            valid = history.loc[history["timestamp"].isin(inner_valid)]
            np.random.seed(42)
            fitted = new_model()
            fitted.fit(RollingDataset(qlib_frame(fit, features), qlib_frame(valid, features)))
        day = panel.loc[panel["timestamp"] == date, [
            "timestamp", "symbol", "label_next_open_net", *features
        ]].copy()
        day["score"] = predict(fitted, day[features])
        predictions.append(day[["timestamp", "symbol", "label_next_open_net", "score"]])
        print(f"completed {offset + 1}/126 {date}", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    result["rank"] = result.groupby("timestamp")["score"].rank(method="first", ascending=False)
    selected = result.loc[result["rank"] <= 10].copy()
    slot_rows = []
    for rank, group in selected.groupby("rank"):
        slot_rows.append({"rank": int(rank), **metrics(group.sort_values("timestamp")["label_next_open_net"])})
    slots = pd.DataFrame(slot_rows).sort_values("rank")
    cumulative_rows = []
    for n in range(1, 11):
        daily = selected.loc[selected["rank"] <= n].groupby("timestamp")["label_next_open_net"].mean().sort_index()
        cumulative_rows.append({"top_n": n, **metrics(daily)})
    cumulative = pd.DataFrame(cumulative_rows)
    folds = []
    for number, start in enumerate((0, 63), 1):
        fold_dates = sorted(result["timestamp"].unique())[start:start + 63]
        rows = {"fold": number}
        for n in (1, 5, 10):
            daily = selected.loc[
                selected["timestamp"].isin(fold_dates) & (selected["rank"] <= n)
            ].groupby("timestamp")["label_next_open_net"].mean()
            rows[f"top{n}"] = metrics(daily)
        folds.append(rows)
    means = slots["mean_net_bps"].to_numpy(float)
    summary = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model": "Microsoft Qlib DEnsembleModel",
        "source": "examples/benchmarks/DoubleEnsemble/workflow_config_doubleensemble_Alpha158.yaml",
        "dataset": "Alpha360-compatible", "features": len(features), "effective_features": 300,
        "rolling_train_sessions": 504, "inner_validation_sessions": 63,
        "validation_sessions": 126, "retrain_every": 5,
        "rank_slots": slots.to_dict("records"),
        "cumulative_top_n": cumulative.to_dict("records"),
        "monotonic_violations": int((means[:-1] - means[1:] <= 0).sum()),
        "folds": folds, "real_orders_enabled": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qlib_alpha360_doubleensemble.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    slots.to_csv(OUT / "qlib_alpha360_doubleensemble_rank_slots.csv", index=False)
    cumulative.to_csv(OUT / "qlib_alpha360_doubleensemble_topn.csv", index=False)
    selected.to_parquet(OUT / "qlib_alpha360_doubleensemble_predictions.parquet", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
