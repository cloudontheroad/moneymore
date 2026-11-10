"""Official Qlib static train/valid/test DoubleEnsemble protocol on Alpha360."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from train_us_overnight_catboost import metrics
from train_us_overnight_doubleensemble import (
    FEATURES, OUT, PANEL, RollingDataset, new_model, predict, qlib_frame,
)


def main() -> int:
    features = json.loads(FEATURES.read_text(encoding="utf-8"))
    panel = pd.read_parquet(PANEL).dropna(subset=["label_next_open_net"]).copy()
    panel = panel.loc[panel["label_next_open_net"].abs() <= 0.5]
    dates = sorted(panel["timestamp"].unique())
    test_dates = set(dates[-126:])
    valid_dates = set(dates[-189:-126])
    train_dates = set(dates[:-189])
    train = panel.loc[panel["timestamp"].isin(train_dates)]
    valid = panel.loc[panel["timestamp"].isin(valid_dates)]
    test = panel.loc[panel["timestamp"].isin(test_dates)].copy()
    np.random.seed(42)
    fitted = new_model()
    fitted.fit(RollingDataset(qlib_frame(train, features), qlib_frame(valid, features)))
    test["score"] = predict(fitted, test[features])
    test["rank"] = test.groupby("timestamp")["score"].rank(method="first", ascending=False)
    selected = test.loc[test["rank"] <= 10].copy()
    slots = []
    cumulative = []
    for rank, group in selected.groupby("rank"):
        slots.append({"rank": int(rank), **metrics(group.sort_values("timestamp")["label_next_open_net"])})
    slot_frame = pd.DataFrame(slots).sort_values("rank")
    for n in range(1, 11):
        daily = selected.loc[selected["rank"] <= n].groupby("timestamp")["label_next_open_net"].mean().sort_index()
        cumulative.append({"top_n": n, **metrics(daily)})
    topn = pd.DataFrame(cumulative)
    folds = []
    ordered = sorted(test_dates)
    for number, fold_dates in enumerate((ordered[:63], ordered[63:]), 1):
        row = {"fold": number}
        for n in (1, 5, 10):
            daily = selected.loc[
                selected["timestamp"].isin(fold_dates) & (selected["rank"] <= n)
            ].groupby("timestamp")["label_next_open_net"].mean()
            row[f"top{n}"] = metrics(daily)
        folds.append(row)
    means = slot_frame["mean_net_bps"].to_numpy(float)
    summary = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model": "Microsoft Qlib DEnsembleModel",
        "protocol": "official-style static chronological train/valid/test",
        "comparison_warning": "not directly comparable to 5-session rolling LightGBM/CatBoost",
        "train_sessions": len(train_dates), "valid_sessions": len(valid_dates),
        "test_sessions": len(test_dates), "features": len(features), "effective_features": 300,
        "rank_slots": slot_frame.to_dict("records"),
        "cumulative_top_n": topn.to_dict("records"),
        "monotonic_violations": int((means[:-1] - means[1:] <= 0).sum()),
        "folds": folds, "real_orders_enabled": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qlib_alpha360_doubleensemble_static.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    slot_frame.to_csv(OUT / "qlib_alpha360_doubleensemble_static_rank_slots.csv", index=False)
    topn.to_csv(OUT / "qlib_alpha360_doubleensemble_static_topn.csv", index=False)
    selected.to_parquet(OUT / "qlib_alpha360_doubleensemble_static_predictions.parquet", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
