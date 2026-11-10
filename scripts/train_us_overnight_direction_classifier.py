"""Walk-forward probability-of-profit classifier for clean daily factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarize(selected: pd.DataFrame) -> dict[str, object]:
    slots = []
    for rank, group in selected.groupby("rank"):
        returns = group["label_next_open_net"]
        equity = (1 + returns).cumprod()
        slots.append({
            "rank": int(rank), "net_win_rate": float((returns > 0).mean()),
            "gross_win_rate": float((returns > -0.002).mean()),
            "mean_net_bps": float(returns.mean() * 10_000),
            "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
            "cumulative": float(equity.iloc[-1] - 1),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
        })
    frame = pd.DataFrame(slots).sort_values("rank")
    top10 = selected.groupby("timestamp")["label_next_open_net"].mean()
    means = frame["mean_net_bps"].to_numpy()
    return {
        "slots": frame.to_dict("records"),
        "violations": int(((means[:-1] - means[1:]) <= 0).sum()),
        "top10_net_win_rate": float((top10 > 0).mean()),
        "top10_mean_net_bps": float(top10.mean() * 10_000),
        "top10_sharpe": float(top10.mean() / top10.std(ddof=1) * np.sqrt(252)),
        "top10_cumulative": float((1 + top10).prod() - 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--target", choices=["gross_up", "net_profit"], required=True)
    args = parser.parse_args()
    base = ROOT / "data/us_overnight_corrected/features/standard_daily_clean"
    panel = pd.read_parquet(base / "panel.parquet").dropna(subset=["label_next_open_net"])
    features = json.loads((base / "selected_features.json").read_text(encoding="utf-8"))
    dates = sorted(panel["timestamp"].unique())
    validation = dates[-126:]
    predictions = []
    model = None
    for offset, date in enumerate(validation):
        if model is None or offset % 21 == 0:
            position = dates.index(date)
            train_dates = dates[max(0, position - args.window):position]
            train = panel.loc[panel["timestamp"].isin(train_dates)]
            threshold = -0.002 if args.target == "gross_up" else 0.0
            target = (train["label_next_open_net"] > threshold).astype(int)
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=350, learning_rate=0.025,
                num_leaves=15, min_child_samples=400, min_child_weight=50,
                colsample_bytree=0.75, reg_lambda=3, extra_trees=True,
                max_bin=63, class_weight="balanced", random_state=42,
                n_jobs=-1, verbosity=-1,
            )
            model.fit(train[features], target)
        day = panel.loc[panel["timestamp"] == date, [
            "timestamp", "symbol", "label_next_open_net", *features
        ]].copy()
        day["score"] = model.predict_proba(day[features])[:, 1]
        day["rank"] = day["score"].rank(ascending=False, method="first")
        predictions.append(day.loc[day["rank"] <= 10])
    selected = pd.concat(predictions, ignore_index=True)
    result = {"window": args.window, "target": args.target, **summarize(selected)}
    out = ROOT / "state/us-overnight-direction-classifier"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.target}_{args.window}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
