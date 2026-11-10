"""Point-in-time rolling-IC ensemble for screened standard daily factors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "data/us_overnight_corrected/features/standard_daily"
OUT = ROOT / "state/us-overnight-rolling-ic"


def metrics(selected: pd.DataFrame) -> dict[str, object]:
    rows = []
    for rank, group in selected.groupby("rank"):
        returns = group.sort_values("timestamp")["label_next_open_net"]
        equity = (1 + returns).cumprod()
        rows.append({
            "rank": int(rank), "mean_net_bps": float(returns.mean() * 10_000),
            "win_rate": float((returns > 0).mean()),
            "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
            "cumulative": float(equity.iloc[-1] - 1),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
        })
    slots = pd.DataFrame(rows).sort_values("rank")
    top10 = selected.groupby("timestamp")["label_next_open_net"].mean()
    means = slots["mean_net_bps"].to_numpy()
    return {
        "slots": slots.to_dict("records"),
        "violations": int(((means[:-1] - means[1:]) <= 0).sum()),
        "top10_mean_bps": float(top10.mean() * 10_000),
        "top10_win_rate": float((top10 > 0).mean()),
        "top10_sharpe": float(top10.mean() / top10.std(ddof=1) * np.sqrt(252)),
        "top10_cumulative": float((1 + top10).prod() - 1),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(FEATURE_ROOT / "panel.parquet")
    features = json.loads((FEATURE_ROOT / "selected_features.json").read_text(encoding="utf-8"))
    dates = sorted(panel["timestamp"].unique())
    validation = dates[-126:]
    results = []
    for window in (21, 42, 63, 126):
        predictions = []
        for date in validation:
            position = dates.index(date)
            history_dates = dates[max(0, position - window):position]
            history = panel.loc[panel["timestamp"].isin(history_dates)]
            daily_ics = []
            for _, day in history.groupby("timestamp"):
                daily_ics.append(day[features].corrwith(day["label_next_open_net"], method="spearman"))
            weights = pd.DataFrame(daily_ics).mean().fillna(0)
            weights = weights / max(weights.abs().sum(), 1e-12)
            day = panel.loc[panel["timestamp"] == date, [
                "timestamp", "symbol", "label_next_open_net", *features
            ]].copy()
            day["score"] = day[features].fillna(0).to_numpy() @ weights.to_numpy()
            day["rank"] = day["score"].rank(ascending=False, method="first")
            predictions.append(day.loc[day["rank"] <= 10])
        result = {"window": window, **metrics(pd.concat(predictions, ignore_index=True))}
        results.append(result)
        print(json.dumps(result), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
