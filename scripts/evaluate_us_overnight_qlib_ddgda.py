"""Evaluate the combined Qlib DDG-DA signal on raw open-to-open returns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "mlruns/5/4e584eefd4914687b6b4f47f67051cf3/artifacts/pred.pkl"
PANEL = ROOT / "data/us_overnight_open_to_open/features/alpha360/panel.parquet"
OUT = ROOT / "state/us-overnight-open2open-v1/ddgda"


def metrics(returns: pd.Series) -> dict:
    returns = returns.dropna().astype(float)
    equity = (1.0 + returns).cumprod()
    std = returns.std(ddof=1)
    return {
        "sessions": int(len(returns)),
        "mean_net_bps": float(returns.mean() * 10_000),
        "cumulative_return": float(equity.iloc[-1] - 1),
        "win_rate": float((returns > 0).mean()),
        "annualized_volatility": float(std * np.sqrt(252)),
        "sharpe": float(returns.mean() / std * np.sqrt(252)) if std else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def main() -> int:
    pred = pd.read_pickle(PRED).rename(columns={"score": "prediction"}).reset_index()
    raw = pd.read_parquet(PANEL, columns=["timestamp", "symbol", "label_next_open_net"])
    raw["datetime"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    raw["instrument"] = raw["symbol"]
    joined = pred.merge(raw[["datetime", "instrument", "label_next_open_net"]],
                        on=["datetime", "instrument"], how="inner", validate="one_to_one")
    joined["rank"] = joined.groupby("datetime")["prediction"].rank(method="first", ascending=False).astype(int)
    rank_daily = joined[joined["rank"] <= 10].pivot(index="datetime", columns="rank", values="label_next_open_net")
    rank_metrics = {str(rank): metrics(rank_daily[rank]) for rank in range(1, 11)}
    topn_metrics = {str(n): metrics(rank_daily.loc[:, 1:n].mean(axis=1)) for n in range(1, 11)}
    rank_means = [rank_metrics[str(i)]["mean_net_bps"] for i in range(1, 11)]
    payload = {
        "model": "Qlib DDG-DA / Alpha300 / LightGBM",
        "economic_target": "buy open(t), sell open(t+1), net of 20 bps",
        "period": [str(rank_daily.index.min().date()), str(rank_daily.index.max().date())],
        "rows_joined": int(len(joined)),
        "rank_strategies": rank_metrics,
        "cumulative_top_n": topn_metrics,
        "rank_mean_bps": rank_means,
        "adjacent_monotonic_pairs": int(sum(a >= b for a, b in zip(rank_means, rank_means[1:]))),
        "strictly_monotonic": bool(all(a > b for a, b in zip(rank_means, rank_means[1:]))),
    }
    (OUT / "ddgda_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    joined.to_parquet(OUT / "ddgda_predictions_with_returns.parquet", index=False)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
