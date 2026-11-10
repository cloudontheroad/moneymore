"""Evaluate a pre-registered score-margin abstention gate on a held-out fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/us_overnight_paper_only/models/pareto_00_baseline_replay_selected_predictions.parquet"
OUT = ROOT / "data/us_overnight_paper_only/models/pareto_03_confidence_gate.json"


def metrics(returns: pd.Series, traded: pd.Series) -> dict:
    active = returns.loc[traded]
    strategy = returns.where(traded, 0.0)
    equity = (1 + strategy).cumprod()
    return {
        "sessions": len(strategy), "traded_sessions": int(traded.sum()),
        "coverage": float(traded.mean()), "active_win_rate": float((active > 0).mean()),
        "mean_all_sessions_bps": float(strategy.mean() * 10_000),
        "mean_active_bps": float(active.mean() * 10_000),
        "sharpe_all_sessions": float(strategy.mean() / strategy.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def main() -> int:
    selected = pd.read_parquet(PREDICTIONS).sort_values(["timestamp", "rank"])
    daily_return = selected.groupby("timestamp")["label_next_open_net"].mean()
    score = selected.pivot(index="timestamp", columns="rank", values="score")
    margin = score[1.0] - score[2.0]
    dates = sorted(daily_return.index)
    calibration, test = dates[:63], dates[63:126]
    threshold = float(margin.reindex(calibration).median())
    test_returns = daily_return.reindex(test)
    gated = margin.reindex(test) >= threshold
    payload = {
        "method": "top1_minus_top2_score_margin",
        "threshold_policy": "median fixed on first 63 validation sessions",
        "threshold": threshold,
        "calibration_range": [str(calibration[0]), str(calibration[-1])],
        "test_range": [str(test[0]), str(test[-1])],
        "ungated_test": metrics(test_returns, pd.Series(True, index=test)),
        "gated_test": metrics(test_returns, gated),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
