"""Daily-volume capacity and volume-share slippage stress for the corrected strategy."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data/us_overnight_corrected/reports"
PREDICTIONS = REPORTS / "investable_corrected_open2open_ensemble_search_winner.parquet"
ELIGIBILITY = REPORTS / "investability_eligibility_history.parquet"
PARTICIPATION_LIMIT = 0.025
PRICE_IMPACT = 0.10


def metrics(series: pd.Series) -> dict[str, float | int]:
    series = series.dropna().sort_index()
    equity = (1 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "sessions": int(len(series)),
        "mean_bps": float(series.mean() * 10_000),
        "sharpe": float(series.mean() / std * np.sqrt(252)),
        "cumulative_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def main() -> int:
    predictions = pd.read_parquet(PREDICTIONS)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    liquidity = pd.read_parquet(ELIGIBILITY)[
        ["timestamp", "symbol", "median_dollar_volume_20d"]
    ].copy()
    liquidity["timestamp"] = pd.to_datetime(liquidity["timestamp"], utc=True)
    frame = predictions.merge(
        liquidity, on=["timestamp", "symbol"], how="left", validate="one_to_one"
    )
    rows = []
    for top_n in (1, 5, 10):
        selected = frame.loc[frame["rank"] <= top_n].copy()
        for portfolio_notional in (100_000, 1_000_000, 10_000_000):
            order_notional = portfolio_notional / top_n
            selected["participation"] = (
                order_notional / selected["median_dollar_volume_20d"]
            )
            selected["additional_round_trip_slippage"] = (
                2 * PRICE_IMPACT * selected["participation"].clip(upper=PARTICIPATION_LIMIT) ** 2
            )
            selected["stressed_return"] = (
                selected["label"] - selected["additional_round_trip_slippage"]
            )
            daily = selected.groupby("timestamp")["stressed_return"].mean()
            fully_fillable = selected.groupby("timestamp")["participation"].max() <= PARTICIPATION_LIMIT
            rows.append({
                "top_n": top_n,
                "portfolio_notional": portfolio_notional,
                "order_notional": order_notional,
                "median_participation": float(selected["participation"].median()),
                "p95_participation": float(selected["participation"].quantile(0.95)),
                "maximum_participation": float(selected["participation"].max()),
                "fully_fillable_session_rate": float(fully_fillable.mean()),
                "mean_additional_slippage_bps_per_position": float(
                    selected["additional_round_trip_slippage"].mean() * 10_000
                ),
                **metrics(daily),
            })
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "corrected_open2open_capacity_stress.csv", index=False)
    report = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "method": "LEAN-style volume-share stress on trailing 20-day median dollar volume",
        "participation_limit": PARTICIPATION_LIMIT,
        "price_impact": PRICE_IMPACT,
        "base_returns_already_include_bps": 20.0,
        "results": result.to_dict("records"),
        "limitation": "Daily dollar volume cannot prove opening-auction capacity; this is a screening stress, not a fill guarantee.",
    }
    (REPORTS / "corrected_open2open_capacity_stress.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
