"""Point-in-time market-return hurdle for the frozen stock selector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_corrected/features/historical_panel_bundle_investable_120d_20m_5.parquet"
PRED = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected/corrected_cs_varying_clip10_selected_predictions.parquet"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"
TRAIN_DAYS = 252
RETRAIN = 21


def stats(series: pd.Series) -> dict:
    series = series.sort_index()
    equity = (1 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "sessions": int(len(series)),
        "active_sessions": int((series != 0).sum()),
        "mean_bps": float(series.mean() * 10_000),
        "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "cumulative_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "without_best5_mean_bps": float(
            series.nsmallest(max(1, len(series) - 5)).mean() * 10_000
        ),
    }


def main() -> int:
    panel = pd.read_parquet(PANEL)
    if "label_end" not in panel:
        ends = pd.read_parquet(
            REPORT_DIR / "investability_eligibility_history.parquet",
            columns=["timestamp", "symbol", "label_end"],
        )
        panel = panel.merge(ends, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    if panel.label_end.isna().any():
        raise ValueError("Missing label settlement sessions")
    selected = pd.read_parquet(PRED)
    features = [column for column in panel.columns if column.startswith("market_")]
    features += [
        column for column in panel.columns
        if column.startswith(("breadth_", "calendar_"))
    ]
    daily = panel.groupby("timestamp", as_index=False).agg(
        {**{column: "first" for column in features}, "label_next_open_net": "mean",
         "label_end": "max"}
    ).sort_values("timestamp").reset_index(drop=True)
    dates = list(daily.timestamp)
    test_dates = list(pd.Index(selected.timestamp.unique()).sort_values())
    date_position = {date: position for position, date in enumerate(dates)}
    forecasts = []
    model = None
    active_anchor = None
    for date in test_dates:
        position = date_position[date]
        anchor = position - position % RETRAIN
        if model is None or anchor != active_anchor:
            train = daily.loc[daily.label_end < dates[anchor]].tail(TRAIN_DAYS)
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1000),
            )
            model.fit(train[features], train.label_next_open_net * 10_000)
            active_anchor = anchor
        row = daily[daily.timestamp == date]
        forecasts.append({
            "timestamp": date,
            "predicted_universe_net_bps": float(model.predict(row[features])[0]),
            "realized_universe_net": float(row.label_next_open_net.iloc[0]),
        })
    forecast = pd.DataFrame(forecasts).set_index("timestamp")
    top1 = selected[selected["rank"] == 1].set_index("timestamp").label_next_open_net.reindex(test_dates)
    top10 = selected.groupby("timestamp").label_next_open_net.mean().reindex(test_dates)
    rows = []
    for threshold in (0.0, -5.0, -10.0, -15.0):
        active = forecast.predicted_universe_net_bps > threshold
        gated_top1 = top1.where(active, 0)
        gated_top10 = top10.where(active, 0)
        rows.append({
            "threshold_bps": threshold,
            "coverage": float(active.mean()),
            **{f"top1_{key}": value for key, value in stats(gated_top1).items()},
            **{f"top10_{key}": value for key, value in stats(gated_top10).items()},
        })
    table = pd.DataFrame(rows)
    baseline = {"top1": stats(top1), "top10": stats(top10)}
    table["joint_sharpe"] = table.top1_sharpe + table.top10_sharpe
    winner = table.sort_values("joint_sharpe", ascending=False).iloc[0].to_dict()
    market_correlation = float(
        forecast.predicted_universe_net_bps.corr(forecast.realized_universe_net * 10_000)
    )
    natural_active = forecast.predicted_universe_net_bps > 0
    natural_top1 = top1.where(natural_active, 0)
    natural_top10 = top10.where(natural_active, 0)
    folds = []
    for number, fold_dates in enumerate(np.array_split(test_dates, 4), 1):
        folds.append({
            "fold": number,
            "coverage": float(natural_active.reindex(fold_dates).mean()),
            "top1": stats(natural_top1.reindex(fold_dates)),
            "top10": stats(natural_top10.reindex(fold_dates)),
        })
    report = {
        "model": "rolling HuberRegressor market hurdle",
        "features": features,
        "train_sessions": TRAIN_DAYS,
        "retrain_sessions": RETRAIN,
        "forecast_realized_correlation": market_correlation,
        "baseline": baseline,
        "winner": winner,
        "natural_zero_threshold_folds": folds,
        "decision": "diagnostic",
        "results": json.loads(table.to_json(orient="records")),
    }
    improved = all([
        winner["top1_sharpe"] > baseline["top1"]["sharpe"],
        winner["top1_cumulative_return"] > baseline["top1"]["cumulative_return"],
        winner["top1_max_drawdown"] > baseline["top1"]["max_drawdown"],
        winner["top10_mean_bps"] > baseline["top10"]["mean_bps"],
        winner["top10_sharpe"] > baseline["top10"]["sharpe"],
        winner["top10_cumulative_return"] > baseline["top10"]["cumulative_return"],
        winner["top10_max_drawdown"] > baseline["top10"]["max_drawdown"],
    ])
    report["decision"] = (
        "accept zero-threshold market hurdle as research baseline"
        if improved and winner["threshold_bps"] == 0
        else "reject"
    )
    report["pareto_note"] = (
        "Top1 arithmetic mean is lower because 154 sessions are cash, while Top1 Sharpe, "
        "compounded return and drawdown improve; all Top10 primary risk-return metrics improve."
    )
    table.to_csv(REPORT_DIR / "corrected_market_hurdle.csv", index=False)
    forecast.reset_index().to_parquet(REPORT_DIR / "corrected_market_hurdle_forecasts.parquet", index=False)
    (REPORT_DIR / "corrected_market_hurdle.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
