"""Train an auditable linear cross-sectional overnight ranker from cached daily bars."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.features import (
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    daily_stock_features,
)

ROOT = Path(__file__).resolve().parents[1]


def _panel(daily_dir: Path, cost_bps: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(daily_dir.glob("US_*.parquet")):
        bars = pd.read_parquet(path)
        if bars.empty:
            continue
        features = daily_stock_features(bars, cost_bps=cost_bps)
        features["symbol"] = str(bars.iloc[0]["symbol"])
        parts.append(features[["timestamp", "symbol", *FEATURE_COLUMNS, "label_next_open_net"]])
    if not parts:
        raise FileNotFoundError("no cached US daily histories")
    return pd.concat(parts, ignore_index=True)


def _winsorize_by_date(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    for column in FEATURE_COLUMNS:
        grouped = result.groupby("timestamp")[column]
        low = grouped.transform(lambda values: values.quantile(0.01))
        high = grouped.transform(lambda values: values.quantile(0.99))
        result[column] = result[column].clip(low, high)
    return result


def _cross_sectionalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Turn raw predictors into same-day percentile ranks for robust stock selection."""
    result = panel.copy()
    for column in FEATURE_COLUMNS:
        result[column] = (
            result.groupby("timestamp")[column].rank(pct=True, method="average") - 0.5
        )
    return result


def _fit_ridge(frame: pd.DataFrame, ridge: float = 10.0) -> dict[str, object]:
    x = frame[FEATURE_COLUMNS].to_numpy(float)
    y = frame["label_next_open_net"].to_numpy(float)
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[scale < 1e-12] = 1.0
    z = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.linalg.solve(z.T @ z + ridge * np.eye(z.shape[1]), z.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def _predict(frame: pd.DataFrame, model: dict[str, object]) -> np.ndarray:
    x = frame[FEATURE_COLUMNS].to_numpy(float)
    z = np.nan_to_num(
        (x - np.asarray(model["mean"])) / np.asarray(model["scale"]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return z @ np.asarray(model["weights"])


def main() -> int:
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    output = ROOT / config.output_root
    panel = _cross_sectionalize(
        _winsorize_by_date(_panel(output / "daily", config.round_trip_cost_bps))
    )
    mature = panel.dropna(subset=["label_next_open_net"]).copy()
    dates = sorted(mature["timestamp"].unique())
    if len(dates) < 252:
        raise RuntimeError(f"insufficient history: {len(dates)} sessions")
    validation_dates = set(dates[-126:])
    train = mature.loc[~mature["timestamp"].isin(validation_dates)]
    validation = mature.loc[mature["timestamp"].isin(validation_dates)].copy()
    train = train.copy()
    train["label_next_open_net"] -= train.groupby("timestamp")[
        "label_next_open_net"
    ].transform("mean")
    initial = _fit_ridge(train)
    validation["score"] = _predict(validation, initial)
    validation["rank"] = validation.groupby("timestamp")["score"].rank(
        method="first", ascending=False
    )
    selected = validation.loc[validation["rank"] <= config.top_n]
    daily_top = selected.groupby("timestamp")["label_next_open_net"].mean()
    daily_all = validation.groupby("timestamp")["label_next_open_net"].mean()
    mature_fit = mature.copy()
    mature_fit["label_next_open_net"] -= mature_fit.groupby("timestamp")[
        "label_next_open_net"
    ].transform("mean")
    final_model = _fit_ridge(mature_fit)
    latest_date = panel["timestamp"].max()
    latest = panel.loc[panel["timestamp"] == latest_date].copy()
    latest["score"] = _predict(latest, final_model)
    latest = latest.sort_values(["score", "symbol"], ascending=[False, True]).head(config.top_n)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "features": FEATURE_COLUMNS,
        "feature_library_size": len(ALL_FEATURE_COLUMNS),
        "feature_sources": [
            "microsoft/qlib Alpha158",
            "AI4Finance-Foundation/FinRL default indicators",
            "MoneyMore overnight-specific features",
        ],
        "mean": np.asarray(final_model["mean"]).tolist(),
        "scale": np.asarray(final_model["scale"]).tolist(),
        "weights": np.asarray(final_model["weights"]).tolist(),
        "ridge": 10.0,
        "transform": "same_day_percentile_rank_centered",
        "cost_bps": config.round_trip_cost_bps,
        "train_rows": len(mature),
        "validation_sessions": len(validation_dates),
        "validation_top_n_mean_net": float(daily_top.mean()),
        "validation_all_mean_net": float(daily_all.mean()),
        "latest_feature_date": str(latest_date),
        "latest_preview_not_tradeable": latest[["symbol", "score"]].to_dict("records"),
        "real_orders_enabled": False,
    }
    (model_dir / "linear_ranker.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
