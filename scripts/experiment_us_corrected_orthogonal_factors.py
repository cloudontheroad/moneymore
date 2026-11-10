"""Point-in-time cross-sectional factor composites on the corrected target.

The experiment follows the standard equity-factor workflow: cross-sectional
rank normalization, optional size/liquidity/volatility neutralization, rolling
IC screening, and frozen monthly (21-session) weights.  Every test decision is
based exclusively on earlier sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_corrected/features/historical_panel_bundle_investable_120d_20m_5.parquet"
OUT_JSON = ROOT / "data/us_overnight_corrected/reports/corrected_orthogonal_factor_experiments.json"
OUT_CSV = ROOT / "data/us_overnight_corrected/reports/corrected_orthogonal_factor_experiments.csv"
OUT_PRED = ROOT / "data/us_overnight_corrected/reports/corrected_orthogonal_factor_predictions.parquet"
LABEL = "label_next_open_net"
LOOKBACK = 252
TEST_DAYS = 252
REBALANCE = 21
TOP_FEATURES = 20

EXCLUDE = {
    "timestamp", "feature_asof", "symbol", LABEL, "open", "high", "low", "close",
    "volume", "next_open", "next_dividend", "listing_sessions", "eligible",
    # Absolute price-level Qlib fields are not comparable across securities.
    "MA5", "MA20", "MA60", "VMA5", "VMA20", "VWAP0",
}
CONTROLS = ["log_dollar_volume", "realized_vol_20", "distance_ma_200"]


def zscore_by_day(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ranks = frame.groupby("timestamp", sort=False)[columns].rank(pct=True)
    means = ranks.groupby(frame.timestamp, sort=False).transform("mean")
    stds = ranks.groupby(frame.timestamp, sort=False).transform("std").replace(0, np.nan)
    return (ranks - means) / stds


def neutralize_by_day(values: pd.DataFrame, controls: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index, columns=values.columns, dtype="float32")
    for _, index in dates.groupby(dates, sort=False).groups.items():
        y = values.loc[index].to_numpy(dtype=float)
        x = controls.loc[index].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(index)), np.nan_to_num(x)])
        y = np.nan_to_num(y)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ beta
        scale = residual.std(axis=0, ddof=1)
        scale[scale == 0] = 1.0
        out.loc[index] = (residual / scale).astype("float32")
    return out


def performance(selected: pd.DataFrame) -> dict:
    def stats(series: pd.Series) -> dict:
        series = series.sort_index()
        equity = (1 + series).cumprod()
        std = series.std(ddof=1)
        return {
            "sessions": int(len(series)),
            "mean_bps": float(series.mean() * 10_000),
            "win_rate": float((series > 0).mean()),
            "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
            "cumulative_return": float(equity.iloc[-1] - 1),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
        }
    top1 = selected[selected["rank"] == 1].set_index("timestamp")[LABEL]
    top10 = selected.groupby("timestamp")[LABEL].mean()
    return {"top1": stats(top1), "top10": stats(top10)}


def rolling_strategy(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    label_z: pd.Series,
    features: list[str],
    test_dates: list,
    weighting: str,
) -> tuple[pd.DataFrame, list[dict]]:
    dates = list(pd.Index(panel.timestamp.unique()).sort_values())
    locations = {date: position for position, date in enumerate(dates)}
    predictions, blocks = [], []
    for offset in range(0, len(test_dates), REBALANCE):
        hold_dates = test_dates[offset:offset + REBALANCE]
        position = locations[hold_dates[0]]
        history_dates = dates[position - LOOKBACK:position]
        history_mask = panel.timestamp.isin(history_dates)
        product = signals.loc[history_mask, features].multiply(label_z.loc[history_mask], axis=0)
        daily_ic = product.groupby(panel.loc[history_mask, "timestamp"], sort=False).mean()
        first, second = np.array_split(daily_ic, 2)
        mean_ic = daily_ic.mean()
        stable = (first.mean() * second.mean() > 0) & (mean_ic.abs() >= 0.003)
        chosen = mean_ic[stable].abs().nlargest(TOP_FEATURES).index.tolist()
        if not chosen:
            chosen = mean_ic.abs().nlargest(TOP_FEATURES).index.tolist()
        if weighting == "equal_sign":
            weights = np.sign(mean_ic[chosen])
        elif weighting == "ic":
            weights = mean_ic[chosen]
        else:
            raise ValueError(weighting)
        weights = weights / weights.abs().sum()
        hold_mask = panel.timestamp.isin(hold_dates)
        scores = signals.loc[hold_mask, chosen].fillna(0).to_numpy() @ weights.to_numpy()
        part = panel.loc[hold_mask, ["timestamp", "symbol", LABEL]].copy()
        part["score"] = scores
        part["rank"] = part.groupby("timestamp")["score"].rank(ascending=False, method="first")
        part = part[part["rank"] <= 10]
        predictions.append(part)
        blocks.append({
            "hold_start": str(pd.Timestamp(hold_dates[0]).date()),
            "hold_end": str(pd.Timestamp(hold_dates[-1]).date()),
            "features": chosen,
            "weights": {name: float(value) for name, value in weights.items()},
        })
    return pd.concat(predictions, ignore_index=True), blocks


def main() -> int:
    panel = pd.read_parquet(PANEL).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    numeric = panel.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    candidates = [column for column in numeric if column not in EXCLUDE and not column.startswith("label")]
    # Remove market/calendar series that are constant across the daily cross-section.
    daily_unique = panel.groupby("timestamp", sort=False)[candidates].nunique(dropna=True).median()
    features = daily_unique[daily_unique >= 5].index.tolist()
    raw_z = zscore_by_day(panel, features).astype("float32").fillna(0)
    control_z = zscore_by_day(panel, CONTROLS).astype("float32").fillna(0)
    neutral_z = neutralize_by_day(raw_z, control_z, panel.timestamp).fillna(0)
    label_rank = panel.groupby("timestamp", sort=False)[LABEL].rank(pct=True)
    label_z = (label_rank - label_rank.groupby(panel.timestamp, sort=False).transform("mean"))
    label_z /= label_rank.groupby(panel.timestamp, sort=False).transform("std").replace(0, np.nan)
    label_z = label_z.fillna(0).astype("float32")
    dates = list(pd.Index(panel.timestamp.unique()).sort_values())
    test_dates = dates[-TEST_DAYS:]
    definitions = [
        ("raw_equal_sign", raw_z, "equal_sign"),
        ("raw_ic_weight", raw_z, "ic"),
        ("neutral_ic_weight", neutral_z, "ic"),
    ]
    rows, report, all_predictions = [], {}, []
    for name, signals, weighting in definitions:
        selected, blocks = rolling_strategy(panel, signals, label_z, features, test_dates, weighting)
        selected["experiment"] = name
        all_predictions.append(selected)
        metrics = performance(selected)
        report[name] = {"metrics": metrics, "blocks": blocks}
        rows.append({
            "experiment": name,
            **{f"top1_{key}": value for key, value in metrics["top1"].items()},
            **{f"top10_{key}": value for key, value in metrics["top10"].items()},
        })
    summary = pd.DataFrame(rows)
    report["protocol"] = {
        "target": "open(t) to open(t+1), net 20 bps",
        "test_start": str(pd.Timestamp(test_dates[0]).date()),
        "test_end": str(pd.Timestamp(test_dates[-1]).date()),
        "lookback_sessions": LOOKBACK,
        "rebalance_sessions": REBALANCE,
        "candidate_features": len(features),
        "selection": "same IC sign in two trailing-history halves; abs IC >= 0.003; top 20",
        "neutralization_controls": CONTROLS,
    }
    summary.to_csv(OUT_CSV, index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(OUT_PRED, index=False)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_json(orient="records", indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
