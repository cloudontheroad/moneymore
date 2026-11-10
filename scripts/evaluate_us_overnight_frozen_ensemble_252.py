"""Evaluate the frozen four-model rank ensemble on a 252-session window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/us_overnight_paper_only/models/ensemble_robustness_252"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"
DEFAULT_MODELS = {
    "champion": MODEL_DIR / "robust252_champion_selected_predictions.parquet",
    "mcw1": MODEL_DIR / "robust252_mcw1_selected_predictions.parquet",
    "half63": MODEL_DIR / "robust252_half63_selected_predictions.parquet",
    "top1w5": MODEL_DIR / "robust252_top1w5_selected_predictions.parquet",
}
WEIGHTS = {"champion": 0.3, "mcw1": 0.2, "half63": 0.3, "top1w5": 0.2}


def load() -> pd.DataFrame:
    parts = []
    for name, path in MODELS.items():
        frame = pd.read_parquet(path)[
            ["timestamp", "symbol", "label_next_open_net", "rank"]
        ].copy()
        parts.append(frame.rename(columns={
            "rank": name,
            "label_next_open_net": f"label_{name}",
        }))
    merged = parts[0]
    for frame in parts[1:]:
        merged = merged.merge(frame, on=["timestamp", "symbol"], how="outer")
    labels = [column for column in merged if column.startswith("label_")]
    merged["label_next_open_net"] = merged[labels].bfill(axis=1).iloc[:, 0]
    return merged.drop(columns=labels)


def select(frame: pd.DataFrame) -> pd.DataFrame:
    score = sum(WEIGHTS[name] * frame[name].fillna(11.0) for name in WEIGHTS)
    selected = frame[["timestamp", "symbol", "label_next_open_net"]].copy()
    selected["ensemble_rank_score"] = score
    selected["rank"] = selected.groupby("timestamp")["ensemble_rank_score"].rank(method="first")
    return selected.loc[selected["rank"] <= 10].copy()


def stats(series: pd.Series) -> dict[str, float | int]:
    equity = (1.0 + series).cumprod()
    std = series.std(ddof=1)
    return {
        "sessions": int(series.size),
        "mean_bps": float(series.mean() * 10_000),
        "cumulative_return": float(equity.iloc[-1] - 1),
        "win_rate": float((series > 0).mean()),
        "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else float("nan"),
        "volatility_annualized": float(std * np.sqrt(252)),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def evaluate_period(frame: pd.DataFrame, dates: list, name: str) -> dict:
    period = frame.loc[frame["timestamp"].isin(dates)]
    top1 = period.loc[period["rank"] == 1].set_index("timestamp")["label_next_open_net"]
    top10 = period.groupby("timestamp")["label_next_open_net"].mean()
    return {"period": name, "top1": stats(top1), "top10": stats(top10)}


def evaluate_selected(frame: pd.DataFrame) -> dict:
    top1 = frame.loc[frame["rank"] == 1].set_index("timestamp")["label_next_open_net"]
    top10 = frame.groupby("timestamp")["label_next_open_net"].mean()
    return {"top1": stats(top1), "top10": stats(top10)}


def reference_diagnostics(selected: pd.DataFrame, universe: pd.Series) -> dict:
    top1 = selected.loc[selected["rank"] == 1].set_index("timestamp")["label_next_open_net"]
    top10 = selected.groupby("timestamp")["label_next_open_net"].mean()
    universe = universe.reindex(top10.index)
    regimes = {}
    for name, mask in {
        "universe_up": universe > 0,
        "universe_down_or_flat": universe <= 0,
    }.items():
        regimes[name] = {
            "sessions": int(mask.sum()),
            "top1_raw_mean_bps": float(top1.loc[mask].mean() * 10_000),
            "top10_raw_mean_bps": float(top10.loc[mask].mean() * 10_000),
            "top1_excess_mean_bps": float((top1 - universe).loc[mask].mean() * 10_000),
            "top10_excess_mean_bps": float((top10 - universe).loc[mask].mean() * 10_000),
        }
    return {
        "adoption_gate": False,
        "benchmark": "same-day eligible-universe equal weight",
        "top1_excess": stats(top1 - universe),
        "top10_excess": stats(top10 - universe),
        "realized_market_regimes_for_diagnosis_only": regimes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchored", action="store_true")
    args = parser.parse_args()
    global MODELS
    if args.anchored:
        anchored_dir = ROOT / "data/us_overnight_paper_only/models/anchor_verification"
        MODELS = {
            "champion": anchored_dir / "anchored_champion_252_selected_predictions.parquet",
            "mcw1": anchored_dir / "anchored_mcw1_252_selected_predictions.parquet",
            "half63": anchored_dir / "anchored_half63_252_selected_predictions.parquet",
            "top1w5": anchored_dir / "anchored_top1w5_252_selected_predictions.parquet",
        }
        report_stem = "anchored_frozen_ensemble_252"
        champion_daily_path = anchored_dir / "anchored_champion_252_walk_forward_daily.csv"
    else:
        MODELS = DEFAULT_MODELS
        report_stem = "frozen_ensemble_252"
        champion_daily_path = MODEL_DIR / "robust252_champion_walk_forward_daily.csv"
    missing = [str(path) for path in MODELS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model predictions: {missing}")
    selected = select(load())
    dates = sorted(selected["timestamp"].unique())
    if len(dates) != 252:
        raise ValueError(f"Expected 252 sessions, found {len(dates)}")
    periods = [evaluate_period(selected, dates, "full_252")]
    periods.extend([
        evaluate_period(selected, dates[:126], "first_126"),
        evaluate_period(selected, dates[126:], "last_126"),
    ])
    for index in range(4):
        start = index * 63
        periods.append(evaluate_period(selected, dates[start:start + 63], f"fold_{index + 1}_63"))
    constituents = {}
    for name, path in MODELS.items():
        constituents[name] = evaluate_selected(pd.read_parquet(path))
    universe_frame = pd.read_csv(champion_daily_path, index_col=0)
    universe_frame.index = pd.to_datetime(universe_frame.index)
    selected["timestamp"] = pd.to_datetime(selected["timestamp"])
    diagnostics = reference_diagnostics(selected, universe_frame["universe_mean_net"])
    full = periods[0]
    champion = constituents["champion"]
    jointly_improves_champion = all([
        full["top1"]["mean_bps"] > champion["top1"]["mean_bps"],
        full["top1"]["sharpe"] > champion["top1"]["sharpe"],
        full["top10"]["mean_bps"] > champion["top10"]["mean_bps"],
        full["top10"]["sharpe"] > champion["top10"]["sharpe"],
    ])
    report = {
        "method": "frozen_weight_rank_ensemble",
        "weights": WEIGHTS,
        "weights_reoptimized": False,
        "validation_sessions": len(dates),
        "note": "Back-extension robustness test; the last 126 sessions overlap the selection window.",
        "periods": periods,
        "constituent_full_252": constituents,
        "jointly_improves_champion": jointly_improves_champion,
        "decision": "accept" if jointly_improves_champion else "reject",
        "reference_diagnostics": diagnostics,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(REPORT_DIR / f"{report_stem}_predictions.parquet", index=False)
    (REPORT_DIR / f"{report_stem}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
