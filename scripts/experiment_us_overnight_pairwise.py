"""Continuous-return pairwise ranking experiment; never replaces production automatically."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.features import FEATURE_COLUMNS

ACTIVE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SESSIONS = 504
VALIDATION_SESSIONS = 520
RETRAIN_EVERY = 63
MIN_PAIR_GAP = 0.002
PAIRS_PER_DAY = 768


class ScoreNet(nn.Module):
    def __init__(self, features: int, linear: bool = False) -> None:
        super().__init__()
        self.layers = nn.Linear(features, 1) if linear else nn.Sequential(
            nn.Linear(features, 64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def _pairs(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    gaps: list[float] = []
    for _, day in frame.groupby("timestamp", sort=False):
        raw_y = day["label_next_open_net"].to_numpy(float)
        median = np.nanmedian(raw_y)
        mad = np.nanmedian(np.abs(raw_y - median))
        y = np.clip((raw_y - median) / max(mad * 1.4826, 0.002), -5.0, 5.0)
        x = np.nan_to_num(
            day[ACTIVE_FEATURE_COLUMNS].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        order = np.argsort(y)
        strata = [values for values in np.array_split(order, 10) if len(values)]
        for _ in range(PAIRS_PER_DAY):
            low_bucket = int(rng.integers(0, len(strata) - 1))
            high_bucket = int(rng.integers(low_bucket + 1, len(strata)))
            low = int(rng.choice(strata[low_bucket]))
            high = int(rng.choice(strata[high_bucket]))
            gap = y[high] - y[low]
            if gap <= 0:
                continue
            left.append(x[high])
            right.append(x[low])
            gaps.append(gap)
    return np.asarray(left), np.asarray(right), np.asarray(gaps, dtype=np.float32)


def _fit(frame: pd.DataFrame, device: torch.device, linear: bool) -> ScoreNet:
    rng = np.random.default_rng(42)
    left, right, gaps = _pairs(frame, rng)
    model = ScoreNet(len(ACTIVE_FEATURE_COLUMNS), linear=linear).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    batch = 4096
    generator = torch.Generator().manual_seed(42)
    for _ in range(20):
        permutation = torch.randperm(len(gaps), generator=generator)
        for start in range(0, len(gaps), batch):
            index = permutation[start : start + batch].numpy()
            high_x = torch.from_numpy(left[index]).to(device)
            low_x = torch.from_numpy(right[index]).to(device)
            weight = torch.from_numpy(np.sqrt(np.clip(gaps[index], 0.25, 5.0))).to(device)
            margin = model(high_x) - model(low_x)
            loss = (torch.nn.functional.softplus(-margin) * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def _predict(model: ScoreNet, frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    array = np.nan_to_num(
        frame[ACTIVE_FEATURE_COLUMNS].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    values = torch.from_numpy(array).to(device)
    with torch.no_grad():
        return model(values).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-path", type=Path)
    parser.add_argument("--feature-columns-json", type=Path)
    parser.add_argument("--validation-sessions", type=int, default=VALIDATION_SESSIONS)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--linear", action="store_true")
    args = parser.parse_args()
    global ACTIVE_FEATURE_COLUMNS
    if args.feature_columns_json:
        ACTIVE_FEATURE_COLUMNS = json.loads(args.feature_columns_json.read_text(encoding="utf-8"))
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    output = ROOT / config.output_root
    panel_path = args.panel_path or (
        output / "features" / f"historical_panel_{len(FEATURE_COLUMNS)}.parquet"
    )
    panel = pd.read_parquet(panel_path)
    mature = panel.dropna(subset=["label_next_open_net"]).copy()
    dates = sorted(mature["timestamp"].unique())
    validation_dates = dates[-args.validation_sessions:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: list[pd.DataFrame] = []
    model: ScoreNet | None = None
    for offset, date in enumerate(validation_dates):
        if model is None or offset % RETRAIN_EVERY == 0:
            position = dates.index(date)
            training = set(dates[max(0, position - TRAIN_SESSIONS) : position])
            model = _fit(mature.loc[mature["timestamp"].isin(training)], device, args.linear)
        day = mature.loc[mature["timestamp"] == date, [
            "timestamp", "symbol", "label_next_open_net", *ACTIVE_FEATURE_COLUMNS
        ]].copy()
        day["score"] = _predict(model, day, device)
        day["rank"] = day["score"].rank(method="first", ascending=False)
        predictions.append(day.loc[day["rank"] <= config.top_n, [
            "timestamp", "symbol", "label_next_open_net", "score", "rank"
        ]])
    selected = pd.concat(predictions, ignore_index=True)
    rows: list[dict[str, object]] = []
    for rank, group in selected.groupby("rank"):
        returns = group.sort_values("timestamp")["label_next_open_net"]
        equity = (1 + returns).cumprod()
        rows.append({
            "rank": int(rank), "sessions": len(returns),
            "mean_net_bps": float(returns.mean() * 10_000),
            "cumulative_return": float(equity.iloc[-1] - 1),
            "win_rate": float((returns > 0).mean()),
            "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
        })
    slots = pd.DataFrame(rows).sort_values("rank")
    means = slots["mean_net_bps"].to_numpy()
    payload = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "continuous_return_pairwise_ranknet",
        "architecture": "linear" if args.linear else "mlp",
        "device": str(device), "minimum_pair_gap_bps": MIN_PAIR_GAP * 10_000,
        "validation_sessions": args.validation_sessions,
        "rank_slots": slots.to_dict("records"),
        "monotonic_violations": int(((means[:-1] - means[1:]) <= 0).sum()),
        "strictly_monotonic": bool(((means[:-1] - means[1:]) > 0).all()),
        "real_orders_enabled": False,
    }
    model_dir = args.model_dir or (output / "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    slots.to_csv(model_dir / "experiment_pairwise_slots.csv", index=False)
    (model_dir / "experiment_pairwise.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
