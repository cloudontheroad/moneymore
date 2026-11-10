"""Audit absolute versus cross-sectional edge across the two validation halves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_corrected/features/historical_panel_bundle_investable_120d_20m_5.parquet"
PRED = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected/corrected_cs_varying_clip10_selected_predictions.parquet"
OUT = ROOT / "data/us_overnight_corrected/reports/corrected_regime_drift_audit.json"


def metrics(series: pd.Series) -> dict:
    std = series.std(ddof=1)
    equity = (1 + series).cumprod()
    return {
        "mean_bps": float(series.mean() * 10_000),
        "win_rate": float((series > 0).mean()),
        "sharpe": float(series.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def main() -> int:
    panel = pd.read_parquet(PANEL)
    selected = pd.read_parquet(PRED)
    dates = list(pd.Index(selected.timestamp.unique()).sort_values())
    halves = []
    for name, half_dates in (("first_126", dates[:126]), ("last_126", dates[126:])):
        top1 = selected[
            selected.timestamp.isin(half_dates) & (selected["rank"] == 1)
        ].set_index("timestamp").label_next_open_net
        top10 = selected[selected.timestamp.isin(half_dates)].groupby("timestamp").label_next_open_net.mean()
        universe = panel[panel.timestamp.isin(half_dates)].groupby("timestamp").label_next_open_net.mean()
        halves.append({
            "period": name,
            "top1": metrics(top1),
            "top10": metrics(top10),
            "universe": metrics(universe),
            "top1_excess_bps": float((top1 - universe).mean() * 10_000),
            "top10_excess_bps": float((top10 - universe).mean() * 10_000),
        })
    top1_all = selected[selected["rank"] == 1].set_index("timestamp").label_next_open_net.reindex(dates)
    top10_all = selected.groupby("timestamp").label_next_open_net.mean().reindex(dates)
    universe_all = panel.groupby("timestamp").label_next_open_net.mean().reindex(dates)
    trailing = universe_all.rolling(21, min_periods=21).mean().shift(1)
    gates = []
    for threshold_bps in (-15, -10, -5, 0, 5):
        active = trailing > threshold_bps / 10_000
        top1 = top1_all.where(active, 0)
        top10 = top10_all.where(active, 0)
        gates.append({
            "threshold_bps": threshold_bps,
            "coverage": float(active.mean()),
            "top1": metrics(top1),
            "top10": metrics(top10),
        })
    report = {
        "halves": halves,
        "market_state_gate_reference": gates,
        "decision": "no market-state gate adopted",
        "reason": (
            "The last half retains positive cross-sectional excess but fails to cover costs. "
            "Tested past-only gates improve drawdown at the expense of primary mean return."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
