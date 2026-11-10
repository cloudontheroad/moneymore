"""Evaluate a cash gate based on agreement between the accepted model and ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data/us_overnight_paper_only/models/investable_open2open_corrected"
REPORT_DIR = ROOT / "data/us_overnight_corrected/reports"


def stats(series: pd.Series) -> dict:
    equity = (1 + series).cumprod()
    return {
        "mean_bps": float(series.mean() * 10_000),
        "sharpe": float(series.mean() / series.std(ddof=1) * np.sqrt(252)),
        "cumulative_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def main() -> int:
    a = pd.read_parquet(MODEL_DIR / "corrected_cs_varying_clip10_selected_predictions.parquet")
    b = pd.read_parquet(MODEL_DIR / "corrected_cs_varying_clip10_no_falling_selected_predictions.parquet")
    merged = a[["timestamp", "symbol", "label_next_open_net", "rank"]].rename(
        columns={"rank": "main_rank", "label_next_open_net": "main_label"}
    ).merge(
        b[["timestamp", "symbol", "label_next_open_net", "rank"]].rename(
            columns={"rank": "ablation_rank", "label_next_open_net": "ablation_label"}
        ), on=["timestamp", "symbol"], how="outer"
    )
    merged["label"] = merged.main_label.fillna(merged.ablation_label)
    merged["score"] = 0.9 * merged.main_rank.fillna(11) + 0.1 * merged.ablation_rank.fillna(11)
    merged["rank"] = merged.groupby("timestamp")["score"].rank(method="first")
    selected = merged[merged["rank"] <= 10]
    dates = pd.Index(sorted(merged.timestamp.unique()))
    top1 = selected[selected["rank"] == 1].set_index("timestamp")
    top10 = selected.groupby("timestamp")["label"].mean().reindex(dates)
    rows = []
    for threshold in (1, 2, 3, 5, 10):
        active = (top1.main_rank <= threshold) & (top1.ablation_rank <= threshold)
        position = active.astype(float).reindex(dates, fill_value=0)
        top1_returns = top1.label.reindex(dates).fillna(0) * position
        top10_returns = top10.fillna(0) * position
        rows.append({
            "agreement_top_k": threshold,
            "active_sessions": int(active.sum()),
            "coverage": float(active.mean()),
            **{f"top1_{key}": value for key, value in stats(top1_returns).items()},
            **{f"top10_{key}": value for key, value in stats(top10_returns).items()},
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT_DIR / "corrected_agreement_gate.csv", index=False)
    k5 = frame[frame.agreement_top_k == 5].iloc[0].to_dict()
    report = {
        "tested": len(frame),
        "reference_gate": k5,
        "decision": "record as optional risk overlay; do not replace alpha baseline",
        "reason": "K=5 improves drawdown and Top10 Sharpe but reduces Top1 and Top10 mean return.",
        "results": json.loads(frame.to_json(orient="records")),
    }
    (REPORT_DIR / "corrected_agreement_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
