"""Quarantine material source mismatches/extremes and redo daily-factor screening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/us_overnight_open_to_open/features/standard_daily/panel.parquet"
SCREEN = ROOT / "data/us_overnight_open_to_open/features/standard_daily/factor_screen.csv"
AUDIT = ROOT / "state/us-overnight-full-source-validation/summary.csv"
OUT = ROOT / "data/us_overnight_open_to_open/features/standard_daily_clean"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT)
    quarantine = audit.loc[
        (audit["stable_label_rows_over_10bps"] >= 10)
        | (audit["stable_label_p99_abs_diff_bps"] > 25)
        | (audit["extreme_yahoo_label_rows"] > 0), "symbol"
    ].tolist()
    factors = pd.read_csv(SCREEN)["feature"].tolist()
    panel = pd.read_parquet(SOURCE)
    panel = panel.loc[~panel["symbol"].isin(quarantine)].copy()
    dates = sorted(panel["timestamp"].unique())
    development = panel.loc[panel["timestamp"].isin(dates[:-126])]
    rows = []
    for factor in factors:
        daily_ic = development.groupby("timestamp").apply(
            lambda day, column=factor: day[column].corr(
                day["label_next_open_net"], method="spearman"
            ), include_groups=False,
        ).dropna()
        direction = 1.0 if daily_ic.mean() >= 0 else -1.0
        values = development[["timestamp", factor, "label_next_open_net"]].dropna().copy()
        values["bucket"] = values.groupby("timestamp")[factor].transform(
            lambda series: pd.qcut(series.rank(method="first"), 10, labels=False)
        )
        bucket = values.groupby("bucket")["label_next_open_net"].mean() * direction
        rows.append({
            "feature": factor, "direction": int(direction),
            "mean_rank_ic": float(daily_ic.mean()),
            "rank_ic_ir": float(daily_ic.mean() / daily_ic.std(ddof=1) * np.sqrt(252)),
            "positive_ic_rate": float(((daily_ic * direction) > 0).mean()),
            "decile_violations": int((bucket.diff().dropna() <= 0).sum()),
            "top_minus_bottom_bps": float((bucket.iloc[-1] - bucket.iloc[0]) * 10_000),
        })
    screen = pd.DataFrame(rows).sort_values("rank_ic_ir", key=lambda x: x.abs(), ascending=False)
    selected = screen.loc[
        (screen["rank_ic_ir"].abs() >= 0.5)
        & (screen["top_minus_bottom_bps"] > 0)
        & (screen["decile_violations"] <= 4)
    ]
    panel.to_parquet(OUT / "panel.parquet", index=False)
    screen.to_csv(OUT / "factor_screen.csv", index=False)
    (OUT / "selected_features.json").write_text(
        json.dumps(selected["feature"].tolist(), indent=2), encoding="utf-8"
    )
    (OUT / "quarantine.json").write_text(json.dumps(quarantine, indent=2), encoding="utf-8")
    print(json.dumps({
        "symbols": int(panel["symbol"].nunique()), "quarantined": len(quarantine),
        "factors_screened": len(factors), "factors_selected": len(selected),
        "top_factors": selected.head(15)["feature"].tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
