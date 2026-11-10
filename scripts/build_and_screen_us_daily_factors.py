"""Build lagged standard daily factors and screen them by cross-sectional Rank IC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.us_overnight.standard_daily_features import standard_daily_features

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
BASE = ROOT / "data/us_overnight_open_to_open/features/base/historical_panel_61_open2open_lag1.parquet"
OUT = ROOT / "data/us_overnight_open_to_open/features/standard_daily"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for path in sorted(DAILY.glob("US_*.parquet")):
        features = standard_daily_features(pd.read_parquet(path))
        columns = [column for column in features if column not in {"timestamp", "symbol"}]
        features.loc[:, columns] = features[columns].shift(1)
        parts.append(features)
    raw = pd.concat(parts, ignore_index=True)
    columns = [column for column in raw if column not in {"timestamp", "symbol"}]
    ranked = raw[["timestamp", "symbol"]].copy()
    for column in columns:
        ranked[column] = raw.groupby("timestamp")[column].rank(pct=True) - 0.5
    labels = pd.read_parquet(BASE)
    panel = labels.merge(ranked, on=["timestamp", "symbol"], how="left")
    dates = sorted(panel["timestamp"].dropna().unique())
    development = panel.loc[panel["timestamp"].isin(dates[:-126])].copy()
    rows = []
    for column in columns:
        daily_ic = development.groupby("timestamp").apply(
            lambda day, factor=column: day[factor].corr(
                day["label_next_open_net"], method="spearman"
            ),
            include_groups=False,
        ).dropna()
        signed = 1.0 if daily_ic.mean() >= 0 else -1.0
        values = development[[column, "label_next_open_net"]].dropna().copy()
        values["bucket"] = values.groupby(development.loc[values.index, "timestamp"])[column].transform(
            lambda series: pd.qcut(series.rank(method="first"), 10, labels=False, duplicates="drop")
        )
        bucket_mean = values.groupby("bucket")["label_next_open_net"].mean() * signed
        violations = int((bucket_mean.diff().dropna() <= 0).sum())
        rows.append({
            "feature": column,
            "direction": int(signed),
            "mean_rank_ic": float(daily_ic.mean()),
            "rank_ic_ir": float(daily_ic.mean() / daily_ic.std(ddof=1) * np.sqrt(252)),
            "positive_ic_rate": float(((daily_ic * signed) > 0).mean()),
            "decile_violations": violations,
            "top_minus_bottom_bps": float((bucket_mean.iloc[-1] - bucket_mean.iloc[0]) * 10_000),
        })
    screen = pd.DataFrame(rows).sort_values("rank_ic_ir", key=lambda s: s.abs(), ascending=False)
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
    base_columns = [
        column for column in labels
        if column not in {"timestamp", "feature_asof", "symbol", "label_next_open_net"}
    ]
    (OUT / "base_features.json").write_text(
        json.dumps(base_columns, indent=2), encoding="utf-8"
    )
    (OUT / "combined_features.json").write_text(
        json.dumps([*base_columns, *selected["feature"].tolist()], indent=2), encoding="utf-8"
    )
    print(json.dumps({"factors": len(columns), "selected": len(selected), "top": screen.head(15).to_dict("records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
