"""Build an isolated, audited open-to-next-open panel from adjusted daily bars.

Every predictor used for trade session t is shifted from session t-1.  The
label is open(t) -> open(t+1), including cash dividends paid at t+1.  Thus a
row timestamped t is fully knowable before open(t), except for its forward
label, which is used only for training/evaluation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from train_us_overnight_ranker import _cross_sectionalize, _winsorize_by_date

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.features import FEATURE_COLUMNS, daily_stock_features

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    source = ROOT / config.output_root / "daily_yahoo_adjusted"
    destination = ROOT / "data/us_overnight_open_to_open/features/base"
    destination.mkdir(parents=True, exist_ok=True)

    parts: list[pd.DataFrame] = []
    for path in sorted(source.glob("US_*.parquet")):
        bars = pd.read_parquet(path).sort_values("timestamp")
        if bars.empty:
            continue
        features = daily_stock_features(bars, cost_bps=config.round_trip_cost_bps)
        feature_asof = features["timestamp"].shift(1)
        features.loc[:, FEATURE_COLUMNS] = features[FEATURE_COLUMNS].shift(1)
        features["feature_asof"] = feature_asof
        features["symbol"] = str(bars.iloc[0]["symbol"])
        features["label_next_open_net"] = features["label_open_to_next_open_net"]
        parts.append(
            features[
                [
                    "timestamp",
                    "feature_asof",
                    "symbol",
                    *FEATURE_COLUMNS,
                    "label_next_open_net",
                ]
            ]
        )
    if not parts:
        raise FileNotFoundError(f"no adjusted histories in {source}")

    raw = pd.concat(parts, ignore_index=True)
    panel = _cross_sectionalize(_winsorize_by_date(raw))
    finite_labels = raw["label_next_open_net"].replace([np.inf, -np.inf], np.nan).dropna()
    outliers = raw.loc[raw["label_next_open_net"].abs() > 0.5, [
        "timestamp", "symbol", "label_next_open_net"
    ]].sort_values("label_next_open_net")
    panel_path = destination / f"historical_panel_{len(FEATURE_COLUMNS)}_open2open_lag1.parquet"
    panel.to_parquet(panel_path, index=False)
    outliers.to_csv(destination / "label_outliers_abs50pct.csv", index=False)

    audit = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "panel": str(panel_path),
        "symbols": int(panel["symbol"].nunique()),
        "rows": len(panel),
        "sessions": int(panel["timestamp"].nunique()),
        "start": str(panel["timestamp"].min()),
        "end": str(panel["timestamp"].max()),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_timing": "all predictors shifted one completed symbol session",
        "decision": "before open(t), using completed daily bars through t-1",
        "label": "open(t) to open(t+1), cash dividend included, net of configured cost",
        "label_min": float(finite_labels.min()),
        "label_max": float(finite_labels.max()),
        "label_abs_gt_50pct": int((finite_labels.abs() > 0.5).sum()),
        "known_limitation": "current Top1000 membership, not historical point-in-time universe",
    }
    (destination / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
