"""Build Qlib Alpha360-compatible lagged raw OHLCV features.

Qlib Alpha360 uses 60 lags for close/open/high/low/VWAP/volume, normalized by
the latest close or volume.  Yahoo daily bars do not contain VWAP; matching
Qlib's documented Yahoo compatibility, those 60 fields are filled with zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
BASE = ROOT / "data/us_overnight_open_to_open/features/standard_daily_clean/panel.parquet"
OUT = ROOT / "data/us_overnight_open_to_open/features/alpha360"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE, columns=["timestamp", "feature_asof", "symbol", "label_next_open_net"])
    allowed = set(base["symbol"].unique())
    parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    for path in sorted(DAILY.glob("US_*.parquet")):
        bars = pd.read_parquet(path).sort_values("timestamp")
        if bars.empty or str(bars.iloc[0]["symbol"]) not in allowed:
            continue
        # Row t must use the latest completed bar t-1. Qlib lag 0 therefore
        # corresponds to shift(1), and lag 59 to shift(60).
        latest_close = bars["close"].astype(float).shift(1)
        latest_volume = bars["volume"].astype(float).shift(1)
        values: dict[str, pd.Series | float] = {
            "timestamp": bars["timestamp"], "symbol": bars["symbol"]
        }
        names: list[str] = []
        for field in ("close", "open", "high", "low"):
            series = bars[field].astype(float)
            for lag in range(59, -1, -1):
                name = f"{field.upper()}{lag}"
                values[name] = series.shift(lag + 1) / latest_close
                names.append(name)
        for lag in range(59, -1, -1):
            name = f"VWAP{lag}"
            values[name] = 0.0
            names.append(name)
        volume = bars["volume"].astype(float)
        for lag in range(59, -1, -1):
            name = f"VOLUME{lag}"
            values[name] = volume.shift(lag + 1) / (latest_volume + 1e-12)
            names.append(name)
        feature_names = names
        parts.append(pd.DataFrame(values))
    raw = pd.concat(parts, ignore_index=True)
    panel = base.merge(raw, on=["timestamp", "symbol"], how="left")
    panel[feature_names] = panel[feature_names].replace([np.inf, -np.inf], np.nan)
    panel.to_parquet(OUT / "panel.parquet", index=False)
    (OUT / "features.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    audit = {
        "rows": len(panel), "symbols": int(panel["symbol"].nunique()),
        "schema_features": len(feature_names), "effective_nonconstant_features": 300,
        "vwap_features_filled_zero": 60,
        "timing": "lag0 is completed bar t-1 for a decision before open(t)",
        "source": "Microsoft Qlib Alpha360DL.get_feature_config",
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
