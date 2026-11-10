"""Join lagged daily option-information features to the corrected stock panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.us_overnight.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/us_overnight_corrected/features/historical_panel_61_lag1.parquet"
RAW = ROOT / "data/us_overnight_event_warehouse/options_daily"
OUT = ROOT / "data/us_overnight_corrected/features"
OPTION_COLUMNS = [
    "option_volume_log", "option_volume_z20", "option_call_put_imbalance",
    "option_put_call_volume_ratio", "option_open_interest_log",
    "option_open_interest_change_1d", "option_put_call_open_interest_ratio",
    "option_iv", "option_hv", "option_iv_minus_hv", "option_iv_change_1d",
]


def one(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("time")
    frame["symbol"] = frame["code"].astype(str)
    frame["asof_date"] = pd.to_datetime(frame["time"], errors="coerce").dt.date
    def numeric(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")
    volume, call, put = numeric("option_volume"), numeric("call_volume"), numeric("put_volume")
    interest, iv, hv = numeric("option_open_interest"), numeric("iv"), numeric("hv")
    result = frame[["symbol", "asof_date"]].copy()
    result["option_volume_log"] = np.log1p(volume)
    result["option_volume_z20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    result["option_call_put_imbalance"] = (call - put) / (call + put).replace(0, np.nan)
    result["option_put_call_volume_ratio"] = numeric("put_call_volume_ratio")
    result["option_open_interest_log"] = np.log1p(interest)
    result["option_open_interest_change_1d"] = interest.pct_change(fill_method=None).clip(-2, 2)
    result["option_put_call_open_interest_ratio"] = numeric("put_call_open_interest_ratio")
    result["option_iv"] = iv
    result["option_hv"] = hv
    result["option_iv_minus_hv"] = iv - hv
    result["option_iv_change_1d"] = iv.diff()
    return result


def main() -> int:
    paths = sorted(RAW.glob("US_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no option histories in {RAW}")
    option = pd.concat([one(path) for path in paths], ignore_index=True)
    panel = pd.read_parquet(BASE)
    panel["asof_date"] = pd.to_datetime(panel["feature_asof"], utc=True).dt.tz_convert("America/New_York").dt.date
    enriched = panel.merge(option, on=["symbol", "asof_date"], how="left", validate="many_to_one")
    enriched["option_data_available"] = enriched["option_iv"].notna().astype(float)
    for column in OPTION_COLUMNS:
        enriched[column] = enriched.groupby("timestamp")[column].rank(pct=True) - 0.5
    feature_columns = [*FEATURE_COLUMNS, *OPTION_COLUMNS, "option_data_available"]
    target = OUT / "historical_panel_61_plus_options.parquet"
    enriched.drop(columns="asof_date").to_parquet(target, index=False)
    (OUT / "feature_columns_61_plus_options.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    audit = {
        "panel": str(target), "raw_symbols": len(paths), "panel_symbols_covered": int(enriched.loc[enriched.option_data_available > 0, "symbol"].nunique()),
        "coverage_rows": float(enriched.option_data_available.mean()), "features": OPTION_COLUMNS,
        "causal_policy": "option close snapshot at feature_asof, used no earlier than next regular open",
    }
    (OUT / "option_factor_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
