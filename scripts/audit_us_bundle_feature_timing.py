"""Audit bundle feature provenance and pre-open availability."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_us_overnight_event_factor_panel import _decision_table, rating_factors
from build_us_overnight_factor_panel import context, stock_factors

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/us_overnight_corrected/features"
DAILY = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
OPTIONS = ROOT / "data/us_overnight_event_warehouse/options_daily"
REPORTS = ROOT / "data/us_overnight_corrected/reports"
PANEL = FEATURES / "historical_panel_bundle_investable_120d_20m_5.parquet"
CORRECT_BASE = (
    ROOT / "data/us_overnight_open_to_open/features/base/historical_panel_61_open2open_lag1.parquet"
)


def max_numeric_difference(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> float:
    differences = []
    for column in columns:
        a = pd.to_numeric(left[column], errors="coerce")
        b = pd.to_numeric(right[column], errors="coerce")
        both = a.notna() & b.notna()
        if both.any():
            differences.append(float((a[both] - b[both]).abs().max()))
        if not a.isna().equals(b.isna()):
            return float("inf")
    return max(differences, default=0.0)


def main() -> int:
    panel = pd.read_parquet(PANEL).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel["feature_asof"] = pd.to_datetime(panel["feature_asof"], utc=True)
    groups = json.loads((FEATURES / "factor_groups.json").read_text(encoding="utf-8"))
    keys = panel[["timestamp", "symbol"]]

    base_columns = groups["base"]
    base = pd.read_parquet(CORRECT_BASE, columns=["timestamp", "symbol", *base_columns])
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    base_check = panel[["timestamp", "symbol", *base_columns]].merge(
        base, on=["timestamp", "symbol"], how="left", suffixes=("", "_expected"),
        validate="one_to_one",
    )
    expected_base = base_check.rename(
        columns={f"{column}_expected": column for column in base_columns}
    )[[*base_columns]]
    observed_base = base_check[[*base_columns]]
    # The rename above collides with observed names, so compare directly by suffix.
    base_max = 0.0
    base_nan_mismatch = 0
    for column in base_columns:
        observed = pd.to_numeric(base_check[column], errors="coerce")
        expected = pd.to_numeric(base_check[f"{column}_expected"], errors="coerce")
        both = observed.notna() & expected.notna()
        if both.any():
            base_max = max(base_max, float((observed[both] - expected[both]).abs().max()))
        base_nan_mismatch += int((observed.isna() != expected.isna()).sum())

    # Recompute the stock-specific right-side factors from source bars.
    right_columns = groups["right_side_trend"]
    stock = pd.concat(
        [stock_factors(path)[["timestamp", "symbol", *right_columns]]
         for path in sorted(DAILY.glob("US_*.parquet"))],
        ignore_index=True,
    )
    for column in right_columns:
        stock[column] = stock.groupby("timestamp")[column].rank(pct=True) - 0.5
    right_check = panel[["timestamp", "symbol", *right_columns]].merge(
        stock, on=["timestamp", "symbol"], how="left", suffixes=("", "_expected"),
        validate="one_to_one",
    )
    right_max = 0.0
    right_nan_mismatch = 0
    for column in right_columns:
        observed = pd.to_numeric(right_check[column], errors="coerce")
        expected = pd.to_numeric(right_check[f"{column}_expected"], errors="coerce")
        both = observed.notna() & expected.notna()
        if both.any():
            right_max = max(right_max, float((observed[both] - expected[both]).abs().max()))
        right_nan_mismatch += int((observed.isna() != expected.isna()).sum())

    market_columns = groups["market_regime"]
    expected_market = context()
    expected_market["timestamp"] = pd.to_datetime(expected_market["timestamp"], utc=True)
    market_check = panel[["timestamp", *market_columns]].drop_duplicates("timestamp").merge(
        expected_market[["timestamp", *market_columns]], on="timestamp", how="left",
        suffixes=("", "_expected"), validate="one_to_one",
    )
    market_max = 0.0
    market_nan_mismatch = 0
    for column in market_columns:
        observed = pd.to_numeric(market_check[column], errors="coerce")
        expected = pd.to_numeric(market_check[f"{column}_expected"], errors="coerce")
        both = observed.notna() & expected.notna()
        if both.any():
            market_max = max(market_max, float((observed[both] - expected[both]).abs().max()))
        market_nan_mismatch += int((observed.isna() != expected.isna()).sum())

    option_keys = []
    for path in sorted(OPTIONS.glob("US_*.parquet")):
        raw = pd.read_parquet(path, columns=["code", "time"])
        option_keys.append(pd.DataFrame({
            "symbol": raw["code"].astype(str),
            "asof_date": pd.to_datetime(raw["time"], errors="coerce").dt.date,
        }))
    option_keys = pd.concat(option_keys, ignore_index=True).drop_duplicates()
    option_rows = panel.loc[panel["option_data_available"].fillna(0) > 0, [
        "timestamp", "symbol", "feature_asof"
    ]].copy()
    option_rows["asof_date"] = option_rows["feature_asof"].dt.tz_convert(
        "America/New_York"
    ).dt.date
    option_check = option_rows.merge(
        option_keys.assign(raw_option_key=True), on=["symbol", "asof_date"], how="left",
        validate="many_to_one",
    )

    full_session_panel = pd.read_parquet(
        FEATURES / "historical_panel_factor_groups_lag1.parquet",
        columns=["timestamp"],
    ).drop_duplicates()
    sessions = _decision_table(full_session_panel)
    rows = keys.merge(sessions, on="timestamp", how="left")
    expected_ratings = rating_factors(rows, sessions)
    rating_columns = [
        "event_rating_direction", "event_rating_target_change",
        "event_rating_age_days", "event_rating_decay_20d",
    ]
    for column in rating_columns:
        expected_ratings[column] = pd.to_numeric(expected_ratings[column], errors="coerce")
        expected_ratings[column] = expected_ratings[column].fillna(
            999.0 if column.endswith("age_days") else 0.0
        )
    rating_check = panel[["timestamp", "symbol", *rating_columns]].merge(
        expected_ratings, on=["timestamp", "symbol"], how="left",
        suffixes=("", "_expected"), validate="one_to_one",
    )
    rating_max = 0.0
    rating_nan_mismatch = 0
    for column in rating_columns:
        observed = pd.to_numeric(rating_check[column], errors="coerce")
        expected = pd.to_numeric(rating_check[f"{column}_expected"], errors="coerce")
        both = observed.notna() & expected.notna()
        if both.any():
            rating_max = max(rating_max, float((observed[both] - expected[both]).abs().max()))
        rating_nan_mismatch += int((observed.isna() != expected.isna()).sum())

    checks = {
        "feature_asof_not_before_signal": int((panel["feature_asof"] >= panel["timestamp"]).sum()),
        "base_max_abs_difference": base_max,
        "base_nan_mismatch": base_nan_mismatch,
        "right_side_max_abs_difference": right_max,
        "right_side_nan_mismatch": right_nan_mismatch,
        "market_max_abs_difference": market_max,
        "market_nan_mismatch": market_nan_mismatch,
        "option_available_rows": int(len(option_rows)),
        "option_available_without_exact_feature_asof_source": int(
            option_check["raw_option_key"].isna().sum()
        ),
        "rating_max_abs_difference": rating_max,
        "rating_nan_mismatch": rating_nan_mismatch,
    }
    failed = any([
        checks["feature_asof_not_before_signal"], base_max > 1e-12, base_nan_mismatch,
        right_max > 1e-12, right_nan_mismatch, market_max > 1e-12,
        market_nan_mismatch, checks["option_available_without_exact_feature_asof_source"],
        rating_max > 1e-12, rating_nan_mismatch,
    ])
    report = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "panel": str(PANEL),
        "rows": int(len(panel)),
        "feature_families": {
            "base": len(base_columns), "right_side": len(right_columns),
            "market": len(market_columns), "options": 12, "ratings": len(rating_columns),
            "calendar": len(groups["calendar"]),
        },
        "checks": checks,
        "calendar_policy": "deterministic from signal session date; known before open",
        "option_policy": "exact option session equal to feature_asof local date",
        "rating_policy": "date-only rating becomes available on the next US session",
        "decision": "FAIL" if failed else "PASS",
    }
    (REPORTS / "investable_bundle_feature_timing_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
