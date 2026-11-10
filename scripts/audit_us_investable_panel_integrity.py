"""Full-row timing, key and label audit for the investable overnight panel."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
PANEL = ROOT / "data/us_overnight_corrected/features/historical_panel_bundle_investable_120d_20m_5.parquet"
REPORTS = ROOT / "data/us_overnight_corrected/reports"
COST = 20.0 / 10_000


def expected_labels(path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(path).sort_values("timestamp").copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    open_ = pd.to_numeric(bars["open"], errors="coerce")
    dividend = pd.to_numeric(bars.get("cash_dividend", 0.0), errors="coerce").fillna(0.0)
    return pd.DataFrame({
        "timestamp": bars["timestamp"],
        "symbol": bars["symbol"].astype(str),
        "expected_label": (open_.shift(-1) + dividend.shift(-1)) / open_ - 1 - COST,
        "label_end": bars["timestamp"].shift(-1),
        "open": open_,
        "next_open": open_.shift(-1),
        "cash_dividend_next": dividend.shift(-1),
    })


def main() -> int:
    panel = pd.read_parquet(PANEL).copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    expected = pd.concat(
        [expected_labels(path) for path in sorted(DAILY.glob("US_*.parquet"))],
        ignore_index=True,
    )
    checked = panel[["timestamp", "symbol", "feature_asof", "label_next_open_net"]].merge(
        expected, on=["timestamp", "symbol"], how="left", validate="many_to_one"
    )
    checked["label_error"] = checked["label_next_open_net"] - checked["expected_label"]
    finite_error = checked["label_error"].replace([np.inf, -np.inf], np.nan).dropna().abs()
    feature_asof = pd.to_datetime(checked["feature_asof"], utc=True, errors="coerce")
    label_end = pd.to_datetime(checked["label_end"], utc=True, errors="coerce")
    key_duplicates = int(panel.duplicated(["timestamp", "symbol"]).sum())
    unmatched = int(checked["expected_label"].isna().sum())
    timing_violations = int((feature_asof >= checked["timestamp"]).fillna(True).sum())
    label_horizon_violations = int((label_end <= checked["timestamp"]).fillna(True).sum())
    invalid_prices = int(
        ((checked["open"] <= 0) | (checked["next_open"] <= 0)
         | ~np.isfinite(checked["open"]) | ~np.isfinite(checked["next_open"])).sum()
    )
    mismatch = checked.loc[checked["label_error"].abs() > 1e-10].copy()
    mismatch.to_csv(REPORTS / "investable_panel_label_mismatches.csv", index=False)
    summary = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "panel": str(PANEL),
        "rows": int(len(panel)),
        "sessions": int(panel["timestamp"].nunique()),
        "symbols": int(panel["symbol"].nunique()),
        "duplicate_timestamp_symbol_rows": key_duplicates,
        "unmatched_source_labels": unmatched,
        "feature_asof_not_before_signal": timing_violations,
        "label_end_not_after_signal": label_horizon_violations,
        "invalid_open_or_next_open": invalid_prices,
        "label_mismatch_over_1e_10": int(len(mismatch)),
        "maximum_absolute_label_error": float(finite_error.max()) if len(finite_error) else None,
        "cash_dividend_rows": int((checked["cash_dividend_next"].fillna(0) != 0).sum()),
        "cost_bps_recomputed": 20.0,
        "decision": "PASS" if not any([
            key_duplicates, unmatched, timing_violations, label_horizon_violations,
            invalid_prices, len(mismatch),
        ]) else "FAIL",
    }
    (REPORTS / "investable_panel_integrity_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
