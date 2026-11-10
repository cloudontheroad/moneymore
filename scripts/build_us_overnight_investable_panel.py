"""Build a point-in-time investable US overnight panel and audit extreme gaps."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
FEATURES = ROOT / "data/us_overnight_corrected/features"
SOURCE_PANEL = FEATURES / "historical_panel_bundle_modules.parquet"
OUTPUT_PANEL = FEATURES / "historical_panel_bundle_investable_120d_20m_5.parquet"
REPORTS = ROOT / "data/us_overnight_corrected/reports"

MIN_LISTING_DAYS = 120
MIN_MEDIAN_DOLLAR_VOLUME_20D = 20_000_000.0
MIN_PRICE = 5.0


def eligibility_from_bars(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pd.read_parquet(path).sort_values("timestamp").copy()
    if bars.empty:
        return pd.DataFrame(), pd.DataFrame()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars["symbol"] = bars["symbol"].astype(str)
    prior_close = bars["close"].astype(float).shift(1)
    prior_volume = bars["volume"].astype(float).shift(1)
    dollar_volume = bars["close"].astype(float) * bars["volume"].astype(float)
    history_days = pd.Series(np.arange(len(bars)), index=bars.index, dtype=float)
    median_dollar_volume = dollar_volume.rolling(20, min_periods=20).median().shift(1)
    next_open = bars["open"].astype(float).shift(-1)
    next_dividend = pd.to_numeric(
        bars.get("cash_dividend", pd.Series(0.0, index=bars.index)), errors="coerce"
    ).fillna(0.0).shift(-1)
    valid_prior_bar = (
        prior_close.notna()
        & np.isfinite(prior_close)
        & (prior_close > 0)
        & prior_volume.notna()
        & (prior_volume > 0)
    )
    eligible = pd.DataFrame({
        "timestamp": bars["timestamp"],
        "symbol": bars["symbol"],
        "listing_days_before_signal": history_days,
        "prior_close": prior_close,
        "prior_volume": prior_volume,
        "median_dollar_volume_20d": median_dollar_volume,
        "valid_prior_bar": valid_prior_bar,
        "correct_open_to_next_open_label": (next_open + next_dividend)
        / bars["open"].astype(float)
        - 1
        - 20.0 / 10_000,
        "label_end": bars["timestamp"].shift(-1),
    })
    eligible["pass_listing"] = eligible["listing_days_before_signal"] >= MIN_LISTING_DAYS
    eligible["pass_liquidity"] = (
        eligible["median_dollar_volume_20d"] >= MIN_MEDIAN_DOLLAR_VOLUME_20D
    )
    eligible["pass_price"] = eligible["prior_close"] >= MIN_PRICE
    eligible["investable"] = (
        eligible["pass_listing"]
        & eligible["pass_liquidity"]
        & eligible["pass_price"]
        & eligible["valid_prior_bar"]
    )

    open_to_next_open = next_open / bars["open"].astype(float) - 1
    gaps = pd.DataFrame({
        "timestamp": bars["timestamp"],
        "symbol": bars["symbol"],
        "open": bars["open"].astype(float),
        "next_open": next_open,
        "open_to_next_open": open_to_next_open,
        "signal_day_volume": bars["volume"].astype(float),
        "listing_days_before_signal": history_days,
        "median_dollar_volume_20d": median_dollar_volume,
    })
    gaps = gaps.loc[gaps["open_to_next_open"].abs() >= 0.30].copy()
    return eligible, gaps


def main() -> int:
    eligibility_parts: list[pd.DataFrame] = []
    gap_parts: list[pd.DataFrame] = []
    for path in sorted(DAILY.glob("US_*.parquet")):
        eligibility, gaps = eligibility_from_bars(path)
        if not eligibility.empty:
            eligibility_parts.append(eligibility)
        if not gaps.empty:
            gap_parts.append(gaps)
    if not eligibility_parts:
        raise FileNotFoundError(f"No adjusted daily histories found in {DAILY}")
    eligibility = pd.concat(eligibility_parts, ignore_index=True)
    gaps = pd.concat(gap_parts, ignore_index=True) if gap_parts else pd.DataFrame()
    panel = pd.read_parquet(SOURCE_PANEL)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    merged = panel.merge(
        eligibility,
        on=["timestamp", "symbol"],
        how="left",
        validate="many_to_one",
    )
    merged["investable"] = merged["investable"].fillna(False)
    merged["legacy_label_next_open_net"] = merged["label_next_open_net"]
    merged["label_next_open_net"] = merged["correct_open_to_next_open_label"]
    clean = merged.loc[
        merged["investable"] & merged["label_next_open_net"].notna()
    ].copy()
    legacy_error = (
        merged["legacy_label_next_open_net"] - merged["correct_open_to_next_open_label"]
    ).abs()
    audit_columns = [
        "listing_days_before_signal", "prior_close", "prior_volume",
        "median_dollar_volume_20d", "valid_prior_bar", "pass_listing",
        "pass_liquidity", "pass_price", "investable",
        "correct_open_to_next_open_label", "legacy_label_next_open_net",
    ]
    clean = clean.drop(columns=audit_columns)
    OUTPUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUTPUT_PANEL, index=False)
    eligibility.to_parquet(REPORTS / "investability_eligibility_history.parquet", index=False)
    if not gaps.empty:
        gaps = gaps.merge(
            eligibility[["timestamp", "symbol", "investable"]],
            on=["timestamp", "symbol"], how="left", validate="one_to_one",
        ).sort_values("open_to_next_open")
        gaps.to_csv(REPORTS / "open_gap_abs30pct_audit.csv", index=False)
    session_before = panel.groupby("timestamp")["symbol"].size()
    session_after = clean.groupby("timestamp")["symbol"].size()
    exclusion = {
        "missing_eligibility": int(merged["listing_days_before_signal"].isna().sum()),
        "listing": int((~merged["pass_listing"].fillna(False)).sum()),
        "liquidity": int((~merged["pass_liquidity"].fillna(False)).sum()),
        "price": int((~merged["pass_price"].fillna(False)).sum()),
        "invalid_prior_bar": int((~merged["valid_prior_bar"].fillna(False)).sum()),
    }
    audit = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "source_panel": str(SOURCE_PANEL),
        "output_panel": str(OUTPUT_PANEL),
        "rules": {
            "minimum_listing_days": MIN_LISTING_DAYS,
            "minimum_median_dollar_volume_20d": MIN_MEDIAN_DOLLAR_VOLUME_20D,
            "minimum_prior_close": MIN_PRICE,
            "valid_prior_bar_required": True,
            "all_inputs_lagged_before_signal_open": True,
        },
        "rows_before": int(len(panel)),
        "rows_after": int(len(clean)),
        "retained_pct": float(len(clean) / len(panel)),
        "symbols_before": int(panel["symbol"].nunique()),
        "symbols_after": int(clean["symbol"].nunique()),
        "sessions": int(clean["timestamp"].nunique()),
        "median_symbols_per_session_before": float(session_before.median()),
        "median_symbols_per_session_after": float(session_after.median()),
        "minimum_symbols_per_session_after": int(session_after.min()),
        "exclusion_counts_not_mutually_exclusive": exclusion,
        "extreme_open_to_next_open_abs30pct": int(len(gaps)),
        "extreme_gaps_investable": int(gaps["investable"].fillna(False).sum()) if not gaps.empty else 0,
        "hona_2026_06_15_investable": bool(
            eligibility.loc[
                (eligibility["symbol"] == "US.HONA")
                & (eligibility["timestamp"].dt.date == pd.Timestamp("2026-06-15").date()),
                "investable",
            ].any()
        ),
        "legacy_label_definition_detected": "close(t) to open(t+1), replaced",
        "legacy_label_mismatch_rows": int((legacy_error > 1e-10).sum()),
        "legacy_label_max_absolute_error": float(legacy_error.max()),
        "correct_label_definition": "open(t) to open(t+1), dividend included, net 20 bps",
    }
    (REPORTS / "investable_panel_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
