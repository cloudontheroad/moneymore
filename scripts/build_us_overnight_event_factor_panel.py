"""Build causal sparse-event factors for the US open-to-open selector.

Exact-timestamp events become visible only at a later US regular open. Ratings
have date-only timestamps and therefore become visible on the next session.
Current plate membership is deliberately excluded from historical backtests.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.us_overnight.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/us_overnight_corrected/features/historical_panel_61_lag1.parquet"
EVENT_ROOT = ROOT / "data/us_overnight_event_warehouse"
OUT = ROOT / "data/us_overnight_corrected/features"

EARNINGS_COLUMNS = [
    "event_earnings_eps_surprise",
    "event_earnings_revenue_surprise",
    "event_earnings_agreement",
    "event_earnings_age_days",
    "event_earnings_decay_5d",
    "event_earnings_decay_20d",
]
RATING_COLUMNS = [
    "event_rating_direction",
    "event_rating_target_change",
    "event_rating_age_days",
    "event_rating_decay_20d",
]
MACRO_COLUMNS = [
    "event_macro_surprise",
    "event_macro_abs_surprise",
    "event_macro_age_days",
    "event_macro_high_importance",
]
FORWARD_EARNINGS_COLUMNS = [
    "forward_earnings_days_to_event",
    "forward_earnings_within_3d",
    "forward_earnings_within_10d",
    "forward_earnings_eps_expected_change",
    "forward_earnings_revenue_expected_change",
    "forward_earnings_iv",
    "forward_earnings_iv_rank",
    "forward_earnings_iv_percentile",
    "forward_earnings_option_attention",
]
EVENT_COLUMNS = EARNINGS_COLUMNS + RATING_COLUMNS + MACRO_COLUMNS


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    cleaned = cleaned.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _decision_table(panel: pd.DataFrame) -> pd.DataFrame:
    sessions = pd.DataFrame({"timestamp": sorted(panel["timestamp"].unique())})
    sessions["timestamp"] = pd.to_datetime(sessions["timestamp"], utc=True)
    local_midnight = sessions["timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    sessions["decision_open_utc"] = (local_midnight + timedelta(minutes=570)).dt.tz_convert("UTC")
    sessions["session_date"] = local_midnight.dt.date
    return sessions


def _asof_by_symbol(
    rows: pd.DataFrame,
    events: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    left = rows.sort_values(["symbol", "decision_open_utc"]).copy()
    right = events.sort_values(["symbol", "available_at_utc"])[
        ["symbol", "available_at_utc", *value_columns]
    ].copy()
    pieces = []
    for symbol, group in left.groupby("symbol", sort=False):
        source = right.loc[right["symbol"] == symbol].drop(columns="symbol")
        if source.empty:
            result = group.copy()
            for column in ["available_at_utc", *value_columns]:
                result[column] = pd.NaT if column == "available_at_utc" else np.nan
        else:
            result = pd.merge_asof(
                group.sort_values("decision_open_utc"),
                source.sort_values("available_at_utc"),
                left_on="decision_open_utc",
                right_on="available_at_utc",
                direction="backward",
                allow_exact_matches=False,
            )
        pieces.append(result)
    result = pd.concat(pieces, ignore_index=True)
    result["decision_open_utc"] = pd.to_datetime(result["decision_open_utc"], utc=True)
    result["available_at_utc"] = pd.to_datetime(result["available_at_utc"], utc=True)
    return result


def earnings_factors(rows: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_parquet(EVENT_ROOT / "earnings/canonical.parquet")
    events = events.rename(columns={"security": "symbol", "event_at_utc": "available_at_utc"})
    events["eps"] = pd.to_numeric(events["eps_surprise_pct"], errors="coerce").clip(-5, 5)
    events["revenue"] = pd.to_numeric(events["revenue_surprise_pct"], errors="coerce").clip(-2, 2)
    events["agreement"] = np.sign(events["eps"]) * np.sign(events["revenue"])
    merged = _asof_by_symbol(rows, events, ["eps", "revenue", "agreement"])
    age = (merged["decision_open_utc"] - merged["available_at_utc"]).dt.total_seconds() / 86400
    result = merged[["timestamp", "symbol"]].copy()
    result["event_earnings_eps_surprise"] = merged["eps"]
    result["event_earnings_revenue_surprise"] = merged["revenue"]
    result["event_earnings_agreement"] = merged["agreement"]
    result["event_earnings_age_days"] = age.clip(0, 120)
    result["event_earnings_decay_5d"] = merged["eps"] * np.exp(-age / 5)
    result["event_earnings_decay_20d"] = merged["revenue"] * np.exp(-age / 20)
    return result


def forward_earnings_factors(rows: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_parquet(EVENT_ROOT / "earnings/canonical.parquet")
    events = events.rename(columns={"security": "symbol", "event_at_utc": "event_time"})
    numeric = ["eps_predict", "revenue_predict", "eps_actual", "revenue_actual", "iv", "iv_rank", "iv_percentile", "option_volume"]
    for column in numeric:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    pieces = []
    for symbol, group in rows.sort_values(["symbol", "decision_open_utc"]).groupby("symbol", sort=False):
        calendar = events.loc[events["symbol"] == symbol].sort_values("event_time")
        left = group.sort_values("decision_open_utc").copy()
        if calendar.empty:
            merged = left
            for column in ["event_time", *numeric]:
                merged[column] = pd.NaT if column == "event_time" else np.nan
            merged["previous_eps_actual"] = np.nan
            merged["previous_revenue_actual"] = np.nan
        else:
            upcoming = calendar[["event_time", "eps_predict", "revenue_predict", "iv", "iv_rank", "iv_percentile", "option_volume"]]
            merged = pd.merge_asof(
                left, upcoming, left_on="decision_open_utc", right_on="event_time",
                direction="forward", allow_exact_matches=True,
            )
            history = calendar[["event_time", "eps_actual", "revenue_actual"]].rename(columns={
                "event_time": "previous_event_time", "eps_actual": "previous_eps_actual",
                "revenue_actual": "previous_revenue_actual",
            })
            merged = pd.merge_asof(
                merged.sort_values("decision_open_utc"), history,
                left_on="decision_open_utc", right_on="previous_event_time",
                direction="backward", allow_exact_matches=False,
            )
        pieces.append(merged)
    merged = pd.concat(pieces, ignore_index=True)
    merged["event_time"] = pd.to_datetime(merged["event_time"], utc=True)
    days = (merged["event_time"] - pd.to_datetime(merged["decision_open_utc"], utc=True)).dt.total_seconds() / 86400
    valid = days.between(0, 120)
    result = merged[["timestamp", "symbol"]].copy()
    result["forward_earnings_days_to_event"] = days.where(valid, 999.0)
    result["forward_earnings_within_3d"] = days.between(0, 3).astype(float)
    result["forward_earnings_within_10d"] = days.between(0, 10).astype(float)
    result["forward_earnings_eps_expected_change"] = (
        merged["eps_predict"] / merged["previous_eps_actual"].abs().replace(0, np.nan) - np.sign(merged["previous_eps_actual"])
    ).where(valid).clip(-5, 5)
    result["forward_earnings_revenue_expected_change"] = (
        merged["revenue_predict"] / merged["previous_revenue_actual"].replace(0, np.nan) - 1
    ).where(valid).clip(-2, 2)
    result["forward_earnings_iv"] = merged["iv"].where(valid).clip(0, 300)
    result["forward_earnings_iv_rank"] = merged["iv_rank"].where(valid).clip(0, 100)
    result["forward_earnings_iv_percentile"] = merged["iv_percentile"].where(valid).clip(0, 100)
    result["forward_earnings_option_attention"] = np.log1p(merged["option_volume"].clip(lower=0)).where(valid)
    return result


def rating_factors(rows: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_parquet(EVENT_ROOT / "ratings/canonical.parquet").rename(columns={"security": "symbol"})
    dates = sorted(sessions["session_date"])
    event_dates = pd.to_datetime(events["recommendation_date"], errors="coerce").dt.date
    next_date = event_dates.map(lambda value: next((date for date in dates if value is not None and date > value), None))
    opening = sessions.set_index("session_date")["decision_open_utc"]
    events["available_at_utc"] = next_date.map(opening) - timedelta(microseconds=1)
    change = events["change_type"].astype("string").str.upper()
    events["direction"] = np.select(
        [change.str.contains("UP", na=False), change.str.contains("DOWN", na=False)], [1.0, -1.0], default=0.0
    )
    target = pd.to_numeric(events["target_price"], errors="coerce")
    previous = pd.to_numeric(events["last_target_price"], errors="coerce")
    events["target_change"] = (target / previous.replace(0, np.nan) - 1).clip(-2, 2)
    events = events.dropna(subset=["available_at_utc"])
    merged = _asof_by_symbol(rows, events, ["direction", "target_change"])
    age = (merged["decision_open_utc"] - merged["available_at_utc"]).dt.total_seconds() / 86400
    result = merged[["timestamp", "symbol"]].copy()
    result["event_rating_direction"] = merged["direction"]
    result["event_rating_target_change"] = merged["target_change"]
    result["event_rating_age_days"] = age.clip(0, 180)
    result["event_rating_decay_20d"] = merged["direction"] * np.exp(-age / 20)
    return result


def macro_factors(rows: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_parquet(EVENT_ROOT / "economic/canonical.parquet").rename(columns={"event_at_utc": "available_at_utc"})
    actual, consensus = _numeric(events["actual"]), _numeric(events["consensus"])
    scale = _numeric(events["previous"]).abs().replace(0, np.nan)
    raw = (actual - consensus) / scale
    events["surprise"] = raw.clip(-5, 5)
    events["high_importance"] = events["star"].astype("string").str.upper().eq("HIGH").astype(float)
    events = events.dropna(subset=["available_at_utc", "surprise"]).sort_values("available_at_utc")
    left = rows[["timestamp", "decision_open_utc"]].drop_duplicates().sort_values("decision_open_utc")
    merged = pd.merge_asof(
        left, events[["available_at_utc", "surprise", "high_importance"]],
        left_on="decision_open_utc", right_on="available_at_utc", direction="backward", allow_exact_matches=False,
    )
    age = (merged["decision_open_utc"] - merged["available_at_utc"]).dt.total_seconds() / 86400
    merged["event_macro_surprise"] = merged["surprise"]
    merged["event_macro_abs_surprise"] = merged["surprise"].abs()
    merged["event_macro_age_days"] = age.clip(0, 30)
    merged["event_macro_high_importance"] = merged["high_importance"]
    return merged[["timestamp", *MACRO_COLUMNS]]


def main() -> int:
    panel = pd.read_parquet(BASE)
    sessions = _decision_table(panel)
    rows = panel[["timestamp", "symbol"]].merge(sessions, on="timestamp", how="left")
    factors = earnings_factors(rows)
    factors = factors.merge(rating_factors(rows, sessions), on=["timestamp", "symbol"], how="left")
    factors = factors.merge(macro_factors(rows), on="timestamp", how="left")
    factors = factors.merge(forward_earnings_factors(rows), on=["timestamp", "symbol"], how="left")
    for column in EVENT_COLUMNS:
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
        if column.endswith("age_days"):
            factors[column] = factors[column].fillna(999.0)
        else:
            factors[column] = factors[column].fillna(0.0)
    enriched = panel.merge(factors, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    for column in FORWARD_EARNINGS_COLUMNS:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
        enriched[column] = enriched[column].fillna(999.0 if column.endswith("days_to_event") else 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    panel_path = OUT / "historical_panel_61_plus_event_v1.parquet"
    features_path = OUT / "feature_columns_61_plus_event_v1.json"
    enriched.to_parquet(panel_path, index=False)
    features_path.write_text(json.dumps([*FEATURE_COLUMNS, *EVENT_COLUMNS], indent=2), encoding="utf-8")
    for group_name, columns in {
        "earnings": EARNINGS_COLUMNS,
        "ratings": RATING_COLUMNS,
        "macro": MACRO_COLUMNS,
        "forward_earnings": FORWARD_EARNINGS_COLUMNS,
    }.items():
        (OUT / f"feature_columns_61_plus_event_{group_name}_v1.json").write_text(
            json.dumps([*FEATURE_COLUMNS, *columns], indent=2), encoding="utf-8"
        )
    audit = {
        "panel": str(panel_path), "rows": len(enriched), "sessions": enriched["timestamp"].nunique(),
        "symbols": enriched["symbol"].nunique(), "base_features": len(FEATURE_COLUMNS),
        "event_features": EVENT_COLUMNS,
        "forward_earnings_features": FORWARD_EARNINGS_COLUMNS,
        "nonzero_rate": {column: float((enriched[column] != 0).mean()) for column in EVENT_COLUMNS if not column.endswith("age_days")},
        "causal_policy": "exact events strictly before open(t); date-only ratings from next session",
        "excluded": ["web_events_not_yet_available", "historical_industry_diffusion_not_point_in_time"],
    }
    (OUT / "event_factor_v1_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
