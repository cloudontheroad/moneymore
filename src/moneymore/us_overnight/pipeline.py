from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import USOvernightConfig
from .signals import MarketRiskGate, TransparentRiskGate, baseline_scores


def _local_stamp(session: date, clock: str, timezone: str) -> datetime:
    hour, minute = map(int, clock.split(":"))
    return datetime.combine(session, time(hour, minute), ZoneInfo(timezone))


def point_in_time_universe(
    universe: pd.DataFrame, session: date, cutoff: datetime, config: USOvernightConfig
) -> pd.DataFrame:
    frame = universe.copy()
    known = pd.to_datetime(frame["known_at"], utc=True)
    mask = (
        (pd.to_datetime(frame["session_date"]).dt.date == session)
        & (known <= pd.Timestamp(cutoff).tz_convert("UTC"))
        & (frame["security_type"] == "STOCK")
        & (frame["price"] >= config.minimum_price)
        & (frame["median_dollar_volume_20d"] >= config.minimum_median_dollar_volume_20d)
        & (frame["listing_days"] >= config.minimum_listing_days)
    )
    return (
        frame.loc[mask].sort_values(["symbol", "known_at"]).drop_duplicates("symbol", keep="last")
    )


def window_vwap(bars: pd.DataFrame, start: datetime, end: datetime) -> float:
    if bars.empty:
        return float("nan")
    stamp = pd.to_datetime(bars["timestamp"], utc=True)
    selected = bars.loc[
        (stamp >= pd.Timestamp(start).tz_convert("UTC"))
        & (stamp < pd.Timestamp(end).tz_convert("UTC"))
    ]
    valid = selected.loc[(selected["volume"] > 0) & selected["price"].notna()]
    if valid.empty:
        return float("nan")
    return float((valid["price"] * valid["volume"]).sum() / valid["volume"].sum())


def overnight_label(entry_vwap: float, exit_vwap: float, round_trip_cost_bps: float) -> float:
    if pd.isna(entry_vwap) or pd.isna(exit_vwap) or entry_vwap <= 0:
        return float("nan")
    return exit_vwap / entry_vwap - 1 - round_trip_cost_bps / 10_000


def generate_daily_candidates(
    features: pd.DataFrame,
    universe: pd.DataFrame,
    session: date,
    generated_at: datetime,
    config: USOvernightConfig,
    market: pd.Series | None = None,
    risk_gate: MarketRiskGate | None = None,
) -> pd.DataFrame:
    cutoff = _local_stamp(session, config.signal_cutoff, config.timezone)
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    if generated_at.astimezone(ZoneInfo(config.timezone)) < cutoff:
        raise ValueError("candidates cannot be generated before signal cutoff")
    eligible = point_in_time_universe(universe, session, cutoff, config)
    visible = features.copy()
    observed = pd.to_datetime(visible["observed_at"], utc=True)
    visible = visible.loc[observed <= pd.Timestamp(cutoff).tz_convert("UTC")]
    visible = visible.sort_values(["symbol", "observed_at"]).drop_duplicates("symbol", keep="last")
    panel = eligible[["symbol"]].merge(visible, on="symbol", how="inner")
    panel = panel.dropna(subset=["ret_1d", "ret_3d", "close_position", "last_30m_return"])
    decision = (risk_gate or TransparentRiskGate()).evaluate(
        market if market is not None else pd.Series(dtype=object)
    )
    ranked = baseline_scores(panel)
    ranked["risk_allowed"] = decision.allowed
    ranked["risk_scale"] = decision.scale
    ranked["risk_flags"] = "|".join(decision.flags)
    ranked["generated_at"] = generated_at.isoformat()
    ranked["session_date"] = session.isoformat()
    ranked["cost_bps"] = config.round_trip_cost_bps
    ranked = ranked.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
    if not decision.allowed:
        ranked = ranked.iloc[0:0]
    return ranked.head(config.top_n).reset_index(drop=True)


def write_candidates(
    frame: pd.DataFrame, session: date, config: USOvernightConfig
) -> tuple[Path, Path]:
    target = config.output_root / "candidates"
    target.mkdir(parents=True, exist_ok=True)
    csv_path, parquet_path = (
        target / f"{session.isoformat()}.csv",
        target / f"{session.isoformat()}.parquet",
    )
    frame.to_csv(csv_path, index=False)
    temporary = parquet_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(parquet_path)
    return csv_path, parquet_path
