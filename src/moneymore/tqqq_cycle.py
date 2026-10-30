"""Cycle-aware TQQQ sizing, independently implemented from public research.

The model deliberately separates a slow macro/cycle target from execution.
It is long-only: inverse daily-reset ETFs are not a persistent hedge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CycleDecision:
    target_tqqq: float
    raw_target: float
    regime: str
    combined_rank: float
    fear_rank: float
    rate_rank: float
    term_structure: float
    rate_change_60d: float
    ma250: float
    distance_from_ma250: float
    crash_guard_active: bool
    rebalance_band: float


@dataclass(frozen=True)
class CycleParameters:
    rank_window: int = 1260
    minimum_rank_history: int = 504
    base_exposure: float = 0.45
    cycle_slope: float = 1.0
    minimum_exposure: float = 0.10
    maximum_exposure: float = 1.0
    crash_ma_days: int = 250
    crash_distance: float = -0.25
    crash_multiplier: float = 0.50
    rebalance_band: float = 0.12
    rebalance_frequency: str = "monthly"


def trailing_percentile(series: pd.Series, window: int, minimum: int) -> pd.Series:
    """Percentile of the current observation using trailing data only."""

    minimum = min(minimum, window)

    def rank_last(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.nan
        current = finite[-1]
        # Mid-rank makes a flat series neutral (50), rather than maximally high.
        return float(
            (np.sum(finite < current) + 0.5 * np.sum(finite == current))
            / len(finite)
            * 100.0
        )

    return series.astype(float).rolling(window, min_periods=minimum).apply(
        rank_last, raw=True
    )


def cycle_decisions(
    qqq_close: pd.Series,
    vix_close: pd.Series,
    rate_10y: pd.Series,
    vix3m_close: pd.Series | None = None,
    params: CycleParameters | None = None,
) -> pd.DataFrame:
    """Build point-in-time targets for every date without future leakage."""

    params = params or CycleParameters()
    qqq = qqq_close.astype(float)
    vix = vix_close.reindex(qqq.index).ffill().astype(float)
    rate = rate_10y.reindex(qqq.index).ffill().astype(float)
    vix_rank = trailing_percentile(vix, params.rank_window, params.minimum_rank_history)
    if vix3m_close is not None:
        vix3m = vix3m_close.reindex(qqq.index).ffill().astype(float)
        term = vix / vix3m.replace(0, np.nan)
        term_rank = trailing_percentile(
            term, params.rank_window, params.minimum_rank_history
        )
        fear_rank = term_rank.combine_first(vix_rank)
    else:
        term = pd.Series(np.nan, index=qqq.index)
        fear_rank = vix_rank

    rate_change = rate.diff(60)
    rate_pressure_rank = trailing_percentile(
        rate_change, params.rank_window, params.minimum_rank_history
    )
    rate_rank = 100.0 - rate_pressure_rank
    combined = (fear_rank + rate_rank) / 2.0
    ma250 = qqq.rolling(params.crash_ma_days).mean()
    distance = qqq / ma250 - 1.0
    raw = params.base_exposure + params.cycle_slope * (combined / 100.0 - 0.5)
    raw = raw.clip(params.minimum_exposure, params.maximum_exposure)
    guarded = distance < params.crash_distance
    desired = raw.where(~guarded, raw * params.crash_multiplier).clip(
        params.minimum_exposure, params.maximum_exposure
    )

    implemented = np.full(len(desired), np.nan)
    current = np.nan
    prior_guarded = False
    for offset, value in enumerate(desired.to_numpy(dtype=float)):
        if not np.isfinite(value):
            continue
        guard_now = bool(guarded.iloc[offset])
        guard_transition = guard_now and not prior_guarded
        if params.rebalance_frequency != "monthly":
            raise ValueError("only calendar-month rebalancing is supported")
        cadence_ready = offset == 0 or qqq.index[offset].to_period("M") != qqq.index[
            offset - 1
        ].to_period("M")
        if (
            not np.isfinite(current)
            or guard_transition
            or (cadence_ready and abs(value - current) >= params.rebalance_band)
        ):
            current = value
        implemented[offset] = current
        prior_guarded = guard_now

    regime = np.select(
        [
            guarded.fillna(False),
            (fear_rank >= 70) & (rate_rank >= 45),
            rate_rank <= 30,
            combined >= 60,
        ],
        ["STRUCTURAL_CRASH", "FEAR_REBOUND", "RATE_PRESSURE", "RISK_ON"],
        default="BALANCED",
    )
    return pd.DataFrame(
        {
            "target_tqqq": implemented,
            "raw_target": desired,
            "regime": regime,
            "combined_rank": combined,
            "fear_rank": fear_rank,
            "rate_rank": rate_rank,
            "term_structure": term,
            "rate_change_60d": rate_change,
            "ma250": ma250,
            "distance_from_ma250": distance,
            "crash_guard_active": guarded.fillna(False),
        },
        index=qqq.index,
    )


def latest_cycle_decision(
    qqq_close: pd.Series,
    vix_close: pd.Series,
    rate_10y: pd.Series,
    vix3m_close: pd.Series | None = None,
    previous_target: float | None = None,
    params: CycleParameters | None = None,
) -> CycleDecision:
    params = params or CycleParameters()
    decisions = cycle_decisions(qqq_close, vix_close, rate_10y, vix3m_close, params)
    row = decisions.iloc[-1]
    if pd.isna(row["raw_target"]):
        raise RuntimeError("cycle strategy needs more trailing macro history")
    raw_target = float(row["raw_target"])
    latest_is_month_open = len(qqq_close) == 1 or qqq_close.index[-1].to_period(
        "M"
    ) != qqq_close.index[-2].to_period("M")
    if (
        previous_target is None
        or bool(row["crash_guard_active"])
        or (
            latest_is_month_open
            and abs(raw_target - previous_target) >= params.rebalance_band
        )
    ):
        target = raw_target
    else:
        target = previous_target
    return CycleDecision(
        target_tqqq=float(np.clip(target, 0.0, 1.0)),
        raw_target=raw_target,
        regime=str(row["regime"]),
        combined_rank=float(row["combined_rank"]),
        fear_rank=float(row["fear_rank"]),
        rate_rank=float(row["rate_rank"]),
        term_structure=float(row["term_structure"]),
        rate_change_60d=float(row["rate_change_60d"]),
        ma250=float(row["ma250"]),
        distance_from_ma250=float(row["distance_from_ma250"]),
        crash_guard_active=bool(row["crash_guard_active"]),
        rebalance_band=params.rebalance_band,
    )
