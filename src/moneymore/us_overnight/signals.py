from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from .models import RiskDecision


class MarketRiskGate(Protocol):
    def evaluate(self, market: pd.Series) -> RiskDecision: ...


class TransparentRiskGate:
    def evaluate(self, market: pd.Series) -> RiskDecision:
        flags: list[str] = []
        if bool(market.get("macro_event", False)):
            flags.append("MACRO_EVENT")
        if float(market.get("spy_last_30m_return", 0.0)) <= -0.015:
            flags.append("EXTREME_MARKET_SELL_OFF")
        if float(market.get("vix_level", 0.0)) >= 35:
            flags.append("HIGH_VIX")
        blocked = "EXTREME_MARKET_SELL_OFF" in flags
        return RiskDecision(
            allowed=not blocked,
            scale=0.0 if blocked else (0.5 if flags else 1.0),
            flags=tuple(flags),
        )


def baseline_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Transparent falling-deceleration plus late-session recovery score."""
    required = {"ret_1d", "ret_3d", "close_position", "last_30m_return"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"missing signal columns: {sorted(missing)}")
    result = frame.copy()
    falling = (-result["ret_1d"]).clip(lower=0)
    deceleration = (result["ret_1d"] - result["ret_3d"] / 3).clip(lower=0)
    result["signal_falling_deceleration"] = falling * deceleration
    result["signal_late_recovery"] = (
        result["close_position"].clip(0, 1) + result["last_30m_return"].clip(lower=0) * 10
    )
    for column in ("signal_falling_deceleration", "signal_late_recovery"):
        result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    result["score"] = 0.6 * result["signal_falling_deceleration"].rank(
        pct=True, method="average"
    ) + 0.4 * result["signal_late_recovery"].rank(pct=True, method="average")
    return result
