"""Independent TQQQ/cash allocation by scenario-based log-wealth optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEIGHTS = np.linspace(0, 1, 5)
RISK_FLOOR = 0.75
RISK_FREQUENCY = 0.20
RISK_EXPONENT = np.log(RISK_FREQUENCY) / np.log(RISK_FLOOR)
MIN_SCENARIOS = 18
PRACTICE_START = "2013-01-02"


@dataclass(frozen=True)
class Scenarios:
    gross: np.ndarray
    volatility_scaled: np.ndarray
    reference: np.ndarray
    adverse: np.ndarray
    count: int
    effective_count: float


@dataclass(frozen=True)
class Allocation:
    weight: float
    keep: bool
    expected_log_growth: float
    risk_moment: float


def market_features(market: pd.DataFrame) -> pd.DataFrame:
    """Continuous observations used to estimate returns, never as position rules."""
    price = market.qqq_close
    log = np.log(price)
    vol = log.diff().rolling(63).std()
    return pd.DataFrame({
        "medium_trend": log.diff(126) / (vol * np.sqrt(126)),
        "near_peak": np.log(price / price.rolling(21).max()) / (vol * np.sqrt(21)),
        "vol_term": np.log(market.vix / market.vix3m),
    })


def scenario_book(market: pd.DataFrame) -> dict[pd.Timestamp, Scenarios]:
    """Estimate next-month outcomes only from completed, nonoverlapping months."""
    features = market_features(market).to_numpy()
    months = market.index.to_period("M")
    opens = market.tqqq_open.to_numpy()
    returns = np.log(market.tqqq_close).diff()
    volatility = pd.concat([returns.rolling(days).std() for days in (20, 60)], axis=1).max(axis=1).to_numpy()
    boundaries = np.flatnonzero(~months.duplicated())
    entries, ends = boundaries[:-1], boundaries[1:]
    valid = (entries > 0) & np.isfinite(features[np.maximum(entries - 1, 0)]).all(axis=1)
    entries, ends = entries[valid], ends[valid]
    all_gross = opens[ends] / opens[entries]
    all_features = features[entries - 1]
    historical_volatility = volatility[entries - 1]
    result = {}
    for i in range(1, len(market)):
        # Even the latest outcome must have ended before the decision close.
        count = int(np.searchsorted(ends, i))
        query = features[i - 1]
        if count < MIN_SCENARIOS or not np.isfinite(query).all():
            continue
        gross, history = all_gross[:count], all_features[:count]
        scale = np.subtract(*np.quantile(history, [0.75, 0.25], axis=0))
        scaled = (history - query) / np.maximum(scale, 1e-8)
        distance = np.linalg.norm(scaled, axis=1)
        # Use a soft nearest-neighbor distribution, with sample-size shrinkage.
        bandwidth = max(0.5, float(np.partition(distance, min(11, count - 1))[min(11, count - 1)]))
        similar = np.exp(-0.5 * (distance / bandwidth) ** 2)
        similar /= similar.sum()
        effective = float(1 / (similar @ similar))
        shrink = effective / (effective + 12)
        reference = shrink * similar + (1 - shrink) / count
        # A transparent ambiguity scenario: modest extra mass on historical left-tail outcomes.
        tail = gross <= np.quantile(gross, 0.10)
        adverse = 0.95 * reference + 0.05 * tail / tail.sum()
        vol_ratio = np.clip(volatility[i - 1] / historical_volatility[:count], 0.5, 2)
        volatility_scaled = np.exp(np.log(gross) * vol_ratio)
        result[market.index[i]] = Scenarios(gross, volatility_scaled, reference, adverse, count, effective)
    return result


def choose_allocation(
    scenarios: Scenarios, current: float, cost_bps: float, mode: str, *, volatility_scaled: bool = False,
) -> Allocation:
    """Return the best permissible weight, including the existing allocation as an action."""
    binary = mode == "binary_robust"
    candidates = np.r_[current, [0.0, 1.0] if binary else WEIGHTS]
    cost = cost_bps / 10_000
    difference = candidates - current
    trade_cost = cost * np.abs(difference) / (1 + np.sign(difference) * candidates * cost)
    gross = scenarios.volatility_scaled if volatility_scaled else scenarios.gross
    payoff = (1 - trade_cost) * (1 + np.outer(gross - 1, candidates))
    logs = np.log(payoff)
    model = np.full(scenarios.count, 1.0 / scenarios.count) if mode == "pooled" else scenarios.reference
    normal_growth = model @ logs
    normal_risk = model @ np.exp(-RISK_EXPONENT * logs)
    if mode in {"robust", "binary_robust"}:
        adverse = scenarios.adverse
        if volatility_scaled:
            tail = gross <= np.quantile(gross, 0.10)
            adverse = 0.95 * scenarios.reference + 0.05 * tail / tail.sum()
        objective = np.minimum(normal_growth, adverse @ logs)
        risk = np.maximum(normal_risk, adverse @ np.exp(-RISK_EXPONENT * logs))
    else:
        objective, risk = normal_growth, normal_risk
    if mode in {"constrained", "robust", "binary_robust"}:
        # If cash itself breaches the moment bound because of exit fees, still allow
        # liquidation. A cost paid now is separate from future risky exposure.
        permitted = (risk <= 1 + 1e-12) | (candidates == 0)
        objective = np.where(permitted, objective, -np.inf)
    index = int(np.argmax(objective))
    return Allocation(float(candidates[index]), index == 0, float(objective[index]), float(risk[index]))
