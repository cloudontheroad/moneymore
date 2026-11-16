"""Pure cross-sectional ranking and equal-slot research portfolio construction.

Scores are ordinal selection signals, never forecasts of monetary returns.  All
inputs must already be point-in-time snapshots: this module does no data loading,
date selection, covariance estimation, trading, or relaxation of sector limits.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from math import floor
from numbers import Integral

import numpy as np
import pandas as pd

_TOLERANCE = 1e-12


def _cross_section(scores: pd.Series, industries: pd.Series) -> tuple[pd.Series, pd.Series]:
    if not isinstance(scores, pd.Series) or not isinstance(industries, pd.Series):
        raise TypeError("SCORES_AND_INDUSTRIES_MUST_BE_SERIES")
    if scores.index.nlevels != 1 or industries.index.nlevels != 1:
        raise ValueError("SINGLE_CROSS_SECTION_REQUIRED")
    if not scores.index.is_unique or not industries.index.is_unique:
        raise ValueError("DUPLICATE_CROSS_SECTION_INDEX")
    if any(not isinstance(symbol, str) or not symbol for symbol in scores.index):
        raise ValueError("SYMBOLS_MUST_BE_NONEMPTY_STRINGS")
    numeric = pd.to_numeric(scores, errors="coerce").astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("NONFINITE_SCORE")
    aligned = industries.reindex(scores.index)
    if aligned.isna().any() or any(
        not isinstance(sector, str) or not sector for sector in aligned
    ):
        raise ValueError("MISSING_OR_INVALID_INDUSTRY")
    return numeric, aligned


def sector_relative_scores(scores: pd.Series, industries: pd.Series) -> pd.Series:
    """Average within-industry percent ranks; a singleton receives neutral 0.5.

    Tied observations have identical scores.  Symbol ordering is never used to
    manufacture alpha.  An independent increasing transform of any sector's raw
    scores cannot affect the result.  Pass only the current score cross-section;
    multi-date panels are rejected rather than accidentally ranked across time.
    """
    scores, industries = _cross_section(scores, industries)
    result = scores.groupby(industries, sort=False).rank(method="average", pct=True)
    counts = industries.map(industries.value_counts())
    result.loc[counts.eq(1)] = 0.5
    result.name = scores.name
    return result


def _weights(values: Mapping | pd.Series | None, name: str) -> dict[str, float]:
    if values is None:
        return {}
    if isinstance(values, pd.Series) and not values.index.is_unique:
        raise ValueError(f"DUPLICATE_{name}_INDUSTRY")
    result = dict(values)
    for sector, weight in result.items():
        if not isinstance(sector, str) or not sector:
            raise ValueError(f"INVALID_{name}_INDUSTRY")
        if not np.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError(f"INVALID_{name}_WEIGHT")
    return {sector: float(weight) for sector, weight in result.items()}


def allocate_ranked(
    scores: pd.Series,
    industries: pd.Series,
    benchmark_industry: Mapping | pd.Series,
    previous_symbols: Iterable[str],
    *,
    slots: int = 50,
    slot_weight: float = 0.0199,
    industry_active_limit: float = 0.08,
    frozen_industry_weights: Mapping | pd.Series | None = None,
    max_optional_exits: int | None = None,
) -> tuple[dict[str, float], dict]:
    """Select equal-weight names subject to hard upper industry active limits.

    Without a buffer, descending-score partition-cap greedy exactly maximizes
    score among maximum-cardinality feasible selections.  With a buffer, retain
    the best feasible old holdings, fill vacancies, and perform at most the
    requested number of strictly improving feasible old-for-new swaps.  This is
    deliberately a membership buffer, not a transaction-cost or return forecast
    optimizer.  Ties are broken by symbol only for deterministic selection, and
    never cause optional exits.  The buffer is not a global optimizer under a
    turnover constraint, nor an assertion of exact Qlib cash-weight semantics.

    ``slots * slot_weight`` is the total exposure budget (default 0.995), BEFORE
    frozen weights.  Frozen names must be absent from scores/previous_symbols;
    frozen weights consume both this budget and their sector capacities.  A
    fractional residual slot remains cash.  Returned targets cover only movable
    names.  Benchmark and frozen sector weights are absolute net-asset weights.

    Missing held scores and already-infeasible frozen weights raise explicitly.
    Insufficient investable capacity returns a partial portfolio with audit
    status INSUFFICIENT_CAPACITY; caps and slot weights are never relaxed.
    """
    scores, industries = _cross_section(scores, industries)
    if not isinstance(slots, Integral) or isinstance(slots, bool) or slots < 0:
        raise ValueError("INVALID_SLOTS")
    slots = int(slots)
    if not np.isfinite(slot_weight) or slot_weight <= 0:
        raise ValueError("INVALID_SLOT_WEIGHT")
    if not np.isfinite(industry_active_limit) or industry_active_limit < 0:
        raise ValueError("INVALID_INDUSTRY_ACTIVE_LIMIT")
    if max_optional_exits is not None and (
        not isinstance(max_optional_exits, Integral)
        or isinstance(max_optional_exits, bool)
        or max_optional_exits < 0
    ):
        raise ValueError("INVALID_MAX_OPTIONAL_EXITS")
    target_exposure = slots * float(slot_weight)
    if target_exposure > 1 + _TOLERANCE:
        raise ValueError("TARGET_EXPOSURE_EXCEEDS_ONE")
    benchmark = _weights(benchmark_industry, "BENCHMARK")
    frozen = _weights(frozen_industry_weights, "FROZEN")
    if sum(benchmark.values()) > 1 + _TOLERANCE:
        raise ValueError("BENCHMARK_EXPOSURE_EXCEEDS_ONE")
    frozen_total = sum(frozen.values())
    if frozen_total > target_exposure + _TOLERANCE:
        raise ValueError("FROZEN_EXPOSURE_EXCEEDS_TARGET")
    previous = set(previous_symbols)
    if previous - set(scores.index):
        raise ValueError("HELD_SCORE_MISSING_NOT_AN_EXIT")

    sector_of = industries.to_dict()
    score_of = scores.to_dict()
    ranked = sorted(scores.index, key=lambda symbol: (-score_of[symbol], symbol))
    sectors = sorted(set(industries) | set(benchmark) | set(frozen))
    weight_caps = {
        sector: benchmark.get(sector, 0.0) + float(industry_active_limit)
        for sector in sectors
    }
    if any(frozen.get(sector, 0.0) > weight_caps[sector] + _TOLERANCE for sector in sectors):
        raise ValueError("FROZEN_INDUSTRY_CAP_VIOLATION")
    caps = {
        sector: max(
            0,
            floor((weight_caps[sector] - frozen.get(sector, 0.0) + _TOLERANCE)
                  / slot_weight),
        )
        for sector in sectors
    }
    available_budget = max(0.0, target_exposure - frozen_total)
    effective_slots = min(slots, floor((available_budget + _TOLERANCE) / slot_weight))
    candidates_per_sector = Counter(industries)
    capacity = sum(min(caps[sector], candidates_per_sector[sector]) for sector in sectors)
    goal_count = min(effective_slots, capacity)

    # Minimum unavoidable exits are determined before optional score upgrades.
    # Sector repairs take precedence over global slot-budget repairs in the audit.
    feasible_previous: list[str] = []
    previous_counts: Counter = Counter()
    sector_forced: list[str] = []
    for symbol in ranked:
        if symbol not in previous:
            continue
        sector = sector_of[symbol]
        if previous_counts[sector] >= caps[sector]:
            sector_forced.append(symbol)
        else:
            feasible_previous.append(symbol)
            previous_counts[sector] += 1
    slot_forced = feasible_previous[goal_count:]
    retained_base = feasible_previous[:goal_count]
    forced = set(sector_forced) | set(slot_forced)

    selected = set(retained_base) if max_optional_exits is not None else set()
    counts = Counter(sector_of[symbol] for symbol in selected)
    for symbol in ranked:
        if len(selected) >= goal_count:
            break
        sector = sector_of[symbol]
        if symbol not in selected and counts[sector] < caps[sector]:
            selected.add(symbol)
            counts[sector] += 1

    swaps: list[dict] = []
    if max_optional_exits is not None:
        for _ in range(int(max_optional_exits)):
            # Use the greatest feasible score improvement.  Lexical tie-breaking
            # in the key makes the outcome independent of input/set iteration.
            best = None
            outgoing_candidates = sorted(selected & previous)
            for incoming in ranked:
                if incoming in selected or incoming in previous:
                    continue
                sector = sector_of[incoming]
                for outgoing in outgoing_candidates:
                    if sector != sector_of[outgoing] and counts[sector] >= caps[sector]:
                        continue
                    gain = score_of[incoming] - score_of[outgoing]
                    if gain <= 0:
                        continue
                    key = (-gain, incoming, outgoing)
                    if best is None or key < best[0]:
                        best = (key, incoming, outgoing, gain)
            if best is None:
                break
            _, incoming, outgoing, gain = best
            selected.remove(outgoing)
            selected.add(incoming)
            counts[sector_of[outgoing]] -= 1
            counts[sector_of[incoming]] += 1
            swaps.append({"incoming": incoming, "outgoing": outgoing, "score_gain": gain})

    targets = {symbol: float(slot_weight) for symbol in ranked if symbol in selected}
    removed = previous - selected
    entered = selected - previous
    selected_weights = {sector: counts[sector] * float(slot_weight) for sector in sectors}
    total_industry_weights = {
        sector: selected_weights[sector] + frozen.get(sector, 0.0) for sector in sectors
    }
    selected_weight = len(targets) * float(slot_weight)
    total_weight = selected_weight + frozen_total
    audit = {
        "status": "OK" if len(selected) == effective_slots else "INSUFFICIENT_CAPACITY",
        "method": (
            "max_score_partition_greedy" if max_optional_exits is None
            else "max_retention_then_beneficial_swaps"
        ),
        "requested_slots": slots,
        "effective_slots": effective_slots,
        "selected_count": len(targets),
        "available_capacity": capacity,
        "slot_weight": float(slot_weight),
        "target_exposure": target_exposure,
        "available_budget": available_budget,
        "selected_weight": selected_weight,
        "frozen_weight": frozen_total,
        "total_weight": total_weight,
        "cash_weight": max(0.0, 1.0 - total_weight),
        "unallocated_budget": max(0.0, target_exposure - total_weight),
        "industry_active_limit": float(industry_active_limit),
        "industry_count_caps": caps,
        "industry_weight_caps": weight_caps,
        "benchmark_industry_weights": benchmark,
        "frozen_industry_weights": frozen,
        "selected_industry_weights": selected_weights,
        "total_industry_weights": total_industry_weights,
        "industry_active_weights": {
            sector: total_industry_weights[sector] - benchmark.get(sector, 0.0)
            for sector in sectors
        },
        "risk_forced_exits": sorted(sector_forced),
        "slot_capacity_forced_exits": sorted(slot_forced),
        "optional_exits": sorted(removed - forced),
        "max_optional_exits": None if max_optional_exits is None else int(max_optional_exits),
        "removed_symbols": sorted(removed),
        "new_symbols": sorted(entered),
        "retained_symbols": sorted(selected & previous),
        "membership_turnover": {"entries": len(entered), "exits": len(removed)},
        "beneficial_swaps": swaps,
    }
    return targets, audit
