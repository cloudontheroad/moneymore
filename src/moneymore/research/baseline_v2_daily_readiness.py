"""Exact-session inputs for daily research, never a monthly-score forward fill."""

from __future__ import annotations

import pandas as pd

FIELDS = {"ts_code", "trade_date", "close", "pb", "pe_ttm", "dv_ttm", "total_mv"}


def daily_schedule(calendar, start, end):
    dates = sorted({str(d) for d in calendar})
    return [(d, dates[i - 1]) for i, d in enumerate(dates) if i > 0 and start <= d <= end]


def validate_snapshot(frame, date):
    if frame is None or not FIELDS.issubset(frame.columns):
        raise ValueError("MISSING_DAILY_SCHEMA")
    if not 1000 <= len(frame) < 6000:
        raise ValueError("INCOMPLETE_OR_TRUNCATED_DAILY_SNAPSHOT")
    if not frame.trade_date.astype(str).eq(date).all() or frame.ts_code.duplicated().any():
        raise ValueError("DAILY_DATE_OR_DUPLICATE_ERROR")
    cap = pd.to_numeric(frame.total_mv, errors="coerce")
    codes = frame.ts_code.astype(str)
    mainland = (codes.str.endswith(".SH") & codes.str.startswith(("60", "68"))) | (
        codes.str.endswith(".SZ") & codes.str.startswith(("00", "30"))
    )
    top = frame.loc[mainland & cap.gt(0)].copy()
    top["total_mv"] = cap.loc[top.index]
    top = top.sort_values(["total_mv", "ts_code"], ascending=[False, True]).head(1000)
    if len(top) != 1000:
        raise ValueError("INCOMPLETE_MAINLAND_TOP1000")
    return top


def classify_held_inputs(actual_holdings, scored_symbols, *, universe_members=None):
    """An explicit complete pool is needed to distinguish exclusion from missing data."""
    held = {str(s) for s, quantity in actual_holdings.items() if quantity > 0}
    scored = set(scored_symbols)
    if universe_members is None:
        return {"universe_exits": [], "input_faults": sorted(held - scored)}
    universe = set(universe_members)
    if not universe or not scored.issubset(universe):
        raise ValueError("INVALID_COMPLETE_CANDIDATE_UNIVERSE")
    return {
        "universe_exits": sorted(held - universe),
        "input_faults": sorted((held & universe) - scored),
    }


def construct_daily_targets(
    scores, actual_holdings, prices, signal_date, execution_date, policy, *, universe_members=None
):
    """Replay adapter: actual incumbents, exact dated scores, no future correlation."""
    from ..portfolio_constructor import global_topk_portfolio, trailing_return_correlation

    signal = pd.Timestamp(signal_date)
    if signal >= pd.Timestamp(execution_date):
        raise ValueError("SIGNAL_MUST_PRECEDE_EXECUTION")
    if "date" not in scores or not pd.to_datetime(scores.date).eq(signal).all():
        raise ValueError("EXACT_DAILY_SCORES_REQUIRED")
    if scores.symbol.duplicated().any() or scores.score.isna().any():
        raise ValueError("INVALID_DAILY_SCORES")
    held = {str(s) for s, quantity in actual_holdings.items() if quantity > 0}
    classification = classify_held_inputs(
        actual_holdings, scores.symbol, universe_members=universe_members
    )
    if classification["input_faults"]:
        raise ValueError("HELD_V2_INPUT_INCOMPLETE_HOLD_AND_ALERT")
    p = policy["portfolio"]
    correlation = trailing_return_correlation(
        prices, signal, lookback=int(p["correlation_lookback"])
    )
    selected, targets, ranking = global_topk_portfolio(
        scores,
        held,
        symbol_column="symbol",
        top_k=int(p["top_k"]),
        exit_rank=int(p["exit_rank"]),
        max_replacements=int(p["max_replacements"]),
        gross_exposure=float(p["gross_exposure"]),
        minimum_weight=float(p["minimum_weight"]),
        maximum_weight=float(p["maximum_weight"]),
        correlation=correlation,
        cluster_correlation_threshold=float(p["cluster_correlation_threshold"]),
        maximum_cluster_members=int(p["maximum_cluster_members"]),
        weighting_method=p["weighting_method"],
    )
    if (
        len(selected) != int(p["top_k"])
        or abs(sum(targets.values()) - float(p["gross_exposure"])) > 1e-6
    ):
        raise ValueError("INCOMPLETE_V2_TARGET_SUPPORT")
    return selected, targets, ranking
