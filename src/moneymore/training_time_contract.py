"""Causal date planning; does not certify market data completeness or fit a model."""
from __future__ import annotations

import pandas as pd


def weekly_time_contract(dates, *, as_of, horizon=5, valid_days=126, test_days=126):
    """Keep research holdouts separate from a latest-matured deployment refit.

    Input dates must be an independently verified complete trading calendar.
    A label starting at index i consumes the close at i+horizon. Purging
    ensures that close precedes the next research segment's first signal.
    """
    if min(horizon, valid_days, test_days) < 1:
        raise ValueError("horizon and segment sizes must be positive")
    cutoff = pd.Timestamp(as_of).normalize()
    calendar = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    calendar = calendar[calendar <= cutoff]
    mature_count = len(calendar) - horizon
    test_start = mature_count - test_days
    valid_start = test_start - horizon - valid_days
    train_stop = valid_start - horizon
    if train_stop < 252:
        raise ValueError("insufficient history for purged weekly evaluation")

    def span(start, stop):
        return [calendar[start].date().isoformat(), calendar[stop - 1].date().isoformat()]

    return {
        "protocol": "weekly_latest_matured_v1",
        "as_of": cutoff.date().isoformat(),
        "data_end": calendar[-1].date().isoformat(),
        "latest_matured_signal": calendar[mature_count - 1].date().isoformat(),
        "research_segments": {
            "train": span(0, train_stop),
            "valid": span(valid_start, test_start - horizon),
            "test": span(test_start, mature_count),
        },
        "deployment_refit": span(0, mature_count),
        "unmatured_features_only": span(mature_count, len(calendar)),
        "purge_trading_days": horizon,
        "requires_independent_completeness_check": True,
        "deployment_refit_has_no_inherited_oos_score": True,
    }
