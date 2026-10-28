"""Common disclosed fiscal period; don't compare Q1 ROE with annual ROE."""

import pandas as pd

from ..data.fundamental_research import financial_features_as_of


def required_period(as_of):
    date = pd.Timestamp(as_of)
    # Standard SSE/SZSE deadlines, plus the existing announcement T+1 rule.
    # Exceptional extensions are NOT presumed; missing evidence remains missing.
    if date.month <= 4:
        return f"{date.year - 1}0930"
    if date.month <= 8:
        return f"{date.year}0331"
    if date.month <= 10:
        return f"{date.year}0630"
    return f"{date.year}0930"


def common_period_features(frame, as_of):
    period = required_period(as_of)
    selected = frame.loc[frame.end_date.astype(str).eq(period)].copy()
    result = financial_features_as_of(selected, as_of)
    for field in ("roe", "q_sales_yoy"):
        if field + "_end_date" not in result:
            result[field + "_end_date"] = pd.Series(dtype=str)
    result["common_required_period"] = period
    return result
