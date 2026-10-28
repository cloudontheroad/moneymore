"""Conservative explanation helpers; evidence does not change model eligibility."""

import pandas as pd


def announcement_gap_reason(symbol, date, evidence):
    date = pd.Timestamp(date).strftime("%Y%m%d")
    for row in evidence.get("symbol_changes", []):
        if symbol == row["old"] and date >= row["effective"]:
            return "SYMBOL_CHANGE_REQUIRES_IDENTITY_MAPPING", row["source"]
    for row in evidence.get("suspensions", []):
        if symbol == row["symbol"] and row["start"] <= date < row["resume"]:
            return "ISSUER_ANNOUNCEMENT_CONFIRMED_SUSPENSION", row["source"]
    return "UNEXPLAINED_PRICE_GAP", None


def full_session_dates(suspensions):
    # An intraday halt is not proof that a whole daily bar may be absent.
    rows = suspensions.loc[
        suspensions.suspend_type.eq("S")
        & (suspensions.suspend_timing.isna() | suspensions.suspend_timing.eq(""))
    ]
    return set(rows.trade_date.astype(str))


def financial_hole_reason(period_rows, as_of, field):
    visible = period_rows.loc[
        pd.to_datetime(period_rows.ann_date, format="%Y%m%d", errors="coerce") + pd.offsets.Day(1)
        <= pd.Timestamp(as_of)
    ]
    if visible.empty:
        return "NO_VISIBLE_PERIOD_PAYLOAD"
    latest = visible.loc[visible.ann_date.eq(visible.ann_date.max())]
    conflict = "conflict__" + field
    if conflict in latest and latest[conflict].eq(True).any():
        return "UNRESOLVED_DUPLICATE_VERSION_CONFLICT"
    if pd.to_numeric(latest[field], errors="coerce").notna().any():
        return "HAS_NUMERIC_SOURCE_REQUIRES_FEATURE_PROVENANCE_REVIEW"
    return "SOURCE_FIELD_NOT_REPORTED"
