"""Practical vendor-data controls, explicitly NOT original financial PIT proof."""

import pandas as pd

from .baseline_v2_financials import common_period_features


def audit_vendor_financials(frame, as_of):
    features = common_period_features(frame, as_of)
    cutoff = pd.Timestamp(as_of)
    invalid_visible_dates = 0
    for field in ("roe", "q_sales_yoy"):
        dates = pd.to_datetime(
            features.get(field + "_ann_date", pd.Series(index=features.index, dtype="object")),
            format="%Y%m%d",
            errors="coerce",
        )
        populated = pd.to_numeric(features[field], errors="coerce").notna()
        invalid_visible_dates += int((populated & (dates.isna() | dates.ge(cutoff))).sum())
    usable = int(features[["roe", "q_sales_yoy"]].notna().any(axis=1).sum())
    failures = []
    if not usable:
        failures.append("NO_VISIBLE_VENDOR_FINANCIAL_FEATURES")
    if invalid_visible_dates:
        failures.append("ANNOUNCEMENT_CUTOFF_VIOLATION")
    return {
        "status": "PASS" if not failures else "BLOCKED_VENDOR_DATA_CONTROLS",
        "acceptance_scope": "VENDOR_DATA_CONTROLS_NOT_ORIGINAL_FINANCIAL_PIT",
        "as_of": as_of,
        "visible_symbols": len(features),
        "symbols_with_at_least_one_feature": usable,
        "invalid_visible_announcement_dates": invalid_visible_dates,
        "gate_failures": failures,
        "historical_point_in_time_attested": False,
        "exhaustive_original_document_review_required": False,
        "known_limitations": [
            "Historical revisions may not retain their original availability timestamp.",
            "Publication lag does not eliminate unobserved restatements.",
        ],
        "real_orders_enabled": False,
        "production_changed": False,
    }
