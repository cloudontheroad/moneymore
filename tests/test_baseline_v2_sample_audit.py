from moneymore.research.baseline_v2_sample_audit import exclusion_reason


def test_listing_age_is_separate_from_unverified_suspension():
    import pandas as pd

    from moneymore.research.baseline_v2_sample_audit import classify_price_history

    calendar = pd.bdate_range("2020-01-01", periods=260).strftime("%Y%m%d").tolist()
    f = pd.DataFrame(
        {
            "date": [calendar[-1]] * 3,
            "symbol": ["A", "B", "C"],
            "reason": ["INCOMPLETE_PRICE_WINDOW_NOT_PROVEN_SUSPENSION_OR_IPO"] * 3,
        }
    )
    instruments = pd.DataFrame(
        {"ts_code": ["A", "B", "C"], "list_date": ["20200301", "20100101", float("nan")]}
    )
    actual = classify_price_history(f, instruments, calendar)
    assert actual.price_gap_class.tolist() == [
        "LISTED_TOO_RECENT_FOR_253_SESSION_RULE",
        "OLDER_STOCK_GAP_REQUIRES_SUSPENSION_OR_BACKFILL_PROOF",
        "LISTING_METADATA_UNKNOWN",
    ]


def test_missing_history_never_claims_proven_ipo_or_suspension():
    assert (
        exclusion_reason(observations=0, expected=253, financial_ok=True, industry_ok=True)
        == "NO_VALID_PRICE_HISTORY"
    )
    assert (
        exclusion_reason(observations=252, expected=253, financial_ok=True, industry_ok=True)
        == "INCOMPLETE_PRICE_WINDOW_NOT_PROVEN_SUSPENSION_OR_IPO"
    )


def test_exclusive_reasons_reconcile_without_double_counting():
    assert (
        exclusion_reason(observations=253, expected=253, financial_ok=False, industry_ok=False)
        == "REQUIRED_FINANCIAL_MISSING"
    )
    assert (
        exclusion_reason(observations=253, expected=253, financial_ok=True, industry_ok=False)
        == "HISTORICAL_INDUSTRY_UNKNOWN"
    )
    assert (
        exclusion_reason(observations=253, expected=253, financial_ok=True, industry_ok=True)
        == "ELIGIBLE"
    )
