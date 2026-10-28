import pandas as pd

from moneymore.research.baseline_v2_gap_evidence import financial_hole_reason, full_session_dates


def test_announcement_boundaries_do_not_make_resumption_day_a_halt():
    from moneymore.research.baseline_v2_gap_evidence import announcement_gap_reason

    evidence = {
        "suspensions": [
            {"symbol": "A", "start": "20240102", "resume": "20240104", "source": "issuer"}
        ],
        "symbol_changes": [{"old": "A", "new": "B", "effective": "20240105", "source": "exchange"}],
    }
    assert (
        announcement_gap_reason("A", "20240103", evidence)[0]
        == "ISSUER_ANNOUNCEMENT_CONFIRMED_SUSPENSION"
    )
    assert announcement_gap_reason("A", "20240104", evidence)[0] == "UNEXPLAINED_PRICE_GAP"
    assert (
        announcement_gap_reason("A", "20240105", evidence)[0]
        == "SYMBOL_CHANGE_REQUIRES_IDENTITY_MAPPING"
    )
    assert announcement_gap_reason("B", "20240105", evidence)[0] == "UNEXPLAINED_PRICE_GAP"


def test_intraday_halt_cannot_explain_missing_daily_bar():
    f = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104", "20240105"],
            "suspend_type": ["S", "S", "S", "R"],
            "suspend_timing": [None, "", "09:30-10:30", ""],
        }
    )
    assert full_session_dates(f) == {"20240102", "20240103"}


def test_financial_hole_distinguishes_time_conflict_and_null():
    f = pd.DataFrame({"ann_date": ["20240430"], "roe": [None], "conflict__roe": [True]})
    assert financial_hole_reason(f, "20240430", "roe") == "NO_VISIBLE_PERIOD_PAYLOAD"
    assert financial_hole_reason(f, "20240501", "roe") == "UNRESOLVED_DUPLICATE_VERSION_CONFLICT"
    f["conflict__roe"] = False
    assert financial_hole_reason(f, "20240501", "roe") == "SOURCE_FIELD_NOT_REPORTED"
    f["roe"] = 1.0
    assert (
        financial_hole_reason(f, "20240501", "roe")
        == "HAS_NUMERIC_SOURCE_REQUIRES_FEATURE_PROVENANCE_REVIEW"
    )
