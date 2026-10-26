import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

spec = importlib.util.spec_from_file_location("minute_acceptance", Path(__file__).parents[1] / "scripts/alphafeed_minute_acceptance.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(minute, **changes):
    stamp = int(datetime(2026, 10, 26, 10, minute, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()*1000)
    return {"timestamp": stamp, "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "amount": 100000, **changes}


def test_conservative_tail_is_not_compared():
    result = module.compare_minutes([row(0), row(1), row(2)], [row(0), row(1), row(2, close=11)], row(3)["timestamp"])
    assert result["status"] == "SAMPLE_MATCH" and result["compared_rows"] == 2
    assert not result["whole_pool_verified"]


def test_missing_duplicate_invalid_and_empty_are_not_success():
    start = row(5)["timestamp"]
    for left, right in [([], []), ([row(0)], []), ([row(0), row(0)], [row(0)]),
                        ([row(0)], [row(0, close=float("nan"))]), ([row(0)], [row(0, volume=101)])]:
        assert module.compare_minutes(left, right, start)["status"] == "BLOCKED_OR_INCOMPLETE"


def test_amount_precision_difference_is_diagnostic_not_silent_acceptance():
    result = module.compare_minutes([row(0, amount=100000.41)], [row(0, amount=100000)], row(5)["timestamp"])
    assert result["status"] == "BLOCKED_OR_INCOMPLETE"
    assert result["field_mismatch_counts"]["amount"] == 1
    assert result["field_mismatch_counts"]["close"] == 0
    assert 0.40 < result["max_field_deltas"]["amount"] < 0.42


def test_shared_missing_slots_are_reported_even_when_interfaces_match():
    result = module.compare_minutes([row(0)], [row(0)], row(5)["timestamp"])
    assert result["missing_regular_session_slots"]
