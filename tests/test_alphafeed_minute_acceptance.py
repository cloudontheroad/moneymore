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
