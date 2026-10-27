import importlib.util
from pathlib import Path

import pandas as pd


spec = importlib.util.spec_from_file_location("prepare_style_replay", Path(__file__).resolve().parents[1] / "scripts/prepare_style_replay.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_limits_fail_closed_and_market_gaps_are_separate():
    daily = pd.DataFrame([dict(trade_date="20261012", ts_code="A", open=10., close=11.)])
    limits = pd.DataFrame(columns=["trade_date", "ts_code", "up_limit", "down_limit"])
    result = module.execution_coverage(daily, limits, ["20261012", "20261013"], ["A"], "20261012", "20261013")
    assert result["status"] == "BLOCKED_MISSING_OFFICIAL_LIMITS"
    assert result["missing_limit_rows"] == 1
    assert result["missing_market_rows"] == 1


def test_official_limits_require_positive_valid_bounds():
    daily = pd.DataFrame([dict(trade_date="20261012", ts_code="A", open=10., close=11.)])
    limits = pd.DataFrame([dict(trade_date="20261012", ts_code="A", up_limit=11., down_limit=9.)])
    result = module.execution_coverage(daily, limits, ["20261012"], ["A"], "20261012", "20261012")
    assert result["missing_limit_rows"] == 0
    limits.loc[0, "down_limit"] = 12.
    assert module.execution_coverage(daily, limits, ["20261012"], ["A"], "20261012", "20261012")["missing_limit_rows"] == 1
