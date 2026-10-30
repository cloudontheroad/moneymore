import pandas as pd
import pytest

from moneymore.training_time_contract import weekly_time_contract


def test_latest_matured_refit_and_purged_research():
    dates = pd.bdate_range("2022-01-01", periods=1000)
    result = weekly_time_contract(dates, as_of=dates[-1])
    assert result["deployment_refit"][1] == str(dates[-6].date())
    sections = result["research_segments"]
    for left, right in [("train", "valid"), ("valid", "test")]:
        last = dates.get_loc(pd.Timestamp(sections[left][1]))
        first = dates.get_loc(pd.Timestamp(sections[right][0]))
        assert last + 5 < first
    assert result["deployment_refit_has_no_inherited_oos_score"]


def test_future_dates_do_not_change_plan():
    dates = pd.bdate_range("2022-01-01", periods=1000)
    assert weekly_time_contract(dates, as_of=dates[899]) == weekly_time_contract(
        dates[:900], as_of=dates[899]
    )


def test_reject_insufficient_history():
    with pytest.raises(ValueError):
        weekly_time_contract(pd.bdate_range("2026-01-01", periods=100), as_of="2026-10-30")
