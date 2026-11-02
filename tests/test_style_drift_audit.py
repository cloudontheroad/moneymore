import numpy as np
import pandas as pd
import pytest
from moneymore.research.style_drift_audit import centered_exposure, decompose_drift, repair_state


def test_accounting_identity_separates_weight_drift_and_control_update():
    result = decompose_drift({"beta": .08}, {"beta": .09}, {"beta": .16})
    assert result.loc["beta", "within_day_weight_effect"] == pytest.approx(.01)
    assert result.loc["beta", "control_and_reference_update_effect"] == pytest.approx(.07)
    assert abs(result.loc["beta", "identity_residual"]) < 1e-12


@pytest.mark.parametrize("before,after,next_session,passed,expected", [
    (.15, .09, True, True, "REPAIRED_AFTER_EXECUTED_TRADES"),
    (.09, .08, True, True, "NATURAL_REENTRY_BEFORE_EXECUTION"),
    (.15, .12, True, True, "STILL_OUTSIDE_AFTER_EXECUTION"),
    (None, None, True, False, "PENDING_BLOCKED_EXECUTION"),
    (None, None, False, False, "PENDING_WINDOW_END"),
    (None, .09, True, True, "PENDING_MISSING_EVIDENCE"),
])
def test_repair_states_never_equate_no_trade_with_repaired(before, after, next_session, passed, expected):
    assert repair_state(before, after, .1, next_session=next_session, plan_passed=passed) == expected


def test_missing_held_coordinate_not_filled_with_zero():
    z = pd.DataFrame({"beta": [1.]}, index=["A"])
    with pytest.raises(ValueError):
        centered_exposure({"B": 100}, {"B": 10.}, z)
    with pytest.raises(ValueError):
        centered_exposure({"A": 100}, {"A": np.nan}, z)
