"""Real integer feasibility gates on a tiny, explicit diagnostic account."""
import numpy as np
import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.research.integer_execution import plan_integer_execution
from moneymore.research.style_drift_audit import repair_state


def repair_plan(*, sell_allowed=True, buy_allowed=True, available=1000, price_b=10.):
    config = BacktestConfig(initial_cash=10000., commission_rate=.00015, minimum_commission=5.,
        stamp_duty_rate=.0005, transfer_fee_rate=.00001, slippage_bps=10., lot_size=100,
        max_position_weight=1., max_gross_exposure=1., max_drawdown=.2)
    # Diagnostic test uses 2-stock/2.0 turnover budget, not production policy.
    return plan_integer_execution({"A": .5, "B": .5}, {"A":10., "B":price_b}, {"A":1000}, {"A":available},
        0., 10000., config, pd.DataFrame({"beta": [1., -1.]}, index=["A", "B"]), {"beta": .1},
        benchmark_weights={"A":.5, "B":.5}, style_controls_prestandardized=True,
        turnover=2., can_sell={"A":sell_allowed, "B":True}, can_buy={"A":True, "B":buy_allowed},
        time_limit=5., solver_threads=2)


def test_unblocked_integer_repair_preserves_cash_and_style():
    plan = repair_plan()
    assert plan["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert plan["estimated_cash"] >= 0
    assert abs(plan["active_style_exposures"]["beta"]) <= .10001
    assert repair_state(1., plan["active_style_exposures"]["beta"], .1,
        next_session=True, plan_passed=True) == "REPAIRED_AFTER_EXECUTED_TRADES"


@pytest.mark.parametrize("kwargs", [{"sell_allowed":False}, {"buy_allowed":False}, {"available":0}])
def test_halt_or_unsellable_shares_do_not_create_fictional_repair(kwargs):
    plan = repair_plan(**kwargs)
    assert plan["status"] == "BLOCKED"
    assert plan["reason"] == "INTEGER_FEASIBILITY_INFEASIBLE"
    assert plan["orders"] == []
    assert repair_state(1., None, .1, next_session=True, plan_passed=False) == "PENDING_BLOCKED_EXECUTION"


def test_missing_mark_is_not_backfilled_or_called_repaired():
    plan = repair_plan(price_b=np.nan)
    assert plan["status"] == "BLOCKED"
    assert plan["reason"] == "INVALID_QUOTES_QUANTITIES_OR_STYLE_BOUNDS"
    assert plan["orders"] == []
