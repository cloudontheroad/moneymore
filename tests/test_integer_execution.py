import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.costs import transaction_fee
from moneymore.models import Side
from moneymore.research.integer_execution import (
    paper_intents_from_integer_plan,
    plan_integer_execution,
)


def config():
    return BacktestConfig(
        initial_cash=10000,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=2,
        lot_size=100,
        max_position_weight=1.0,
        max_gross_exposure=1.0,
        max_drawdown=0.2,
    )


@pytest.mark.parametrize("secondary_exception", [False, True])
def test_secondary_failure_preserves_certified_primary_plan(monkeypatch, secondary_exception):
    import cvxpy as cp

    original = cp.Problem.solve
    calls = []

    def force_secondary_failure(problem, *args, **kwargs):
        calls.append(problem)
        if len(calls) == 2:
            problem._status = "infeasible"
            for variable in problem.variables():
                variable.save_value(None)
            if secondary_exception:
                raise cp.error.SolverError("test-secondary-error")
            return float("inf")
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", force_secondary_failure)
    result = plan({"A": 0.5, "B": 0.5}, {"A": 10.0, "B": 10.0}, limits=2)
    assert result["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert result["primary_optimal_fallback"]
    assert result["secondary_optimizer_status"] == (
        "solver_error" if secondary_exception else "infeasible"
    )
    assert result["estimated_cash"] >= 0
    assert result["estimated_gross"] <= 1


@pytest.mark.parametrize("failure_status", ["infeasible", "user_limit"])
def test_primary_infeasible_never_uses_fallback(monkeypatch, failure_status):
    import cvxpy as cp

    def fail(problem, *args, **kwargs):
        problem._status = failure_status
        return float("inf")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    result = plan({"A": 0.5, "B": 0.5}, {"A": 10.0, "B": 10.0}, limits=2)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "INTEGER_FEASIBILITY_" + failure_status.upper()
    assert result["orders"] == []


def plan(
    target, price, quantity=None, available=None, cash=10000, equity=10000, limits=2.0, **kwargs
):
    symbols = sorted(price)
    styles = pd.DataFrame({"style": [-1.0, 1.0]}, index=symbols)
    return plan_integer_execution(
        pd.Series(target),
        pd.Series(price),
        pd.Series(quantity or {}, dtype=float),
        pd.Series(available if available is not None else quantity or {}, dtype=float),
        cash,
        equity,
        config(),
        styles,
        pd.Series({"style": limits}),
        turnover=2.0,
        **kwargs,
    )


def test_exact_minimum_commission_and_budget_not_fake_fee_burn():
    result = plan({"A": 0.5, "B": 0.5}, {"A": 10.0, "B": 10.0}, limits=0.2)
    assert result["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert result["estimated_cash"] >= 0
    assert all(o["quantity"] % 100 == 0 for o in result["orders"])
    expected = sum(
        transaction_fee(o["quantity"] * o["estimated_execution_price"], Side(o["side"]), config())
        for o in result["orders"]
    )
    assert result["estimated_fees"] == pytest.approx(expected)
    assert abs(result["active_style_exposures"]["style"]) <= 0.20001


def test_sales_can_fund_buys_but_orders_are_sells_first():
    result = plan({"B": 1.0}, {"A": 10.0, "B": 10.0}, {"A": 500}, cash=0.0, equity=5000.0)
    assert result["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert [o["side"] for o in result["orders"]] == ["SELL", "BUY"]
    assert result["estimated_cash"] >= 0


def test_unavailable_stock_is_not_sold():
    result = plan(
        {"B": 1.0}, {"A": 10.0, "B": 10.0}, {"A": 500}, available={"A": 0}, cash=0.0, equity=5000.0
    )
    assert result["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert not result["orders"]
    assert result["target_quantities"]["A"] == 500


def test_unaffordable_lot_cannot_be_faked_to_hit_neutrality():
    result = plan({"A": 0.5, "B": 0.5}, {"A": 1000.0, "B": 10.0}, limits=0.0)
    assert result["status"] == "BLOCKED"
    assert not result["orders"]


def test_odd_lot_can_be_fully_liquidated_not_partially_sold():
    result = plan({"B": 1.0}, {"A": 10.0, "B": 10.0}, {"A": 150}, cash=0.0, equity=1500.0)
    assert result["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert next(o for o in result["orders"] if o["symbol"] == "A")["quantity"] == 150


def test_quote_gate_prohibits_blocked_buy():
    result = plan({"B": 1.0}, {"A": 10.0, "B": 10.0}, can_buy={"A": False, "B": False})
    assert result["status"] == "BLOCKED"


def test_star_minimum_200_shares_is_not_a_100_share_board():
    blocked = plan({"688001.SH": 1.0}, {"A": 10.0, "688001.SH": 10.0}, cash=1500.0, equity=1500.0)
    assert blocked["status"] == "BLOCKED"
    passed = plan({"688001.SH": 1.0}, {"A": 10.0, "688001.SH": 10.0}, cash=3000.0, equity=3000.0)
    assert passed["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert next(o for o in passed["orders"] if o["side"] == "BUY")["quantity"] >= 200


def test_paper_broker_executes_exact_plan_with_sell_funding_and_t_plus_one(tmp_path):
    from moneymore.execution.paper import ExecutionBar, PaperBroker
    from moneymore.execution.risk import OrderIntent, RiskResult

    broker = PaperBroker(tmp_path / "isolated.sqlite3")
    broker.initialize_account(10000.0, "test")
    broker.submit(
        RiskResult(True, OrderIntent("seed", "seed", "A", Side.BUY, 900, "20260105", "TEST"), None),
        "test",
    )
    broker.execute_pending(ExecutionBar("A", "20260106", 10.0, 10.0), config(), "test")
    snapshot = broker.account_snapshot({"A": 10.0, "B": 10.0}, "test")
    result = plan(
        {"B": 1.0},
        {"A": 10.0, "B": 10.0},
        {"A": 900},
        cash=snapshot["cash"],
        equity=snapshot["equity"],
    )
    assert result["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    for intent in paper_intents_from_integer_plan(result, "integer-test", "20260106"):
        broker.submit(intent, "test")
    for symbol in [o["symbol"] for o in result["orders"]]:
        broker.execute_pending(ExecutionBar(symbol, "20260107", 10.0, 10.0), config(), "test")
    after = broker.account_snapshot({"A": 10.0, "B": 10.0}, "test")
    assert after["cash"] == pytest.approx(result["estimated_cash"], abs=0.01)
    actual = {p["symbol"]: p["quantity"] for p in after["positions"]}
    assert all(actual.get(s, 0) == q for s, q in result["target_quantities"].items())
    assert broker.reconcile("test").matched


def test_zero_star_position_cannot_create_phantom_minimum_commission():
    result = plan(
        {"A": 0.99999, "688001.SH": 0.00001},
        {"A": 10.0, "688001.SH": 1000.0},
        quantity={"A": 500},
        cash=5000.0,
        equity=10000.0,
    )
    assert result["status"] == "PASS_REFERENCE_QUOTES_ONLY"
    assert not any(o["symbol"] == "688001.SH" for o in result["orders"])
    assert result["estimated_fees"] == pytest.approx(
        sum(o["estimated_fee"] for o in result["orders"])
    )
