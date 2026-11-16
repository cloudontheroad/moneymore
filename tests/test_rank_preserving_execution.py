"""Isolated planner checks: no actual accounts, runners, or market data are used."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moneymore.config import BacktestConfig

ROOT = Path(__file__).resolve().parents[1]
ALLOCATORS = ["topk", "sector_topk", "rank_equal", "rank_cap", "sector_rank_cap", "sector_rank_buffer"]
CONSTRAINED_ALLOCATORS = ["rank_cap", "sector_rank_cap", "sector_rank_buffer"]
RANKED_ALLOCATORS = ["rank_equal", *CONSTRAINED_ALLOCATORS]
SIGNAL = "20260130"
EXECUTION = "20260202"
CASE = "doubleensemble_full"


@pytest.fixture
def planner_case(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import replay_ashare_mainline as mainline
    import replay_ashare_rank_preserving as runner

    def forbidden_native(*args, **kwargs):
        raise AssertionError("rank_planner_must_not_invoke_monetary_mu_solver")

    monkeypatch.setattr(mainline, "run_native", forbidden_native)
    planner = runner.RankPlanner.__new__(runner.RankPlanner)
    planner.output = tmp_path
    planner.solver_cache = None
    planner.contract = json.loads((ROOT / "configs/ashare_rank_preserving_v1.json").read_bytes())
    # Small cross-section, retaining the production weight cap and physical checks.
    planner.contract.update(topk=4, stock_exposure=.08)
    planner.scores = {
        (CASE, SIGNAL): pd.Series({"B": 4., "C": 3., "D": 2., "E": 1.}),
        (CASE, EXECUTION): pd.Series({"B": -100., "C": -200., "D": 900., "E": 800.}),
    }
    planner.calibration = {}  # Any attempt to calibrate monetary mu fails.
    planner.industry_history = {
        "A": pd.DataFrame({"date": ["20260129", "20260202"],
                           "industry": ["bank", "future_industry"]})
    }
    attrs = pd.DataFrame(
        {"total_mv": [100.] * 4, "industry": ["bank", "industry", "bank", "industry"]},
        index=list("BCDE"),
    )
    covariance = pd.DataFrame(np.eye(5) * .01, index=list("ABCDE"), columns=list("ABCDE"))
    calls = []

    def risk(signal):
        assert signal == SIGNAL
        calls.append(signal)
        return covariance, attrs, {"signal_date": signal, "risk_end": signal}

    planner.risk = risk
    snapshot = {
        "cash": 99000.,
        "equity": 100000.,
        "unsettled_stock_rights": {},
        "receivable_cash": 0.,
        "positions": [{"symbol": "A", "quantity": 100, "available_quantity": 100}],
    }
    context = {
        "index": 20,
        "day": EXECUTION,
        "signal": SIGNAL,
        "held": {"A": 100},
        "night_holdings": {"A": 100},
        "members": set("ABCDE"),
        "night": snapshot,
        "snapshot": snapshot,
        "last": dict.fromkeys("ABCDE", 10.),
        "opened": dict.fromkeys("ABCDE", 10.),
        "buy": dict.fromkeys("BCDE", True),
        "sell": {},
        "actions": pd.DataFrame(columns=["symbol", "ex_date"]),
        "config": BacktestConfig.from_yaml(ROOT / "configs/default.yaml").model_copy(
            update={"max_position_weight": .05}
        ),
    }
    return planner, context, covariance, attrs, calls


def decision_for(planner, context):
    return json.loads((planner.output / "decisions" / context["kind"]
                       / f"{context['day']}.json").read_bytes())


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_rank_variants_use_only_previous_close_scores_and_industries(planner_case, allocator):
    planner, context, _, _, calls = planner_case
    context["kind"] = CASE + "--" + allocator
    first, first_reason = planner(**context)
    first_decision = decision_for(planner, context)
    assert first_reason is None
    assert first_decision["signal_date"] == SIGNAL < first_decision["execution_date"]
    assert first_decision["risk_audit"]["risk_end"] == SIGNAL
    assert first_decision["next_open_used_for_selection"] is False
    # Perturb every available future score/classification; the prior-close plan
    # and its selected targets must be identical.
    planner.scores[(CASE, EXECUTION)] *= -1e6
    planner.industry_history["A"].loc[1, "industry"] = "different_future_industry"
    second, second_reason = planner(**context)
    second_decision = decision_for(planner, context)
    assert second_reason is None and second == first
    assert second_decision["target_quantities"] == first_decision["target_quantities"]
    assert calls == [SIGNAL, SIGNAL]
    if allocator in RANKED_ALLOCATORS:
        assert "future_industry" not in first_decision["allocation"]["industry_count_caps"]
        assert "different_future_industry" not in second_decision["allocation"]["industry_count_caps"]


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_rank_variants_freeze_missing_held_score_without_freezing_other_names(planner_case, allocator):
    planner, context, _, _, _ = planner_case
    context["kind"] = CASE + "--" + allocator
    quantities, reason = planner(**context)
    assert reason is None
    assert quantities["A"] == 100
    assert quantities["B"] > 0
    decision = decision_for(planner, context)
    assert decision["status"] == "OPEN_PLAN_VERIFIED"
    assert decision["frozen_symbols"] == ["A"]
    assert decision["target_quantities"]["A"] == 100
    if allocator in RANKED_ALLOCATORS:
        assert decision["allocation"]["frozen_weight"] == .01
        assert decision["allocation"]["effective_slots"] == 3
        assert decision["allocation"]["selected_count"] == 3
        assert decision["allocation"]["total_industry_weights"]["bank"] >= .01
        assert decision["open_postcheck"]["passed"] is True


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_missing_held_risk_history_protects_positions_but_checks_mandatory_exits(planner_case, allocator):
    planner, context, covariance, attrs, _ = planner_case
    context["kind"] = CASE + "--" + allocator
    planner.risk = lambda signal: (covariance.loc[list("BCDE"), list("BCDE")], attrs, {})
    context["members"] = set("ABCD")  # E is a genuine pool exit, A is not.
    context["held"] = context["night_holdings"] = {"A": 100, "E": 100}
    snapshot = {**context["snapshot"], "cash": 98000., "positions": [
        {"symbol": symbol, "quantity": 100, "available_quantity": 100}
        for symbol in ["A", "E"]
    ]}
    context["snapshot"] = context["night"] = snapshot
    context["sell"] = {"A": True, "E": True}
    quantities, reason = planner(**context)
    assert reason == "HELD_RISK_HISTORY_MISSING"
    assert quantities == {"A": 100}  # No optional sell of A or new purchases.
    decision = decision_for(planner, context)
    assert decision["status"] == "BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED"
    assert decision["missing_symbols"] == ["A"]
    assert decision["retry_next_session"] is True


@pytest.mark.parametrize("allocator", ALLOCATORS)
def test_off_schedule_pending_exit_never_calls_risk_or_rebalances_others(planner_case, allocator):
    planner, context, _, _, _ = planner_case
    context["kind"] = CASE + "--" + allocator

    def forbidden_risk(signal):
        raise AssertionError("off_schedule_optional_reoptimization")

    planner.risk = forbidden_risk
    context.update(index=21, day="20260203", signal="20260202",
                   held={"A": 100, "B": 100}, night_holdings={"A": 100, "B": 100},
                   members=set("BCDE"))
    context["snapshot"] = {**context["snapshot"], "cash": 98000., "positions": [
        {"symbol": symbol, "quantity": 100, "available_quantity": 100}
        for symbol in ["A", "B"]
    ]}
    quantities, reason = planner(**context)
    assert reason is None and quantities == {"A": 100, "B": 100}
    decision = decision_for(planner, context)
    assert decision["status"] == "MANDATORY_EXITS_ONLY"
    assert decision["pending_exit_symbols"] == ["A"]
    assert decision["retry_next_session"] is False
    context.update(index=22, day="20260204", signal="20260203", sell={"A": True})
    quantities, reason = planner(**context)
    assert reason is None and quantities == {"B": 100}
    decision = decision_for(planner, context)
    assert decision["status"] == "MANDATORY_EXITS_ONLY"
    assert decision["pending_exit_symbols"] == []
    assert decision["retry_next_session"] is False


@pytest.mark.parametrize("allocator", CONSTRAINED_ALLOCATORS)
def test_open_industry_postcheck_blocks_optional_orders_after_clipped_sale(planner_case, allocator):
    planner, context, _, _, _ = planner_case
    context["kind"] = CASE + "--" + allocator
    # The desired reduction of B is blocked at open.  Other buys would be
    # executable, but would preserve a gross sector excess.  Set a loose stock
    # cap here so the industry check, not the single-name check, must stop them.
    planner.contract["max_stock_weight"] = .9
    context["config"] = context["config"].model_copy(update={"max_position_weight": .9})
    context["held"] = context["night_holdings"] = {"B": 6000}
    snapshot = {**context["snapshot"], "cash": 40000., "positions": [
        {"symbol": "B", "quantity": 6000, "available_quantity": 6000}
    ]}
    context["snapshot"] = context["night"] = snapshot
    context["sell"] = {"B": False}
    quantities, reason = planner(**context)
    assert reason == "INTEGER_OPEN_RISK_POSTCHECK"
    assert quantities == {"B": 6000}
    decision = decision_for(planner, context)
    assert decision["status"] == "BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED"
    assert decision["retry_next_session"] is True
    assert decision["allocation"]["status"] == "OK"
    postcheck = decision["open_postcheck"]
    assert postcheck["passed"] is False
    assert postcheck["maximum_stock_weight"] < postcheck["stock_limit"]
    assert postcheck["maximum_industry_violation"] > 0
    assert postcheck["industry_weights"]["bank"] > postcheck["industry_limits"]["bank"]


def test_rank_equal_uses_raw_scores_and_loose_industry_limit_but_same_stock_limit(
    planner_case, monkeypatch
):
    import replay_ashare_rank_preserving as runner

    planner, context, _, _, _ = planner_case
    context["kind"] = CASE + "--rank_equal"

    def forbidden_sector_transform(*args, **kwargs):
        raise AssertionError("rank_equal_must_use_raw_ranks")

    monkeypatch.setattr(runner, "sector_relative_scores", forbidden_sector_transform)
    quantities, reason = planner(**context)
    assert reason is None and quantities["A"] == 100
    decision = decision_for(planner, context)
    allocation = decision["allocation"]
    assert allocation["industry_active_limit"] == 1.0
    for industry, benchmark in allocation["benchmark_industry_weights"].items():
        assert decision["open_postcheck"]["industry_limits"][industry] == pytest.approx(
            benchmark + 1.0
        )
    assert decision["open_postcheck"]["stock_limit"] == .05


@pytest.mark.parametrize("allocator,error", [
    *[(allocator, "FROZEN_EXPOSURE_EXCEEDS_TARGET") for allocator in RANKED_ALLOCATORS],
    *[(allocator, "FROZEN_INDUSTRY_CAP_VIOLATION") for allocator in CONSTRAINED_ALLOCATORS],
])
def test_infeasible_frozen_holdings_protect_optional_trades_but_execute_mandatory_exit(
    planner_case, allocator, error
):
    planner, context, covariance, attrs, _ = planner_case
    context["kind"] = CASE + "--" + allocator
    # A remains held without a current score; E has left the current pool and is
    # not a selectable candidate, but still has complete historical risk data.
    planner.risk = lambda signal: (covariance, attrs.drop(index="E"), {})
    planner.industry_history["E"] = pd.DataFrame(
        {"date": ["20260129"], "industry": ["industry"]}
    )
    if error == "FROZEN_EXPOSURE_EXCEEDS_TARGET":
        # A is frozen because its current score is missing, and its 10% value
        # cannot fit the fixture's 8% target stock budget.
        frozen_quantity = 1000
    else:
        # Keep frozen exposure below the 40% budget, but above bank's 26.67%
        # benchmark plus the unchanged 8% planning allowance.
        planner.contract["stock_exposure"] = .4
        frozen_quantity = 3500
    context["members"] = set("ABCD")
    context["held"] = context["night_holdings"] = {"A": frozen_quantity, "E": 100}
    snapshot = {**context["snapshot"], "cash": 100000. - (frozen_quantity + 100) * 10.,
                "positions": [
                    {"symbol": "A", "quantity": frozen_quantity, "available_quantity": 0},
                    {"symbol": "E", "quantity": 100, "available_quantity": 100},
                ]}
    context["snapshot"] = context["night"] = snapshot
    context["sell"] = {"A": False, "E": True}
    quantities, reason = planner(**context)
    assert reason == "RANK_ALLOCATION_" + error
    assert quantities == {"A": frozen_quantity}
    decision = decision_for(planner, context)
    assert decision["status"] == "BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED"
    assert decision["exit_symbols"] == ["E"]
    assert decision["retry_next_session"] is True


@pytest.mark.parametrize("allocator", RANKED_ALLOCATORS)
def test_frozen_error_protection_does_not_swallow_unexpected_allocator_errors(
    planner_case, monkeypatch, allocator
):
    import replay_ashare_rank_preserving as runner

    planner, context, _, _, _ = planner_case
    context["kind"] = CASE + "--" + allocator

    def unexpected_error(*args, **kwargs):
        raise ValueError("UNEXPECTED_ALLOCATION_FAILURE")

    monkeypatch.setattr(runner, "allocate_ranked", unexpected_error)
    with pytest.raises(ValueError, match="UNEXPECTED_ALLOCATION_FAILURE"):
        planner(**context)
