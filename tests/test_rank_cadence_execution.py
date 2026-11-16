"""The fixed raw-rank cadence/buffer experiment preserves execution safeguards."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_rank_preserving_execution import (
    CASE,
    ROOT,
    SIGNAL,
    decision_for,
    planner_case,  # noqa: F401 -- imported pytest fixture
)

POLICIES = {
    "rank_cap": {"rebalance_sessions": 20, "max_optional_exits": None},
    "rank_cap_buffer": {"rebalance_sessions": 20, "max_optional_exits": 5},
    "rank_cap_fast": {"rebalance_sessions": 10, "max_optional_exits": None},
    "rank_cap_fast_buffer": {"rebalance_sessions": 10, "max_optional_exits": 5},
}


@pytest.fixture
def cadence_case(request):
    base, context, covariance, attrs, calls = request.getfixturevalue("planner_case")
    import replay_ashare_rank_cadence as runner

    planner = runner.CadencePlanner.__new__(runner.CadencePlanner)
    planner.__dict__.update(base.__dict__)
    contract = json.loads((ROOT / "configs/ashare_rank_cadence_v1.json").read_bytes())
    planner.contract = {**contract, "topk": 4, "stock_exposure": 0.08}
    return planner, context, covariance, attrs, calls


@pytest.mark.parametrize("allocator", POLICIES)
@pytest.mark.parametrize("index", [10, 20])
def test_ten_and_twenty_session_cadences_dispatch_at_the_registered_index(
    cadence_case, allocator, index
):
    planner, context, _, _, calls = cadence_case
    context.update(kind=CASE + "--" + allocator, index=index)
    original = planner.contract
    before = copy.deepcopy(original)
    quantities, reason = planner(**context)
    scheduled = index % POLICIES[allocator]["rebalance_sessions"] == 0
    assert reason is None
    assert calls == ([SIGNAL] if scheduled else [])
    assert quantities["A"] == 100
    assert (len(quantities) > 1) == scheduled
    decision = decision_for(planner, context)
    assert decision["cadence_policy"] == POLICIES[allocator]
    assert decision["execution_index"] == index
    assert decision["scheduled_rebalance"] is scheduled
    assert decision["status"] == ("OPEN_PLAN_VERIFIED" if scheduled else "NOT_SCHEDULED")
    assert planner.contract is original and planner.contract == before


@pytest.mark.parametrize("allocator", POLICIES)
def test_cadence_cases_pass_raw_scores_and_exact_registered_buffer(
    cadence_case, monkeypatch, allocator
):
    import replay_ashare_rank_preserving as base_runner

    planner, context, _, _, _ = cadence_case
    context.update(kind=CASE + "--" + allocator, index=20)
    captured = []
    original_allocate = base_runner.allocate_ranked

    def capture(scores, industries, benchmark, previous, **kwargs):
        pd.testing.assert_series_equal(scores, planner.scores[(CASE, SIGNAL)])
        captured.append(kwargs)
        return original_allocate(scores, industries, benchmark, previous, **kwargs)

    def forbidden_sector_transform(*args, **kwargs):
        raise AssertionError("cadence_experiment_must_keep_raw_scores")

    monkeypatch.setattr(base_runner, "allocate_ranked", capture)
    monkeypatch.setattr(base_runner, "sector_relative_scores", forbidden_sector_transform)
    _, reason = planner(**context)
    assert reason is None and len(captured) == 1
    assert captured[0]["max_optional_exits"] == POLICIES[allocator]["max_optional_exits"]
    assert captured[0]["industry_active_limit"] == 0.08
    decision = decision_for(planner, context)
    assert decision["allocation"]["max_optional_exits"] == POLICIES[allocator]["max_optional_exits"]
    assert decision["open_postcheck"]["stock_limit"] == 0.05
    for sector, weight in decision["allocation"]["benchmark_industry_weights"].items():
        assert decision["open_postcheck"]["industry_limits"][sector] == pytest.approx(weight + 0.10)


@pytest.mark.parametrize("allocator", ["rank_cap_buffer", "rank_cap_fast_buffer"])
def test_raw_buffer_replaces_only_five_of_fifty_worse_old_names(cadence_case, allocator):
    planner, context, _, _, _ = cadence_case
    symbols = [f"S{i:03}" for i in range(100)]
    scores = pd.Series(np.arange(100, dtype=float), index=symbols)
    attrs = pd.DataFrame({"total_mv": 100.0, "industry": "bank"}, index=symbols)
    covariance = pd.DataFrame(np.eye(100) * 0.01, index=symbols, columns=symbols)
    planner.scores[(CASE, SIGNAL)] = scores
    planner.risk = lambda signal: (covariance, attrs, {"risk_end": signal})
    planner.contract.update(topk=50, stock_exposure=0.995)
    held = dict.fromkeys(symbols[:50], 19900)
    snapshot = {
        "cash": 50000.0,
        "equity": 10000000.0,
        "unsettled_stock_rights": {},
        "receivable_cash": 0.0,
        "positions": [
            {"symbol": symbol, "quantity": quantity, "available_quantity": quantity}
            for symbol, quantity in held.items()
        ],
    }
    context.update(
        kind=CASE + "--" + allocator,
        index=20,
        held=held,
        night_holdings=held,
        snapshot=snapshot,
        night=snapshot,
        members=set(symbols),
        last=dict.fromkeys(symbols, 10.0),
        opened=dict.fromkeys(symbols, 10.0),
        buy=dict.fromkeys(symbols, True),
        sell=dict.fromkeys(symbols, True),
    )
    quantities, reason = planner(**context)
    assert reason is None and len(quantities) == 50
    assert set(held) - set(quantities) == set(symbols[:5])
    assert set(quantities) - set(held) == set(symbols[-5:])
    allocation = decision_for(planner, context)["allocation"]
    assert allocation["optional_exits"] == symbols[:5]
    assert allocation["risk_forced_exits"] == []
    assert allocation["membership_turnover"] == {"entries": 5, "exits": 5}


def test_unknown_case_fails_before_risk_or_mutating_shared_contract(cadence_case):
    planner, context, _, _, calls = cadence_case
    context["kind"] = CASE + "--rank_cap_unregistered"
    original = planner.contract
    before = copy.deepcopy(original)
    with pytest.raises(ValueError):
        planner(**context)
    assert calls == []
    assert planner.contract is original and planner.contract == before


@pytest.mark.parametrize("allocator", POLICIES)
def test_temporary_cadence_contract_is_restored_when_risk_raises(cadence_case, allocator):
    planner, context, _, _, _ = cadence_case
    context.update(kind=CASE + "--" + allocator, index=20)
    original = planner.contract
    before = copy.deepcopy(original)

    def broken_risk(signal):
        assert planner.contract["rebalance_sessions"] == POLICIES[allocator]["rebalance_sessions"]
        raise RuntimeError("SIMULATED_RISK_FAILURE")

    planner.risk = broken_risk
    with pytest.raises(RuntimeError, match="SIMULATED_RISK_FAILURE"):
        planner(**context)
    assert planner.contract is original and planner.contract == before


@pytest.mark.parametrize("allocator", POLICIES)
def test_mandatory_exit_executes_off_cadence_without_risk_or_optional_trades(
    cadence_case, allocator
):
    planner, context, _, _, _ = cadence_case

    def forbidden_risk(signal):
        raise AssertionError("off_cadence_risk_reoptimization")

    planner.risk = forbidden_risk
    context.update(
        kind=CASE + "--" + allocator,
        index=1,
        held={"A": 100, "B": 100},
        night_holdings={"A": 100, "B": 100},
        members=set("BCDE"),
        sell={"A": True},
    )
    context["snapshot"] = {
        **context["snapshot"],
        "cash": 98000.0,
        "positions": [
            {"symbol": symbol, "quantity": 100, "available_quantity": 100} for symbol in "AB"
        ],
    }
    quantities, reason = planner(**context)
    assert reason is None and quantities == {"B": 100}
    decision = decision_for(planner, context)
    assert decision["status"] == "MANDATORY_EXITS_ONLY"
    assert decision["pending_exit_symbols"] == []
    assert decision["cadence_policy"] == POLICIES[allocator]
    assert decision["scheduled_rebalance"] is False
    assert decision["execution_index"] == 1


@pytest.mark.parametrize("allocator", POLICIES)
def test_frozen_infeasibility_protects_holdings_and_still_exits_pool_leaver(
    cadence_case, allocator
):
    planner, context, covariance, attrs, _ = cadence_case
    context.update(
        kind=CASE + "--" + allocator,
        index=20,
        members=set("ABCD"),
        held={"A": 1000, "E": 100},
        night_holdings={"A": 1000, "E": 100},
        sell={"A": False, "E": True},
    )
    planner.risk = lambda signal: (covariance, attrs.drop(index="E"), {})
    planner.industry_history["E"] = pd.DataFrame({"date": ["20260129"], "industry": ["industry"]})
    snapshot = {
        **context["snapshot"],
        "cash": 89000.0,
        "positions": [
            {"symbol": "A", "quantity": 1000, "available_quantity": 0},
            {"symbol": "E", "quantity": 100, "available_quantity": 100},
        ],
    }
    context["snapshot"] = context["night"] = snapshot
    original = planner.contract
    quantities, reason = planner(**context)
    assert quantities == {"A": 1000}
    assert reason == "RANK_ALLOCATION_FROZEN_EXPOSURE_EXCEEDS_TARGET"
    decision = decision_for(planner, context)
    assert decision["retry_next_session"] is True
    assert decision["exit_symbols"] == ["E"]
    assert decision["cadence_policy"] == POLICIES[allocator]
    assert decision["scheduled_rebalance"] is True
    assert planner.contract is original


def test_registered_source_contract_has_exact_finite_design_and_original_risk_limits(cadence_case):
    import replay_ashare_rank_cadence as runner

    contract = json.loads((ROOT / "configs/ashare_rank_cadence_v1.json").read_bytes())
    assert contract["case_specs"] == POLICIES
    assert set(contract["allocators"]) == set(POLICIES)
    assert contract["models"] == [CASE]
    assert contract["n_drop"] == 5 and contract["topk"] == 50
    assert contract["industry_active_upper_planning"] == 0.08
    assert contract["industry_active_upper_execution"] == 0.10
    assert contract["max_stock_weight"] == 0.05
    assert contract["stock_exposure"] == 0.995
    assert contract["slippage_bps"] == [10, 20]
    sources = {path.resolve() for path in runner.source_files()}
    assert {
        ROOT / "scripts/replay_ashare_rank_cadence.py",
        ROOT / "scripts/replay_ashare_rank_preserving.py",
        ROOT / "configs/ashare_rank_cadence_v1.json",
        ROOT / "src/moneymore/research/rank_preserving.py",
    } <= sources


@pytest.mark.parametrize("mutation", ["period", "buffer", "missing_case", "unknown_case"])
def test_constructor_rejects_unregistered_cadence_or_buffer_contract(
    cadence_case, monkeypatch, tmp_path, mutation
):
    import replay_ashare_rank_cadence as runner

    contract = json.loads((ROOT / "configs/ashare_rank_cadence_v1.json").read_bytes())
    if mutation == "period":
        contract["case_specs"]["rank_cap_fast"]["rebalance_sessions"] = 15
    elif mutation == "buffer":
        contract["case_specs"]["rank_cap_buffer"]["max_optional_exits"] = 10
    elif mutation == "missing_case":
        del contract["case_specs"]["rank_cap_fast_buffer"]
    else:
        contract["case_specs"]["rank_cap_unknown"] = dict(POLICIES["rank_cap"])

    # Prevent unrelated historical-data loading; the real wrapper's initializer
    # must still reject policy changes before an experiment can start.
    monkeypatch.setattr(runner.CadencePlanner.__mro__[1], "__init__", lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        runner.CadencePlanner(tmp_path, contract, ())
