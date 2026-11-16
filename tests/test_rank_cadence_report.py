import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from moneymore.research.rank_preserving import allocate_ranked


def module():
    spec = importlib.util.spec_from_file_location(
        "rank_cadence_report",
        Path(__file__).resolve().parents[1] / "scripts/report_ashare_rank_cadence.py",
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def contract():
    return {
        "models": ["doubleensemble_full"],
        "allocators": list(module().CASE_SPECS),
        "case_specs": module().CASE_SPECS,
        "initial_cash": 10_000_000,
        "slippage_bps": [10, 20],
        "rebalance_sessions": 20,
        "n_drop": 5,
        "topk": 50,
        "stock_exposure": 0.995,
        "max_stock_weight": 0.05,
        "industry_active_upper_planning": 0.08,
        "industry_active_upper_execution": 0.10,
        "real_orders_enabled": False,
        "production_enabled": False,
    }


def decision(allocator="rank_cap", index=0):
    policy = module().CASE_SPECS[allocator]
    names = [f"A{i:02d}" for i in range(50)]
    _, audit = allocate_ranked(
        pd.Series(range(50), index=names, dtype=float),
        pd.Series("bank", index=names),
        {"bank": 0.995},
        [],
        max_optional_exits=policy["max_optional_exits"],
    )
    return {
        "status": "OPEN_PLAN_VERIFIED",
        "signal_date": "20251231",
        "execution_date": "20260105",
        "cadence_policy": policy,
        "execution_index": index,
        "scheduled_rebalance": index % policy["rebalance_sessions"] == 0,
        "risk_audit": {"risk_end": "20251231"},
        "allocation": audit,
        "target_quantities": dict.fromkeys(names, 100),
        "executable_quantities": dict.fromkeys(names, 100),
        "open_postcheck": {
            "passed": True,
            "actual_stock_exposure": 0.99,
            "maximum_stock_weight": 0.02,
            "maximum_industry_violation": -0.105,
            "stock_limit": 0.05,
            "industry_weights": {"bank": 0.99},
            "industry_limits": {"bank": 1.095},
        },
        "next_open_used_for_selection": False,
    }


def no_fills():
    return pd.DataFrame(columns=["symbol", "side"])


def test_contract_is_exact_frozen_two_by_two():
    module().validate_contract(contract())
    changed = contract()
    changed["case_specs"]["rank_cap_fast"]["rebalance_sessions"] = 5
    with pytest.raises(ValueError, match="FROZEN_CADENCE_FACTORIAL_CONTRACT_MISMATCH"):
        module().validate_contract(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("allocators", ["rank_cap"]),
        ("industry_active_upper_planning", 0.09),
        ("stock_exposure", 1.0),
        ("production_enabled", True),
    ],
)
def test_contract_rejects_matrix_expansion_or_risk_relaxation(field, value):
    with pytest.raises(ValueError, match="FROZEN_CADENCE_FACTORIAL_CONTRACT_MISMATCH"):
        module().validate_contract({**contract(), field: value})


@pytest.mark.parametrize("allocator", list(module().CASE_SPECS))
def test_all_four_raw_cases_receive_full_risk_audit(allocator):
    result = module().audit_decision(decision(allocator), "20260105", no_fills(), allocator, 0)
    assert result["open_postcheck_verified"] and result["allocation_caps_verified"]
    invalid = decision(allocator)
    invalid["open_postcheck"]["maximum_stock_weight"] = 0.06
    with pytest.raises(ValueError, match="INDEPENDENT_OPEN_INTEGER_RISK_POSTCHECK_FAILED"):
        module().audit_decision(invalid, "20260105", no_fills(), allocator, 0)


def test_unknown_name_cannot_skip_risk_audit():
    with pytest.raises(ValueError, match="UNKNOWN_CADENCE_ALLOCATOR"):
        module().audit_decision(decision(), "20260105", no_fills(), "rank_cap_fastest", 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_index", 10),
        ("execution_index", True),
        ("scheduled_rebalance", False),
        ("scheduled_rebalance", 1),
        ("cadence_policy", {"rebalance_sessions": 5, "max_optional_exits": None}),
    ],
)
def test_cadence_metadata_must_match_calendar_and_case(field, value):
    with pytest.raises(ValueError, match="CADENCE_DECISION_METADATA_MISMATCH"):
        module().audit_decision({**decision(), field: value}, "20260105", no_fills(), "rank_cap", 0)


def test_fast_due_at_index_ten_but_control_is_not_due():
    module().audit_decision(
        decision("rank_cap_fast", 10), "20260105", no_fills(), "rank_cap_fast", 10
    )
    control = decision("rank_cap", 10)
    with pytest.raises(ValueError, match="OFF_SCHEDULE_OPTIONAL_PLAN"):
        module().audit_decision(control, "20260105", no_fills(), "rank_cap", 10)
    control["status"] = "NOT_SCHEDULED"
    module().audit_decision(control, "20260105", no_fills(), "rank_cap", 10)


@pytest.mark.parametrize("exception", ["previous_retry", "held_before"])
def test_off_schedule_plans_require_retry_or_empty_book(exception):
    record = decision("rank_cap", 1)
    arguments = {exception: exception == "previous_retry"}
    module().audit_decision(record, "20260105", no_fills(), "rank_cap", 1, **arguments)


def test_not_scheduled_day_cannot_trade_and_scheduled_day_cannot_be_skipped():
    record = {**decision("rank_cap", 1), "status": "NOT_SCHEDULED"}
    with pytest.raises(ValueError, match="FILL_ON_NOT_SCHEDULED_DAY"):
        module().audit_decision(
            record, "20260105", pd.DataFrame({"symbol": ["A"], "side": ["BUY"]}), "rank_cap", 1
        )
    with pytest.raises(ValueError, match="SCHEDULED_OR_RETRY_SESSION_SKIPPED"):
        module().audit_decision(
            {**decision(), "status": "NOT_SCHEDULED"}, "20260105", no_fills(), "rank_cap", 0
        )


@pytest.mark.parametrize(
    "allocator", ["rank_cap", "rank_cap_fast", "rank_cap_buffer", "rank_cap_fast_buffer"]
)
def test_optional_exit_budget_cannot_change_with_case_name(allocator):
    record = decision(allocator)
    record["allocation"]["max_optional_exits"] = 0
    with pytest.raises(ValueError, match="OPTIONAL_EXIT_BUDGET_DIFFERS_FROM_FROZEN_CASE"):
        module().audit_decision(record, "20260105", no_fills(), allocator, 0)


def test_raw_buffer_optional_exits_are_checked_without_sector_score_transform():
    record = decision("rank_cap_buffer")
    record["allocation"]["optional_exits"] = list("ABCDEF")
    with pytest.raises(ValueError, match="OPTIONAL_MEMBERSHIP_EXIT_BUFFER_FAILED"):
        module().audit_decision(record, "20260105", no_fills(), "rank_cap_buffer", 0)


def test_protected_day_can_only_sell_declared_mandatory_exits():
    record = {
        **decision(),
        "status": "BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED",
        "exit_symbols": ["A"],
    }
    with pytest.raises(ValueError, match="OPTIONAL_FILL_ON_PROTECTED_DAY"):
        module().audit_decision(
            record, "20260105", pd.DataFrame({"symbol": ["A"], "side": ["BUY"]}), "rank_cap", 0
        )


def test_factorial_comparisons_isolate_both_factors_and_combined_change():
    specs = module().comparison_specs()
    assert len(specs) == 5 and len({(a, b) for a, b, _ in specs}) == 5
    cases = module().CASE_SPECS
    for candidate, reference, role in specs[:4]:
        changed = {
            key for key in cases[candidate] if cases[candidate][key] != cases[reference][key]
        }
        assert len(changed) == 1
        assert changed == (
            {"max_optional_exits"} if role.startswith("buffer") else {"rebalance_sessions"}
        )
