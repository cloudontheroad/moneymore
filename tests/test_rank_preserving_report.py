import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "rank_preserving_report",
        Path(__file__).resolve().parents[1] / "scripts/report_ashare_rank_preserving.py",
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metrics():
    return {
        "net_return": 0.1,
        "sharpe": 1.0,
        "maximum_drawdown": 0.05,
        "annualized_volatility": 0.1,
        "mean_actual_stock_exposure": 0.9,
        "minimum_actual_stock_exposure": 0.8,
        "maximum_actual_stock_exposure": 0.95,
        "annualized_half_gross_turnover": 3.0,
        "fees": 100.0,
        "slippage": 300.0,
        "fills": 100,
        "mean_largest_industry_weight": 0.3,
        "maximum_industry_weight": 0.5,
        "mean_growth_board_weight": 0.1,
        "gross_shares_traded": 10000,
        "buy_shares": 6000,
        "sell_shares": 4000,
        "trading_days": 10,
        "mean_held_names": 50,
        "industry_mean_weights": {"bank": 0.3},
    }


def equity():
    returns = np.random.default_rng(5).normal(0.001, 0.01, 181)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=181).strftime("%Y%m%d"),
            "equity": 1000 * np.cumprod(1 + returns),
        }
    )


def test_identical_paired_cases_have_zero_differences():
    result = module().compare_cases(metrics(), metrics(), equity(), equity(), 1000)
    assert result["net_return_difference_ci95"] == [0.0, 0.0]
    assert result["sharpe_difference_ci95"] == [0.0, 0.0]
    assert result["drawdown_reduction"] == 0
    assert result["execution_cost_difference"] == 0
    assert result["industry_mean_weight_differences"] == {"bank": 0}


def test_comparison_signs_and_actual_exposure_are_explicit():
    candidate = {
        **metrics(),
        "net_return": 0.11,
        "maximum_drawdown": 0.04,
        "mean_actual_stock_exposure": 0.85,
        "fees": 150,
        "industry_mean_weights": {"bank": 0.1, "other": 0.2},
    }
    result = module().compare_cases(candidate, metrics(), equity(), equity(), 1000)
    assert result["net_return_difference"] == pytest.approx(0.01)
    assert result["drawdown_reduction"] == pytest.approx(0.01)
    assert result["mean_stock_exposure_difference"] == pytest.approx(-0.05)
    assert result["execution_cost_difference"] == 50
    assert result["industry_mean_weight_differences"] == pytest.approx({"bank": -0.2, "other": 0.2})


def test_paired_comparison_rejects_different_or_reordered_calendar():
    shifted = equity()
    shifted.loc[0, "date"] = "20260104"
    with pytest.raises(ValueError, match="PAIRED_CALENDAR_MISMATCH"):
        module().compare_cases(metrics(), metrics(), equity(), shifted, 1000)
    with pytest.raises(ValueError, match="INVALID_COMPARISON_EQUITY"):
        module().daily_returns(equity().iloc[::-1], 1000)


def test_daily_returns_include_initial_build_day():
    frame = pd.DataFrame({"date": ["20260105", "20260106"], "equity": [990.0, 1000.0]})
    assert module().daily_returns(frame, 1000) == pytest.approx([-0.01, 1000 / 990 - 1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0, -1])
def test_bad_equity_fails_closed(bad):
    frame = equity()
    frame.loc[1, "equity"] = bad
    with pytest.raises(ValueError, match="INVALID_COMPARISON_EQUITY"):
        module().daily_returns(frame, 1000)


def test_frozen_control_allows_roundoff_but_rejects_date_price_and_order_changes():
    expected = pd.DataFrame(
        {"date": ["20260105", "20260106"], "symbol": ["A", "B"], "price": [10.0, 11.0]}
    )
    actual = expected.copy()
    actual.loc[0, "price"] += 1e-10
    module().assert_replay_equal(actual, expected, "test")
    for field, bad in (("date", "20260104"), ("symbol", "B"), ("price", 10.001)):
        actual = expected.copy()
        actual.loc[0, field] = bad
        with pytest.raises(ValueError, match="FROZEN_TOPK_REPLAY_CHANGED"):
            module().assert_replay_equal(actual, expected, "test")
    with pytest.raises(ValueError, match="FROZEN_TOPK_REPLAY_CHANGED"):
        module().assert_replay_equal(expected.iloc[::-1], expected, "test")


def fills():
    return pd.DataFrame(
        {
            "date": ["20260105", "20260105"],
            "symbol": ["A", "B"],
            "side": ["BUY", "SELL"],
            "quantity": [200, 100],
            "fee": [5.0, 6.0],
            "slippage": [1.0, 2.0],
            "notional": [2000.0, 1000.0],
        }
    )


def allocation():
    return {
        "status": "INSUFFICIENT_CAPACITY",
        "industry_active_limit": 0.08,
        "benchmark_industry_weights": {"bank": 0.2},
        "frozen_industry_weights": {},
        "selected_industry_weights": {"bank": 0.0199},
        "total_industry_weights": {"bank": 0.0199},
        "industry_weight_caps": {"bank": 0.28},
        "new_symbols": ["A"],
        "retained_symbols": [],
        "selected_count": 1,
        "requested_slots": 50,
        "slot_weight": 0.0199,
        "selected_weight": 0.0199,
        "frozen_weight": 0.0,
        "total_weight": 0.0199,
        "target_exposure": 0.995,
        "max_optional_exits": 5,
        "optional_exits": [],
    }


def decision():
    return {
        "signal_date": "20251231",
        "execution_date": "20260105",
        "status": "OPEN_PLAN_VERIFIED",
        "risk_audit": {"risk_end": "20251231"},
        "open_postcheck": {
            "passed": True,
            "actual_stock_exposure": 0.9,
            "maximum_stock_weight": 0.03,
            "maximum_industry_violation": 0,
            "stock_limit": 0.05,
        },
        "allocation": allocation(),
        "target_quantities": {"A": 200},
        "executable_quantities": {"A": 200},
        "next_open_used_for_selection": False,
    }


def test_frozen_symbol_never_trades_in_verified_plan():
    record = {**decision(), "frozen_symbols": ["A"]}
    with pytest.raises(ValueError, match="FROZEN_POSITION_WAS_TRADED"):
        module().audit_decision(record, "20260105", fills(), "rank_cap")


def test_exit_only_must_sell_only_declared_exits():
    record = {**decision(), "status": "MANDATORY_EXITS_ONLY", "exit_symbols": ["B"]}
    module().audit_decision(record, "20260105", fills().iloc[1:], "rank_cap")
    with pytest.raises(ValueError, match="OPTIONAL_TRADE_ON_EXIT_ONLY_DAY"):
        module().audit_decision(record, "20260105", fills(), "rank_cap")
    record["exit_symbols"] = ["C"]
    with pytest.raises(ValueError, match="OPTIONAL_TRADE_ON_EXIT_ONLY_DAY"):
        module().audit_decision(record, "20260105", fills().iloc[1:], "rank_cap")


@pytest.mark.parametrize(
    "change,error",
    [
        ({"signal_date": "20260105"}, "DECISION_TIMELINE_MISMATCH"),
        ({"next_open_used_for_selection": True}, "NEXT_OPEN_USED_FOR_SELECTION"),
        ({"risk_audit": {"risk_end": "20260105"}}, "RISK_TIMELINE_MISMATCH"),
        ({"open_postcheck": {"passed": False}}, "OPEN_INTEGER_RISK_POSTCHECK"),
        ({"open_postcheck": None}, "OPEN_INTEGER_RISK_POSTCHECK"),
        ({"executable_quantities": {"A": 100.5}}, "NONINTEGER_OR_NEGATIVE_SHARE_PLAN"),
    ],
)
def test_decision_audit_rejects_contract_violations(change, error):
    with pytest.raises(ValueError, match=error):
        module().audit_decision({**decision(), **change}, "20260105", fills(), "rank_cap")


def test_executed_share_changes_are_signed_and_include_costs():
    changes = module().executed_share_changes(fills()).set_index("symbol")
    assert changes.signed_executed_shares.to_dict() == {"A": 200, "B": -100}
    assert changes.buy_shares.sum() == 200 and changes.sell_shares.sum() == 100
    assert changes.fee.sum() == 11 and changes.slippage.sum() == 3


def test_topk_controls_do_not_claim_industry_cap_postcheck():
    record = {**decision(), "open_postcheck": None}
    del record["allocation"]
    result = module().audit_decision(record, "20260105", fills(), "sector_topk")
    assert not result["open_postcheck_verified"] and not result["allocation_caps_verified"]


def test_report_independently_rejects_postcheck_marked_passed_but_violated():
    record = decision()
    record["open_postcheck"]["maximum_stock_weight"] = 0.051
    with pytest.raises(ValueError, match="INDEPENDENT_OPEN_INTEGER_RISK_POSTCHECK_FAILED"):
        module().audit_decision(record, "20260105", fills(), "rank_cap")


def test_report_checks_open_industry_detail_when_available():
    record = decision()
    record["open_postcheck"].update(
        actual_stock_exposure=0.2, industry_weights={"bank": 0.2}, industry_limits={"bank": 0.3}
    )
    module().audit_decision(record, "20260105", fills(), "rank_cap")
    record["open_postcheck"]["industry_limits"]["bank"] = 0.5
    with pytest.raises(ValueError, match="INDEPENDENT_OPEN_INDUSTRY_WEIGHTS_FAILED"):
        module().audit_decision(record, "20260105", fills(), "rank_cap")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("industry_weight_caps", {"bank": 0.4}, "INDEPENDENT_ALLOCATION_INDUSTRY_CAP_FAILED"),
        ("total_industry_weights", {"bank": 0.5}, "INDEPENDENT_ALLOCATION_INDUSTRY_CAP_FAILED"),
        ("selected_count", 2, "INDEPENDENT_ALLOCATION_BUDGET_FAILED"),
        ("target_exposure", 1.0, "INDEPENDENT_ALLOCATION_BUDGET_FAILED"),
        ("optional_exits", list("ABCDEF"), "OPTIONAL_MEMBERSHIP_EXIT_BUFFER_FAILED"),
    ],
)
def test_independent_allocation_audit_does_not_trust_status(field, value, error):
    with pytest.raises(ValueError, match=error):
        module().audit_allocation({**allocation(), field: value}, "sector_rank_buffer")


def test_receivable_cash_can_reduce_exposure_without_relaxing_slot_accounting():
    record = allocation()
    record.update(
        slot_weight=0.019, target_exposure=0.95, selected_weight=0.019, total_weight=0.019
    )
    record["selected_industry_weights"] = {"bank": 0.019}
    record["total_industry_weights"] = {"bank": 0.019}
    module().audit_allocation(record, "rank_cap")
    record["target_exposure"] = 0.94
    with pytest.raises(ValueError, match="INDEPENDENT_ALLOCATION_BUDGET_FAILED"):
        module().audit_allocation(record, "rank_cap")


def test_rank_equal_uses_nonbinding_industry_limits_but_keeps_stock_risk_check():
    record = decision()
    record["allocation"].update(industry_active_limit=1.0, industry_weight_caps={"bank": 1.2})
    record["open_postcheck"].update(
        actual_stock_exposure=0.9, industry_weights={"bank": 0.9}, industry_limits={"bank": 1.2}
    )
    result = module().audit_decision(record, "20260105", fills(), "rank_equal")
    assert result["open_postcheck_verified"] and result["allocation_caps_verified"]
    record["open_postcheck"]["maximum_stock_weight"] = 0.051
    with pytest.raises(ValueError, match="INDEPENDENT_OPEN_INTEGER_RISK_POSTCHECK_FAILED"):
        module().audit_decision(record, "20260105", fills(), "rank_equal")


def test_rank_equal_control_cannot_silently_reuse_binding_sector_cap():
    with pytest.raises(ValueError, match="ALLOCATION_CAP_CONTRACT_CHANGED"):
        module().audit_allocation(allocation(), "rank_equal")
    record = decision()
    record["allocation"].update(industry_active_limit=1.0, industry_weight_caps={"bank": 1.2})
    record["open_postcheck"].update(
        actual_stock_exposure=0.2, industry_weights={"bank": 0.2}, industry_limits={"bank": 0.3}
    )
    with pytest.raises(ValueError, match="INDEPENDENT_OPEN_INDUSTRY_WEIGHTS_FAILED"):
        module().audit_decision(record, "20260105", fills(), "rank_equal")


def test_original_case_regression_rejects_even_tiny_numeric_change():
    expected = equity()
    module().assert_exact_replay_equal(expected.copy(), expected, "equity")
    actual = expected.copy()
    actual.loc[0, "equity"] += 1e-10
    with pytest.raises(ValueError, match="ORIGINAL_CASE_NOT_EXACTLY_REPRODUCED"):
        module().assert_exact_replay_equal(actual, expected, "equity")


def test_original_case_regression_rejects_changed_schema_and_fill_key_order():
    expected = fills()
    with pytest.raises(ValueError, match="ORIGINAL_CASE_NOT_EXACTLY_REPRODUCED"):
        module().assert_exact_replay_equal(
            expected.iloc[::-1].reset_index(drop=True), expected, "fills"
        )
    actual = expected.copy()
    actual.quantity = actual.quantity.astype(float)
    with pytest.raises(ValueError, match="ORIGINAL_CASE_NOT_EXACTLY_REPRODUCED"):
        module().assert_exact_replay_equal(actual, expected, "fills")
