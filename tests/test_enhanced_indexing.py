import numpy as np
import pandas as pd
import pytest

from moneymore.research.enhanced_indexing import (
    EnhancedPolicy,
    carry_confirmed_suspensions,
    check_weights,
    optimize_snapshot,
)


def inputs():
    random = np.random.default_rng(31)
    symbols = [str(n) for n in range(12)]
    common = random.normal(0, 0.01, (60, 2))
    exposure = random.normal(size=(2, 12))
    returns = pd.DataFrame(common @ exposure + random.normal(0, 0.005, (60, 12)), columns=symbols)
    scores = pd.Series(random.normal(size=12), index=symbols)
    size = pd.Series(np.exp(np.linspace(10, 15, 12)), index=symbols)
    w0 = pd.Series(1 / 12, index=symbols)
    return scores, returns, size, w0


def test_native_qlib_optimization_and_real_size_control():
    scores, returns, size, w0 = inputs()
    result = optimize_snapshot(
        scores, returns, size, w0, EnhancedPolicy(delta=2, num_factors=3, size_neutral=True)
    )
    assert result["status"] == "PASS_OFFLINE_ONLY"
    assert abs(sum(result["weights"].values()) - 1) < 1e-6
    assert max(result["weights"].values()) <= 0.15001
    assert abs(result["after"]["active_log_size_exposure"]) < 1e-5
    assert result["actual_size_control_column"] == "standardized_log_total_mv"
    assert result["before"]["annualized_tracking_error"] < 1e-8
    assert not result["orders_enabled"] and not result["automatic_promotion"]


def test_constant_scores_and_missing_returns_are_not_imputed():
    scores, returns, size, w0 = inputs()
    with pytest.raises(ValueError, match="CONSTANT_SCORE"):
        optimize_snapshot(scores * 0, returns, size, w0, EnhancedPolicy())
    returns.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NO_IMPLICIT_FILL"):
        optimize_snapshot(scores, returns, size, w0, EnhancedPolicy())


def test_named_style_constraint_uses_native_factor_limits():
    scores, returns, size, w0 = inputs()
    styles = pd.DataFrame({"momentum": returns.tail(20).sum(), "volatility": returns.std()})
    result = optimize_snapshot(
        scores,
        returns,
        size,
        w0,
        EnhancedPolicy(delta=2, num_factors=3),
        style_controls=styles,
        style_limits={"momentum": 0.0, "volatility": 0.0},
    )
    assert result["status"] == "PASS_OFFLINE_ONLY"
    assert all(abs(v["after"]) < 1e-5 for v in result["named_style_active_exposures"].values())


def test_fixed_style_budget_does_not_shrink_with_current_holdings():
    scores, returns, size, w0 = inputs()
    styles = pd.DataFrame({"momentum": returns.tail(20).sum(), "volatility": returns.std()})
    budget = {"momentum": 0.8, "volatility": 0.8}
    for current in (w0, pd.Series(np.linspace(1, 2, 12), index=w0.index) / 18):
        result = optimize_snapshot(
            scores,
            returns,
            size,
            current,
            EnhancedPolicy(delta=2, num_factors=3),
            style_controls=styles,
            style_limits=budget,
        )
        assert result["status"] == "PASS_OFFLINE_ONLY"
        assert result["named_style_limits"] == budget
        assert result["named_style_retention"] is None
        assert result["style_policy"] == "FIXED_EXPLICIT_BUDGET"


def test_objective_preconditioning_preserves_native_economic_problem():
    scores, returns, size, w0 = inputs()
    original = optimize_snapshot(scores, returns, size, w0, EnhancedPolicy(delta=2, num_factors=3))
    scaled = optimize_snapshot(
        scores,
        returns,
        size,
        w0,
        EnhancedPolicy(delta=2, num_factors=3, objective_preconditioning=True),
    )
    assert original["status"] == scaled["status"] == "PASS_OFFLINE_ONLY"
    assert 0 < scaled["native_objective_divisor"] < 1
    assert np.max(np.abs(pd.Series(original["weights"]) - pd.Series(scaled["weights"]))) < 2e-4


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({}, "EXPLICIT_STYLE_BUDGET_REQUIRED"),
        ({"style_limits": {"momentum": 0.8}, "style_retention": 0.75}, "CANNOT_MIX"),
        ({"style_limits": {"wrong": 0.8}}, "SCHEMA_MISMATCH"),
        ({"style_limits": {"momentum": float("nan")}}, "INVALID_FIXED"),
        ({"style_limits": {"momentum": -0.1}}, "INVALID_FIXED"),
    ],
)
def test_fixed_style_budget_rejects_missing_mixed_or_invalid_policy(kwargs, reason):
    scores, returns, size, w0 = inputs()
    styles = pd.DataFrame({"momentum": returns.tail(20).sum()})
    with pytest.raises(ValueError, match=reason):
        optimize_snapshot(
            scores,
            returns,
            size,
            w0,
            EnhancedPolicy(num_factors=3),
            style_controls=styles,
            **kwargs,
        )


def test_budget_without_factor_controls_is_rejected():
    scores, returns, size, w0 = inputs()
    with pytest.raises(ValueError, match="REQUIRES_CONTROLS"):
        optimize_snapshot(
            scores, returns, size, w0, EnhancedPolicy(num_factors=3), style_limits={"momentum": 0.8}
        )


def test_missing_holdings_and_size_cannot_be_dropped():
    scores, returns, size, w0 = inputs()
    w0.index = ["unknown"] + list(w0.index[1:])
    with pytest.raises(ValueError, match="HELD_ASSET_MISSING"):
        optimize_snapshot(scores, returns, size, w0, EnhancedPolicy())
    size.iloc[0] = np.nan
    with pytest.raises(ValueError, match="MARKET_CAP_CONTROL"):
        optimize_snapshot(scores, returns, size, w0, EnhancedPolicy())


def test_postcheck_rejects_relaxed_turnover_and_exposure():
    w = np.array([0.6, 0.4])
    w0 = np.array([0.4, 0.6])
    wb = np.array([0.5, 0.5])
    failures = check_weights(
        w,
        w0,
        wb,
        np.array([[1.0], [-1.0]]),
        np.array([0.0]),
        EnhancedPolicy(delta=0.1, maximum_weight=0.7),
    )
    assert "TURNOVER_LIMIT" in failures
    assert "FACTOR_EXPOSURE" in failures


def test_native_failure_fallback_is_not_a_success(monkeypatch):
    from qlib.contrib.strategy.optimizer import enhanced_indexing as native

    class Failed:
        def __init__(self, **kwargs):
            pass

        def __call__(self, r, F, cov, var, w0, wb):
            native.logger.warning("optimization failed, will return current holding weight")
            return w0

    monkeypatch.setattr(native, "EnhancedIndexingOptimizer", Failed)
    scores, returns, size, w0 = inputs()
    result = optimize_snapshot(scores, returns, size, w0, EnhancedPolicy(num_factors=3))
    assert result["status"] == "BLOCKED"
    assert result["weights"] == {}
    assert "NATIVE_SOLVER_FALLBACK" in result["failures"]


def test_only_official_full_suspensions_carry_no_backfill():
    prices = pd.DataFrame(
        {"A": [10.0, np.nan, np.nan, np.nan], "B": [np.nan, 12.0, np.nan, np.nan]},
        index=["20260101", "20260102", "20260103", "20260104"],
    )
    proof = pd.DataFrame(
        [
            {"ts_code": "A", "trade_date": "20260102", "suspend_type": "S", "suspend_timing": None},
            {
                "ts_code": "A",
                "trade_date": "20260103",
                "suspend_type": "S",
                "suspend_timing": "09:30-10:00",
            },
            {"ts_code": "B", "trade_date": "20260101", "suspend_type": "S", "suspend_timing": None},
        ]
    )
    result, carried = carry_confirmed_suspensions(prices, proof)
    assert result.loc["20260102", "A"] == 10
    assert pd.isna(result.loc["20260103", "A"])
    assert pd.isna(result.loc["20260104", "A"])
    assert pd.isna(result.loc["20260101", "B"])
    assert len(carried) == 1
    no_schema = proof.drop(columns="suspend_timing")
    result, carried = carry_confirmed_suspensions(prices, no_schema)
    assert not carried
