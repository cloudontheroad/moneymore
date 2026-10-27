import numpy as np
import pandas as pd
import pytest

from moneymore.research.style_feasibility import attainable_style_retention, classify_history_gaps


def test_phase_one_relaxes_goal_not_turnover():
    controls = pd.DataFrame({"style": [-1.0, -1.0, 1.0, 1.0]}, index=list("ABCD"))
    w0 = pd.Series([0.45, 0.45, 0.05, 0.05], index=controls.index)
    result = attainable_style_retention(
        controls, w0, maximum_weight=0.5, turnover=0.02, desired=0.75
    )
    assert result["goal_relaxed"]
    assert 0.97 < result["chosen_retention"] <= 1
    assert result["witness_turnover_l1"] <= 0.02001
    assert not result["future_returns_used"]


def test_phase_one_does_not_relax_concentration_policy():
    controls = pd.DataFrame({"style": [1.0, 2.0, 3.0, 4.0]}, index=list("ABCD"))
    with pytest.raises(ValueError, match="INFEASIBLE"):
        attainable_style_retention(
            controls,
            pd.Series([1.0, 0.0, 0.0, 0.0], index=controls.index),
            maximum_weight=0.3,
            turnover=0.01,
        )


def test_listing_and_suspension_proof_are_not_general_imputation():
    rows = classify_history_gaps(
        "A", ["20260101", "20260103", "20260104"], "20260102", {("A", "20260103")}
    )
    assert [r["reason"] for r in rows] == [
        "PRE_LISTING",
        "OFFICIAL_FULL_SESSION_SUSPENSION",
        "UNEXPLAINED_DATA_GAP",
    ]
    assert (
        classify_history_gaps("A", ["20260101"], None, set())[0]["reason"] == "UNEXPLAINED_DATA_GAP"
    )


def test_phase_one_limits_are_enforced_by_native_qlib():
    from moneymore.research.enhanced_indexing import EnhancedPolicy, optimize_snapshot

    rng = np.random.default_rng(91)
    symbols = list("ABCDEFGHIJKL")
    returns = pd.DataFrame(rng.normal(0, 0.01, (60, 12)), columns=symbols)
    styles = pd.DataFrame({"momentum": returns.tail(20).sum(), "volatility": returns.std()})
    w0 = pd.Series([0.14] * 6 + [0.16 / 6] * 6, index=symbols)
    phase = attainable_style_retention(styles, w0)
    solved = optimize_snapshot(
        pd.Series(rng.normal(size=12), index=symbols),
        returns,
        pd.Series(np.arange(1, 13) * 100.0, index=symbols),
        w0,
        EnhancedPolicy(lamb=10.0, num_factors=3),
        style_controls=styles,
        style_retention=phase["chosen_retention"],
        solver_max_iters=1000,
    )
    assert solved["status"] == "PASS_OFFLINE_ONLY"
    assert solved["turnover_l1"] <= 0.20001
    for exposure in solved["named_style_active_exposures"].values():
        assert abs(exposure["after"]) <= abs(exposure["before"]) * phase["chosen_retention"] + 1e-5
