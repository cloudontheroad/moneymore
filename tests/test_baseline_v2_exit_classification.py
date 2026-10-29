import pandas as pd
import pytest

from moneymore.research.baseline_v2_daily_readiness import (
    classify_held_inputs,
    construct_daily_targets,
)


def test_explicit_universe_exclusion_is_not_a_missing_data_fault():
    assert classify_held_inputs(
        {"A": 100, "B": 100, "C": 100}, {"A"}, universe_members={"A", "B"}
    ) == {"universe_exits": ["C"], "input_faults": ["B"]}
    assert classify_held_inputs({"C": 100}, {"A"}) == {"universe_exits": [], "input_faults": ["C"]}
    with pytest.raises(ValueError, match="INVALID_COMPLETE"):
        classify_held_inputs({}, {"A"}, universe_members={"B"})


def test_exit_can_be_constructed_but_pool_member_missing_data_still_blocks():
    scores = pd.DataFrame(
        {"date": [pd.Timestamp("20240131")] * 2, "symbol": ["A", "B"], "score": [2.0, 1.0]}
    )
    prices = pd.DataFrame(
        {"A": [10.0] * 50, "B": [20.0] * 50, "C": [30.0] * 50},
        index=pd.date_range("2023-12-01", periods=50),
    )
    policy = {
        "portfolio": {
            "top_k": 2,
            "exit_rank": 3,
            "max_replacements": 1,
            "gross_exposure": 1.0,
            "minimum_weight": 0.2,
            "maximum_weight": 0.8,
            "correlation_lookback": 40,
            "cluster_correlation_threshold": 0.99,
            "maximum_cluster_members": 2,
            "weighting_method": "rank_linear",
        }
    }
    selected, weights, _ = construct_daily_targets(
        scores, {"C": 100}, prices, "20240131", "20240201", policy, universe_members={"A", "B"}
    )
    assert selected == ["A", "B"] and "C" not in weights
    with pytest.raises(ValueError, match="HELD_V2_INPUT_INCOMPLETE"):
        construct_daily_targets(
            scores,
            {"C": 100},
            prices,
            "20240131",
            "20240201",
            policy,
            universe_members={"A", "B", "C"},
        )
