import numpy as np
import pandas as pd
import pytest

from moneymore.research.baseline_v2_daily_readiness import daily_schedule, validate_snapshot


def test_constructor_rejects_stale_scores_and_missing_held_inputs():
    from moneymore.research.baseline_v2_daily_readiness import construct_daily_targets

    scores = pd.DataFrame({"date": ["2024-01-31"], "symbol": ["A"], "score": [1.0]})
    with pytest.raises(ValueError, match="EXACT_DAILY"):
        construct_daily_targets(scores, {}, pd.DataFrame(), "20240201", "20240202", {})
    with pytest.raises(ValueError, match="HELD_V2_INPUT"):
        construct_daily_targets(scores, {"B": 100}, pd.DataFrame(), "20240131", "20240201", {})
    with pytest.raises(ValueError, match="SIGNAL_MUST"):
        construct_daily_targets(scores, {}, pd.DataFrame(), "20240131", "20240131", {})


def test_schedule_requires_exact_prior_session():
    assert daily_schedule(["20240130", "20240131", "20240201"], "20240131", "20240201") == [
        ("20240131", "20240130"),
        ("20240201", "20240131"),
    ]


def test_constructor_uses_actual_incumbents_and_excludes_future_prices():
    from moneymore.research.baseline_v2_daily_readiness import construct_daily_targets

    prices = pd.DataFrame(
        np.random.default_rng(1729).uniform(10, 20, (61, 5)),
        index=pd.date_range("2024-01-01", periods=61),
        columns=list("ABCDE"),
    )
    scores = pd.DataFrame(
        {"date": "2024-02-28", "symbol": list("ABCDE"), "score": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )
    policy = {
        "portfolio": {
            "correlation_lookback": 50,
            "top_k": 2,
            "exit_rank": 3,
            "max_replacements": 1,
            "gross_exposure": 1.0,
            "minimum_weight": 0.2,
            "maximum_weight": 0.8,
            "cluster_correlation_threshold": 0.99,
            "maximum_cluster_members": 2,
            "weighting_method": "rank_linear",
        }
    }
    selected, weights, _ = construct_daily_targets(
        scores, {"C": 100, "D": 100}, prices, "20240228", "20240229", policy
    )
    assert selected == ["A", "C"]
    prices.loc[prices.index > "2024-02-28"] = 999999
    repeated = construct_daily_targets(
        scores, {"C": 100, "D": 100}, prices, "20240228", "20240229", policy
    )
    assert repeated[0] == selected and repeated[1] == weights
    fresh = construct_daily_targets(scores, {}, prices, "20240228", "20240229", policy)
    assert fresh[0] == ["A", "B"]


def test_snapshot_rejects_monthly_substitute_and_duplicates():
    f = pd.DataFrame(
        {
            "ts_code": [f"{600000 + i}.SH" for i in range(1000)],
            "trade_date": "20240130",
            "close": 10,
            "pb": 1,
            "pe_ttm": 10,
            "dv_ttm": 2,
            "total_mv": range(1, 1001),
        }
    )
    assert validate_snapshot(f, "20240130").iloc[0].ts_code == "600999.SH"
    with pytest.raises(ValueError, match="DATE_OR_DUPLICATE"):
        validate_snapshot(f, "20240131")
    f.loc[1, "ts_code"] = f.loc[0, "ts_code"]
    with pytest.raises(ValueError, match="DATE_OR_DUPLICATE"):
        validate_snapshot(f, "20240130")
