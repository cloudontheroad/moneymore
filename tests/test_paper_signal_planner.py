from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from moneymore.research.baseline_v2_daily_readiness import construct_daily_targets
from moneymore.research.baseline_v2_daily_scores import score_pair
from moneymore.research.paper_signal_planner import FrozenSignalPlanner


def sample():
    policy = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/baseline_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    rng = np.random.default_rng(18)
    symbols = [f"S{i:03d}" for i in range(100)]
    raw = pd.DataFrame(
        {
            "symbol": symbols,
            "date": pd.Timestamp("20261028"),
            "total_mv": np.arange(100) + 100,
            "industry": [f"I{i % 5}" for i in range(100)],
        }
    )
    for name in policy["factors"]:
        raw[name] = rng.normal(size=100)
    prices = pd.DataFrame(
        np.exp(rng.normal(0, 0.01, (130, 100))).cumprod(axis=0) * 10,
        index=pd.bdate_range(end="20261028", periods=130),
        columns=symbols,
    )
    contract = {
        "native_model": "qlib.contrib.model.linear.LinearModel",
        "score_semantics": "ONCE_NORMALIZED_DIRECTION_ADJUSTED_FEATURES_DOT_COEFFICIENTS_NOT_EXPECTED_RETURN",
        "factor_policy": policy,
        "coefficients": policy["factors"],
        "real_orders_enabled": False,
    }
    return raw, prices, contract


def test_paper_targets_exactly_match_research_shared_path():
    raw, prices, contract = sample()
    planner = FrozenSignalPlanner(contract)
    actual = planner.targets(
        raw, {}, prices, "20261028", "20261029", universe_members=set(raw.symbol)
    )
    scores = score_pair(raw, contract["factor_policy"], planner.weights)
    scores["score"] = scores.score_learned
    selected, weights, _ = construct_daily_targets(
        scores,
        {},
        prices,
        "20261028",
        "20261029",
        contract["factor_policy"],
        universe_members=set(raw.symbol),
    )
    assert actual["target_weights"] == weights
    assert actual["selected"] == selected


def test_held_input_loss_clears_prior_plan_not_a_sell_instruction():
    raw, prices, contract = sample()
    result = FrozenSignalPlanner(contract).targets(
        raw,
        {"MISSING": 100},
        prices,
        "20261028",
        "20261029",
        universe_members=set(raw.symbol) | {"MISSING"},
    )
    assert result["status"] == "HOLD_AND_ALERT" and result["clear_previous_plan"]
    assert result["target_weights"] == {}


def test_held_universe_exit_not_misclassified_as_data_loss():
    raw, prices, contract = sample()
    prices["EXITED"] = 10
    result = FrozenSignalPlanner(contract).targets(
        raw, {"EXITED": 100}, prices, "20261028", "20261029", universe_members=set(raw.symbol)
    )
    assert result["universe_exits"] == ["EXITED"]
    assert result["status"] == "TARGET_READY_NOT_ORDER_SUBMISSION"


def test_wrong_day_scores_rejected():
    raw, prices, contract = sample()
    with pytest.raises(ValueError, match="EXACT_SIGNAL"):
        FrozenSignalPlanner(contract).targets(
            raw, {}, prices, "20261027", "20261029", universe_members=set(raw.symbol)
        )


def test_future_price_rows_do_not_change_targets_and_string_dates_supported():
    raw, prices, contract = sample()
    planner = FrozenSignalPlanner(contract)
    expected = planner.targets(
        raw, {}, prices, "20261028", "20261029", universe_members=set(raw.symbol)
    )
    prices.loc[pd.Timestamp("20261029")] = np.arange(100) * 10000 + 1
    prices.index = prices.index.strftime("%Y%m%d")
    actual = planner.targets(
        raw, {}, prices, "20261028", "20261029", universe_members=set(raw.symbol)
    )
    assert actual["target_weights"] == expected["target_weights"]
