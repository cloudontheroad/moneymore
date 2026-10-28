import copy

import numpy as np
import pandas as pd
import pytest

from moneymore.research.baseline_v2_daily_scores import score_pair


def inputs():
    policy = {"factors": {"roe": 0.6, "dividend_yield_ttm": 0.4}}
    weights = {
        "fixed_weights": dict(policy["factors"]),
        "diagnostic_coefficients_not_released": {"roe": -0.2, "dividend_yield_ttm": 0.3},
    }
    return policy, weights


def test_paired_ranking_uses_identical_normalized_features_once(monkeypatch):
    calls = []
    normalized = pd.DataFrame(
        {
            "symbol": ["B", "A", "C"],
            "roe": [1.0, 1.0, 0.0],
            "dividend_yield_ttm": [0.0, 0.0, 1.0],
            "score": [0.6, 0.6, 0.4],
        }
    )

    def preprocess(raw, policy, **kwargs):
        calls.append(kwargs)
        return normalized.copy()

    monkeypatch.setattr("moneymore.research.baseline_v2_daily_scores.comparable_scores", preprocess)
    policy, weights = inputs()
    original = copy.deepcopy(weights)
    out = score_pair(pd.DataFrame(), policy, weights)
    assert len(calls) == 1 and calls[0] == {"size_control": True, "industry_control": True}
    np.testing.assert_allclose(out.score_learned, [-0.2, -0.2, 0.3])
    assert out.set_index("symbol").rank_fixed.to_dict() == {"A": 1.0, "B": 2.0, "C": 3.0}
    assert out.set_index("symbol").rank_learned.to_dict() == {"C": 1.0, "A": 2.0, "B": 3.0}
    assert weights == original


def test_changed_fixed_weights_and_invalid_learned_values_block():
    policy, weights = inputs()
    policy["factors"]["roe"] = 0.5
    with pytest.raises(ValueError, match="FROZEN_FIXED_WEIGHT_POLICY_MISMATCH"):
        score_pair(pd.DataFrame(), policy, weights)
    policy, weights = inputs()
    weights["diagnostic_coefficients_not_released"]["roe"] = float("nan")
    with pytest.raises(ValueError, match="INVALID_FROZEN"):
        score_pair(pd.DataFrame(), policy, weights)
    weights["diagnostic_coefficients_not_released"] = {"roe": 0.0, "dividend_yield_ttm": 0.0}
    with pytest.raises(ValueError, match="INVALID_FROZEN"):
        score_pair(pd.DataFrame(), policy, weights)


def test_diagnostic_does_not_accept_a_production_model_override():
    policy, weights = inputs()
    policy["validated_linear_model"] = {"coefficients": {"roe": 999.0}}
    with pytest.raises(ValueError, match="FROZEN_FIXED"):
        score_pair(pd.DataFrame(), policy, weights)
