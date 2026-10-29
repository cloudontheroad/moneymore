import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "corrected_daily", Path(__file__).resolve().parents[1] / "scripts/replay_corrected_v2.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_corrected_features_are_prefix_causal():
    prices = pd.DataFrame({"A": np.linspace(10, 30, 400), "B": np.linspace(25, 15, 400)})
    expected = module.corrected_features(prices.iloc[:330])
    changed = prices.copy()
    changed.loc[330:, :] *= 5
    actual = module.corrected_features(changed)
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name].iloc[:330])


def test_price_gap_is_not_forward_filled():
    prices = pd.DataFrame({"A": np.arange(1.0, 301.0)})
    prices.loc[280, "A"] = np.nan
    result = module.corrected_features(prices)
    assert pd.isna(result["volatility_60"].loc[281, "A"])
    assert pd.isna(result["drawdown_120"].loc[281, "A"])


def test_variant_selection_does_not_duplicate_inherited_score():
    panel = pd.DataFrame({"score": [1.0], "score_fixed": [1.0], "score_learned": [2.0]})
    selected = module.variant_scores(panel, "learned")
    assert selected.columns.is_unique
    assert selected.score.tolist() == [2.0]
    assert panel.score.tolist() == [1.0]


def test_saved_score_audit_rejects_tampering_and_wrong_tie_rank():
    panel = pd.DataFrame({
        "date": ["20240131", "20240131"], "symbol": ["A", "B"],
        "x": [0.5, 0.5], "score_fixed": [0.5, 0.5],
        "score_learned": [1.0, 1.0], "rank_fixed": [1, 2], "rank_learned": [1, 2],
    })
    policy = {"factors": {"x": 1.0}}
    weights = {"fixed_weights": {"x": 1.0}, "diagnostic_coefficients_not_released": {"x": 2.0}}
    module.audit_scores(panel, policy, weights)
    changed = panel.copy()
    changed.loc[0, "score_learned"] = 2
    with pytest.raises(ValueError, match="SCORE_ARITHMETIC"):
        module.audit_scores(changed, policy, weights)
    changed = panel.copy()
    changed["rank_fixed"] = [2, 1]
    with pytest.raises(ValueError, match="DAILY_RANK"):
        module.audit_scores(changed, policy, weights)
