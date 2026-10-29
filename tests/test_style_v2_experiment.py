import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from moneymore.research.baseline_v2_model import comparable_scores
from moneymore.research.security_identity import SecurityIdentity
from moneymore.research.style_v2_experiment import SCOPE, OverlayStore, validate_experiment


def test_overlay_known_past_and_forward_carry_without_imputation(tmp_path):
    overlay = pd.DataFrame({"ts_code": ["A", "A"], "trade_date": ["20250101", "20250102"], "adjustment_overlay_multiplier": [1.0, 0.5]})
    path = tmp_path / "overlay.parquet"
    overlay.to_parquet(path, index=False)

    class Store:
        curated = tmp_path

        def read(self, table, **kwargs):
            return pd.DataFrame({"ts_code": ["A", "A", "A", "Z"], "trade_date": ["20250101", "20250102", "20260101", "20260101"], "adj_factor": [1.0, 2.0, np.nan, 3.0]}, index=[4, 7, 8, 10])

    store = OverlayStore(Store(), path, hashlib.sha256(path.read_bytes()).hexdigest(), SecurityIdentity([]))
    result = store.read("adj_factor")
    assert result.adj_factor.iloc[:2].tolist() == [1.0, 1.0]
    assert pd.isna(result.adj_factor.iloc[2])
    assert result.adj_factor.iloc[3] == 3.0
    assert result.index.tolist() == [4, 7, 8, 10]


def contract(tmp_path):
    path = tmp_path / "state/baseline-v2-style-release/run/contract.json"
    path.parent.mkdir(parents=True)
    evidence = {"scope": SCOPE, "coefficients": {"roe": 0.2, "dividend_yield_ttm": -0.1}, "real_orders_enabled": False, "artifact_hashes": {}}
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return {"real_orders_enabled": False, "automatic_promotion": False, "factors": {"roe": 0.6, "dividend_yield_ttm": 0.4}, "required_factors": ["roe"], "minimum_cross_section_assets": 5, "minimum_industry_assets": 5, "experimental_linear_model": {"scope": SCOPE, "native_model": "qlib.contrib.model.linear.LinearModel", "evidence_path": str(path), "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "coefficients": evidence["coefficients"]}}


def test_experiment_contract_is_not_a_validated_promotion(tmp_path):
    policy = contract(tmp_path)
    validate_experiment(tmp_path, policy)
    assert "validated_linear_model" not in policy
    invalid = copy.deepcopy(policy)
    invalid["automatic_promotion"] = True
    with pytest.raises(ValueError, match="EXPERIMENT_CONTRACT"):
        validate_experiment(tmp_path, invalid)
    invalid = copy.deepcopy(policy)
    invalid["experimental_linear_model"]["coefficients"]["roe"] = 8
    with pytest.raises(ValueError, match="EXPERIMENT_CONTRACT"):
        validate_experiment(tmp_path, invalid)


def test_learned_score_reuses_once_normalized_features(tmp_path):
    policy = contract(tmp_path)
    raw = pd.DataFrame({"date": [pd.Timestamp("20261029")] * 24, "symbol": [str(i) for i in range(24)], "industry": ["A"] * 12 + ["B"] * 12, "total_mv": np.exp(np.linspace(2, 5, 24)), "roe": [12, 10, 19, 13, 17, 18, 11, 14, 15, 13, 9, 16] * 2, "dividend_yield_ttm": np.linspace(.01, .07, 24)})
    base = copy.deepcopy(policy)
    base.pop("experimental_linear_model")
    normalized = comparable_scores(raw, base)
    actual = comparable_scores(raw, policy)
    np.testing.assert_allclose(actual.score, .2 * normalized.roe - .1 * normalized.dividend_yield_ttm)
    assert "target_weight" not in actual  # No board or industry allocation quota.


def test_removed_prototype_cannot_be_registered_again(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from moneymore.baseline_v2_candidate import register_factor_v2

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/baseline_v2.yaml").write_text(json.dumps({"activation_date": "20261028", "candidate_tag": "20261028", "real_orders_enabled": False}), encoding="utf-8")
    tombstone = tmp_path / "state/removed-factor-candidates/20261028.json"
    tombstone.parent.mkdir(parents=True)
    tombstone.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="FACTOR_CANDIDATE_EXPLICITLY_REMOVED"):
        register_factor_v2(tmp_path, None, None, None, now=datetime(2026, 10, 29, tzinfo=ZoneInfo("Asia/Shanghai")))
