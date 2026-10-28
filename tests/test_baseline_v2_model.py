import copy

import numpy as np
import pandas as pd
import pytest

from moneymore.research.baseline_v2_model import comparable_scores


def policy():
    return {
        "factors": {"roe": 0.6, "dividend_yield_ttm": 0.4},
        "required_factors": ["roe"],
        "minimum_cross_section_assets": 5,
        "minimum_industry_assets": 5,
    }


def sample():
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("20261027")] * 24,
            "symbol": [f"{i:06d}.SZ" for i in range(24)],
            "industry": ["A"] * 12 + ["B"] * 12,
            "total_mv": np.exp(np.linspace(2, 5, 24)),
            "roe": np.array([12, 10, 19, 13, 17, 18, 11, 14, 15, 13, 9, 16] * 2, dtype=float),
            "dividend_yield_ttm": np.linspace(0.01, 0.07, 24),
        }
    )


def test_optional_missing_is_neutral_normalized_signal_not_raw_zero_or_reweight():
    raw = sample()
    raw.loc[0, "dividend_yield_ttm"] = np.nan
    before = raw.copy()
    out = comparable_scores(raw, policy())
    row = out.loc[out.symbol.eq("000000.SZ")].iloc[0]
    assert row.missing__dividend_yield_ttm
    assert row.neutral_signal__dividend_yield_ttm
    assert row.contribution__dividend_yield_ttm == 0
    assert row.effective_weight__roe == 0.6
    assert row.effective_weight__dividend_yield_ttm == 0.4
    assert row.observed_weight == 0.6
    assert row.score == pytest.approx(row.roe * 0.6)
    pd.testing.assert_frame_equal(raw, before)


def test_missing_masks_follow_sorted_symbols_not_original_row_order():
    raw = sample()
    raw.loc[3, "dividend_yield_ttm"] = np.nan
    out = comparable_scores(raw.iloc[::-1], policy())
    assert out.loc[out.missing__dividend_yield_ttm, "symbol"].tolist() == ["000003.SZ"]


def test_missing_core_is_never_imputed():
    raw = sample()
    raw.loc[0, "roe"] = np.nan
    with pytest.raises(ValueError, match="MISSING_CORE_FACTOR"):
        comparable_scores(raw, policy())


def test_invalid_size_and_unknown_industry_fail_closed():
    raw = sample()
    raw.loc[0, "total_mv"] = 0
    with pytest.raises(ValueError, match="INVALID_SIZE_CONTROL"):
        comparable_scores(raw, policy())
    raw = sample()
    raw.loc[0, "industry"] = None
    with pytest.raises(ValueError, match="MISSING_INDUSTRY"):
        comparable_scores(raw, policy())


def test_industry_level_offset_does_not_impose_industry_stock_quotas():
    raw = sample()
    shifted = raw.copy()
    shifted.loc[shifted.industry.eq("B"), "roe"] += 20
    out = comparable_scores(shifted, policy())
    assert len(out) == 24
    assert "target_weight" not in out
    assert "selected" not in out
    for f in policy()["factors"]:
        assert out["effective_weight__" + f].nunique() == 1


def test_entire_optional_column_missing_cannot_masquerade_as_good_data():
    raw = sample()
    raw["dividend_yield_ttm"] = np.nan
    with pytest.raises(ValueError, match="FACTOR_CONTROL_SAMPLE_INSUFFICIENT"):
        comparable_scores(raw, policy())


def test_version_policy_inputs_remain_unchanged():
    p = policy()
    before = copy.deepcopy(p)
    comparable_scores(sample(), p)
    assert p == before


def test_validated_linear_score_uses_native_coefficients_exactly():
    p = policy()
    p["validated_linear_model"] = {
        "native_model": "qlib.contrib.model.linear.LinearModel",
        "coefficients": {"roe": 0.3, "dividend_yield_ttm": -0.2},
        "intercept": 0.1,
        "validation_evidence_sha256": "test-only-evidence",
    }
    scores = comparable_scores(sample(), p)
    assert np.allclose(scores.score, 0.3 * scores.roe - 0.2 * scores.dividend_yield_ttm + 0.1)
    assert np.allclose(
        scores.score,
        scores.contribution__roe
        + scores.contribution__dividend_yield_ttm
        + scores.intercept_contribution,
    )


def test_fit_success_does_not_authorize_release():
    from moneymore.research.baseline_v2_model import require_weight_release_evidence

    with pytest.raises(ValueError, match="NOT_RELEASE_APPROVED"):
        require_weight_release_evidence(
            {"status": "PASS_RESEARCH_ONLY", "gate_failures": [], "real_orders_enabled": False}
        )
