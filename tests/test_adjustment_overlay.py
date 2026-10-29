import numpy as np
import pandas as pd
import pytest

from moneymore.research.adjustment_overlay import non_economic_overlay


def native(factors):
    return pd.DataFrame(
        {
            "ts_code": ["A"] * len(factors),
            "trade_date": [f"2024010{i + 1}" for i in range(len(factors))],
            "adj_factor": factors,
            "close": [10] * len(factors),
        }
    )


def test_only_verified_step_removed_real_later_ratio_retained():
    source = native([1, 2, 2, 3])
    result = non_economic_overlay(source, [("A", "20240102")])
    assert result.derived_adj_factor.tolist() == [1, 1, 1, 1.5]
    pd.testing.assert_series_equal(result.adj_factor, source.adj_factor)
    pd.testing.assert_series_equal(result.close, source.close)


def test_paired_transient_step_and_revert_both_removed():
    result = non_economic_overlay(native([1, 2, 1, 1.1]), [("A", "20240102"), ("A", "20240103")])
    assert result.derived_adj_factor.tolist() == [1, 1, 1, 1.1]


def test_missing_factor_never_forward_filled():
    result = non_economic_overlay(native([1, 2, np.nan, 2.2]), [("A", "20240102")])
    assert np.isnan(result.derived_adj_factor.iloc[2])


@pytest.mark.parametrize(
    "targets", [[("A", "20240101")], [("A", "20240109")], [("A", "20240102"), ("A", "20240102")]]
)
def test_invalid_target_rejected(targets):
    with pytest.raises(ValueError):
        non_economic_overlay(native([1, 2, 2]), targets)
