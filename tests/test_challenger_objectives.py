import numpy as np
import pandas as pd
import pytest
from moneymore.challenger_objectives import alternative_label, fit_rank_blend


def test_labels_do_not_confuse_relative_with_absolute():
    frame = pd.DataFrame({"datetime": ["2026-01-01"]*2, "sector": ["tech"]*2,
        "future_total_return": [-.10, -.20], "trailing_risk": [.05, .10]})
    assert alternative_label(frame, "sector_total_return").iloc[0] > 0
    assert alternative_label(frame, "absolute_total_return").iloc[0] < 0
    assert alternative_label(frame, "risk_adjusted_total_return").tolist() == [-2., -2.]


def test_blend_can_reject_inverted_model():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2025-01-01", periods=40), range(30)], names=["datetime", "instrument"])
    y = pd.Series(np.tile(np.arange(30), 40), index=idx, dtype=float)
    result = fit_rank_blend({"good": y, "bad": -y}, y)
    assert result["weights"]["good"] > .99
    assert abs(sum(result["weights"].values())-1) < 1e-6
    assert not result["test_data_used"]
    with pytest.raises(ValueError):
        fit_rank_blend({"good": y.iloc[:30]}, y.iloc[:30])
