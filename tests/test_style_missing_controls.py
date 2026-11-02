import numpy as np
import pandas as pd
import pytest
from moneymore.research.style_missing_controls import uncertain_controls


def test_optional_missing_is_visible_but_required_missing_excluded():
    raw = pd.DataFrame({"size": [1., 2., np.nan], "value": [1., np.nan, 3.],
        "industry": ["a", "b", "c"], "listing_board": ["MAIN", "STAR", "MAIN"]})
    original = raw.copy(deep=True)
    controls, eligible, metadata = uncertain_controls(raw, ["size"])
    assert eligible.tolist() == [True, True, False]
    assert controls.unknown_value.tolist() == [0., 1.]
    assert np.isfinite(controls.to_numpy()).all()
    assert metadata["not_observed_facts"]
    pd.testing.assert_frame_equal(original, raw)


def test_all_unknown_not_filled_with_zero():
    raw = pd.DataFrame({"size": [1.], "value": [np.nan], "industry": ["a"], "listing_board": ["MAIN"]})
    with pytest.raises(ValueError):
        uncertain_controls(raw, ["size"])
