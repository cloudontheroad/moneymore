import numpy as np
import pandas as pd
from moneymore.research.style_risk_inputs import signed_inverse


def test_negative_earnings_not_discarded_as_a_risk_descriptor():
    result = signed_inverse(pd.Series([10., -20., 0., np.nan, np.inf]))
    assert result.iloc[0] == .1
    assert result.iloc[1] == -.05
    assert result.iloc[2:].isna().all()
