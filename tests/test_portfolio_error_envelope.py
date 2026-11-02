import pandas as pd
import pytest
from moneymore.research.portfolio_error_envelope import fit_envelope,robust_bounds


def test_envelope_uses_only_training_days_and_asymmetric_errors():
    errors=pd.DataFrame({"to_date":["20260903","20260904"],"probe":["a","a"],"factor":["beta","beta"],"error":[.03,-.01]})
    fit=fit_envelope(errors,"20260922")
    lower,upper=robust_bounds(pd.Series({"beta":.1}),fit)
    assert lower.beta==pytest.approx(-.09) and upper.beta==pytest.approx(.07)
    with pytest.raises(ValueError):
        fit_envelope(errors,"20260903")
    with pytest.raises(ValueError):
        robust_bounds(pd.Series({"new":.1}),fit)


def test_infeasible_buffer_is_not_clipped_into_an_acceptable_range():
    lower,upper=robust_bounds(pd.Series({"beta":.1}),{"beta":{"lower":.15,"upper":.15}})
    assert lower.beta>upper.beta
