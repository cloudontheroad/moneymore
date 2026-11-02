import pandas as pd
import pytest
from moneymore.research.style_bounds import style_bounds


def test_tight_bounds_are_not_replaced_with_symmetric_limits():
    lo,hi=style_bounds({"beta":.1},["beta"],{"beta":-.02},{"beta":.03})
    assert lo.beta==-.02 and hi.beta==.03


@pytest.mark.parametrize("lo,hi",[({"beta":-.2},{"beta":.03}),({"beta":.04},{"beta":.03}),({"beta":0.},None),({"other":0.},{"beta":.03})])
def test_bad_asymmetric_proof_is_rejected_not_defaulted(lo,hi):
    with pytest.raises(ValueError):
        style_bounds({"beta":.1},pd.Index(["beta"]),lo,hi)
