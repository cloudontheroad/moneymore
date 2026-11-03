import pandas as pd
import pytest
from moneymore.research.quality_contract import quality_upper_only_bounds


def test_quality_revision_preserves_upper_and_economic_bounds():
    c=pd.DataFrame({"beta":[-1.,1.],"unknown_dividend":[1.,0.]},index=["A","B"])
    lo,hi=quality_upper_only_bounds(c,pd.Series({"A":.3,"B":.7}),{"beta":.1,"unknown_dividend":.01},
        {"beta":{"upper":.02,"lower":.03},"unknown_dividend":{"upper":.002,"lower":.04}})
    assert lo.beta==pytest.approx(-.07) and hi.beta==pytest.approx(.08)
    assert lo.unknown_dividend==-.3
    assert hi.unknown_dividend==.008


def test_deterioration_can_still_make_quality_interval_infeasible():
    c=pd.DataFrame({"unknown_x":[0.,0.]})
    lo,hi=quality_upper_only_bounds(c,pd.Series([.5,.5]),{"unknown_x":.01},
        {"unknown_x":{"upper":.02,"lower":0.}})
    assert lo.unknown_x>hi.unknown_x  # never clip or widen upper cap


def test_missing_quality_indicator_is_rejected():
    with pytest.raises(ValueError):
        quality_upper_only_bounds(pd.DataFrame({"unknown_x":[float("nan")]}),pd.Series([1.]),
            {"unknown_x":.01},{"unknown_x":{"upper":0.,"lower":0.}})


def test_negative_error_margin_cannot_relax_upper_cap():
    with pytest.raises(ValueError,match="ERROR_ENVELOPE"):
        quality_upper_only_bounds(pd.DataFrame({"unknown_x":[0.]}),pd.Series([1.]),
            {"unknown_x":.01},{"unknown_x":{"upper":-.1,"lower":0.}})


def test_revised_quality_contract_cannot_masquerade_as_legacy_tightening():
    from moneymore.research.style_bounds import style_bounds
    c=pd.DataFrame({"unknown_x":[1.,0.]})
    lo,hi=quality_upper_only_bounds(c,pd.Series([.5,.5]),{"unknown_x":.01},
        {"unknown_x":{"upper":.002,"lower":.04}})
    with pytest.raises(ValueError,match="RELAXED"):
        style_bounds({"unknown_x":.01},c.columns,lo,hi)
