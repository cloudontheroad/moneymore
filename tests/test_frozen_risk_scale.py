import copy
import numpy as np
import pandas as pd
import pytest
from moneymore.research.frozen_risk_scale import CORE, fit_scale, apply_scale


def sample(offset=0.):
    raw = pd.DataFrame({name:np.array([1.,2.,4.])+offset for name in CORE},index=["A","B","C"])
    raw["industry"] = ["a","a","b"]; raw["listing_board"] = ["MAIN","STAR","MAIN"]
    return raw


def test_fit_is_not_inflated_by_across_day_common_market_levels():
    fit = fit_scale({"20260902":sample(),"20260903":sample(10000.)})
    assert fit["scale"]["log_size"] == pytest.approx(np.std([1.,2.,4.]))
    with pytest.raises(ValueError):
        fit_scale({"20261012":sample()})


def test_same_raw_risk_region_requires_converted_not_unchanged_limits():
    artifact = fit_scale({"20260902":sample()})
    snapshot = copy.deepcopy(artifact)
    future = sample(); future[CORE] *= 3.
    daily,fixed,ratio = apply_scale(future,artifact)
    active = pd.Series({"A":.1,"B":-.1,"C":0.})
    a = active @ daily; b = active @ fixed
    pd.testing.assert_series_equal(b,a*ratio)
    assert ((abs(a)>.1)==(abs(b)>.1*ratio)).all()
    assert artifact == snapshot
    pd.testing.assert_frame_equal(daily.drop(columns=CORE),fixed.drop(columns=CORE))


def test_each_session_has_equal_weight_despite_universe_size_change():
    a=pd.DataFrame({name:[0.,2.] for name in CORE},index=["A","B"])
    b=pd.DataFrame({name:[0.,0.,0.,4.] for name in CORE},index=["A","B","C","D"])
    for frame in [a,b]:
        frame["industry"]="x";frame["listing_board"]="MAIN"
    artifact=fit_scale({"20260902":a,"20260903":b})
    assert artifact["scale"]["log_size"]==pytest.approx(np.sqrt(2.))


def test_unidentified_calibration_scale_cannot_silently_normalize_new_variation():
    raw=sample();raw["log_size"]=1.
    artifact=fit_scale({"20260902":raw})
    with pytest.raises(ValueError,match="UNIDENTIFIED"):
        apply_scale(sample(),artifact)
