"""Observed error envelopes for diagnostic portfolios, no coverage guarantee."""
import numpy as np
import pandas as pd


def fit_envelope(errors, calibration_end):
    if errors.empty or errors.to_date.max() > calibration_end:
        raise ValueError("EMPTY_OR_FUTURE_ERROR_CALIBRATION")
    if errors.duplicated(["to_date","probe","factor"]).any() or not np.isfinite(errors.error).all():
        raise ValueError("INVALID_OR_DUPLICATE_ERROR_OBSERVATIONS")
    return {factor:{"upper":max(0.,float(group.error.max())),"lower":max(0.,float(-group.error.min())),
        "days":int(group.to_date.nunique())} for factor,group in errors.groupby("factor")}


def robust_bounds(limits, envelope):
    if set(limits.index)-set(envelope):
        raise ValueError("UNSEEN_FACTOR_WITHOUT_BUFFER_NOT_ZERO")
    lower=pd.Series({f:-float(limits[f])+envelope[f]["lower"] for f in limits.index})
    upper=pd.Series({f:float(limits[f])-envelope[f]["upper"] for f in limits.index})
    return lower,upper
