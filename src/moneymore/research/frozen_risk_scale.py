"""Training-only within-cross-section risk units; no portfolio or limit policy."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from moneymore.research.style_missing_controls import uncertain_controls

CORE = ["log_size", "momentum60", "volatility20", "market_beta120", "residual_volatility120"]


def fit_scale(frames, calibration_end="20260930", evaluation_start="20261012"):
    if not frames or max(frames) > calibration_end or calibration_end >= evaluation_start:
        raise ValueError("FUTURE_OR_EMPTY_SCALE_FIT")
    columns = None; arrays = []; weights = []
    for date, raw in sorted(frames.items()):
        _, eligible, metadata = uncertain_controls(raw, CORE)
        numeric = list(raw.select_dtypes(include="number").columns)
        if columns is not None and columns != numeric:
            raise ValueError("CALIBRATION_NUMERIC_SCHEMA_CHANGED")
        columns = numeric
        x = raw.loc[eligible, numeric].replace([np.inf,-np.inf],np.nan).fillna(pd.Series(metadata["medians"]))
        # Remove each day's common level. Pooled across-day market moves must
        # not inflate the cross-sectional scale and silently loosen limits.
        arrays.append((x-x.mean()).to_numpy())
        weights.extend([1/len(x)]*len(x))
    scaler = StandardScaler().fit(np.vstack(arrays), sample_weight=np.asarray(weights))
    return {"version":"within_day_equal_session_standard_scaler_v1", "calibration_start":min(frames),
        "calibration_end":calibration_end, "evaluation_start":evaluation_start, "sessions":len(frames),
        "scale":dict(zip(columns,map(float,scaler.scale_))),
        "zero_variance_columns":[columns[i] for i,v in enumerate(scaler.var_) if v <= 1e-20],
        "risk_limits_selected":False, "estimator":"sklearn.StandardScaler; each session equal total sample weight"}


def apply_scale(raw, artifact):
    daily, eligible, metadata = uncertain_controls(raw, CORE)
    if list(metadata["standard_deviations"]) != list(artifact["scale"]):
        raise ValueError("FROZEN_SCALE_SCHEMA_MISMATCH")
    fixed = daily.copy(); ratio = pd.Series(1., index=daily.columns)
    for name, frozen in artifact["scale"].items():
        if not np.isfinite(frozen) or frozen <= 0:
            raise ValueError("INVALID_FROZEN_SCALE")
        if name in artifact["zero_variance_columns"] and raw.loc[eligible,name].nunique(dropna=True) > 1:
            raise ValueError("UNIDENTIFIED_CALIBRATION_SCALE_BECAME_VARIABLE")
        ratio[name] = metadata["standard_deviations"][name]/frozen
        fixed[name] *= ratio[name]
    return daily, fixed, ratio
