"""Explicit uncertain-control research branch, never replaces raw observations."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def uncertain_controls(raw, required):
    numeric = raw.select_dtypes(include="number").columns
    observed = raw[numeric].replace([np.inf, -np.inf], np.nan)
    eligible = observed[list(required)].notna().all(axis=1) & raw.industry.notna() & raw.listing_board.notna()
    selected = observed.loc[eligible]
    medians = selected.median()
    if medians.isna().any() or not len(selected):
        raise ValueError("NO_OBSERVED_REFERENCE_FOR_CONTROL")
    missing = selected.isna()
    filled = selected.fillna(medians)
    scaler = StandardScaler().fit(filled)
    controls = pd.DataFrame(scaler.transform(filled), index=filled.index, columns=numeric)
    # Do not silently imply a median observation. Every estimated field gets
    # its own indicator, constrained separately from the estimated exposure.
    for name in numeric:
        if missing[name].any():
            controls["unknown_"+name] = missing[name].astype(float)
    controls = pd.concat([controls, pd.get_dummies(raw.loc[eligible, "industry"], prefix="industry", dtype=float)], axis=1)
    for board in ("STAR", "CHINEXT"):
        controls["board_"+board] = raw.loc[eligible, "listing_board"].eq(board).astype(float)
    return controls, eligible, {"medians": medians.to_dict(), "standard_deviations": dict(zip(numeric, scaler.scale_)),
        "missing_counts": missing.sum().to_dict(), "not_observed_facts": True}
