"""Isolated economic factor overlay. Native fields/history are never overwritten."""

import numpy as np
import pandas as pd


def non_economic_overlay(native, targets):
    """Remove ONLY proven non-economic step ratios, retaining every other ratio.

    ``targets`` must already have original-source and independent-proof hashes
    validated by the caller. Missing native factors remain missing; no filling.
    """
    frame = native.sort_values(["ts_code", "trade_date"]).reset_index(drop=True).copy()
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("DUPLICATE_NATIVE_FACTOR_ROW")
    keys = list(zip(frame.ts_code, frame.trade_date, strict=True))
    selected = set(targets)
    if len(selected) != len(targets) or not selected.issubset(set(keys)):
        raise ValueError("DUPLICATE_OR_MISSING_OVERLAY_TARGET")
    factors = pd.to_numeric(frame.adj_factor, errors="coerce")
    previous = factors.groupby(frame.ts_code).shift()
    ratio = factors / previous
    apply = pd.Series([key in selected for key in keys], index=frame.index)
    if not (np.isfinite(ratio.loc[apply]) & ratio.loc[apply].gt(0)).all():
        raise ValueError("OVERLAY_REQUIRES_KNOWN_POSITIVE_PREDECESSOR")
    if ratio.loc[apply].sub(1).abs().le(1e-10).any():
        raise ValueError("OVERLAY_TARGET_HAS_NO_NATIVE_STEP")
    correction = pd.Series(1.0, index=frame.index)
    correction.loc[apply] = 1.0 / ratio.loc[apply]
    frame["adjustment_overlay_multiplier"] = correction.groupby(frame.ts_code).cumprod()
    frame["derived_adj_factor"] = factors * frame.adjustment_overlay_multiplier
    frame["overlay_step_removed"] = apply
    if not np.isfinite(frame.adjustment_overlay_multiplier).all():
        raise ValueError("OVERLAY_MULTIPLIER_OVERFLOW")
    return frame
