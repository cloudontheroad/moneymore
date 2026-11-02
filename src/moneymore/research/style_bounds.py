"""Resolve symmetric or tightening-only asymmetric style contracts."""
import numpy as np
import pandas as pd


def style_bounds(limits, columns, lower=None, upper=None):
    base=pd.Series(limits,dtype=float).reindex(columns)
    if not base.index.is_unique or not np.isfinite(base).all() or (base<0).any():
        raise ValueError("INVALID_BASE_STYLE_LIMITS")
    if (lower is None)!=(upper is None):
        raise ValueError("BOTH_ASYMMETRIC_LIMITS_REQUIRED")
    lo=-base if lower is None else pd.Series(lower,dtype=float).reindex(columns)
    hi=base if upper is None else pd.Series(upper,dtype=float).reindex(columns)
    if (not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo>hi).any()
            or (lo < -base-1e-12).any() or (hi > base+1e-12).any()):
        raise ValueError("MISSING_INFEASIBLE_OR_RELAXED_ASYMMETRIC_LIMITS")
    return lo,hi
