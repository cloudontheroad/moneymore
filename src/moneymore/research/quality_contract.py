"""Explicit research-only contract revision; not a legacy style-bounds override."""
import numpy as np
import pandas as pd
from .portfolio_error_envelope import robust_bounds


def quality_upper_only_bounds(controls, benchmark, limits, envelope):
    """Keep original economic bands and unknown upper caps; no unknown floor.

    Unknown lower bound is absolute zero minus reference unknown weight.
    Do not reserve a negative error buffer for a physical nonnegative weight.
    The positive envelope includes label deterioration, not just price drift.
    """
    if (not controls.index.is_unique or not controls.columns.is_unique or not benchmark.index.is_unique
            or set(benchmark.index)!=set(controls.index)
            or not np.isfinite(controls.to_numpy()).all()
            or not np.isfinite(benchmark).all() or (benchmark<0).any()
            or abs(benchmark.sum()-1)>1e-8):
        raise ValueError("INVALID_QUALITY_CONTRACT_INPUTS")
    base=pd.Series(limits,dtype=float).reindex(controls.columns)
    if not np.isfinite(base).all() or (base<0).any():raise ValueError("INVALID_QUALITY_LIMITS")
    try:
        margins=np.array([[envelope[f][side] for side in ('upper','lower')] for f in controls.columns],dtype=float)
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError("INVALID_OR_MISSING_QUALITY_ERROR_ENVELOPE") from error
    if not np.isfinite(margins).all() or (margins<0).any():
        raise ValueError("INVALID_OR_MISSING_QUALITY_ERROR_ENVELOPE")
    lo,hi=robust_bounds(base,envelope)
    reference=benchmark.reindex(controls.index) @ controls
    for field in controls.columns:
        if field.startswith("unknown_"):
            if not controls[field].isin([0.,1.]).all():raise ValueError("QUALITY_INDICATOR_NOT_BINARY")
            lo[field]=-float(reference[field])
    return lo,hi
