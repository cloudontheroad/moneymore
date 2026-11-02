"""Exposure accounting and repair-state evidence, never trading instructions."""
import numpy as np
import pandas as pd


def centered_exposure(quantities, prices, centered_controls):
    q = pd.Series(quantities, dtype=float)
    if not q.index.is_unique or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("INVALID_QUANTITIES")
    q = q.loc[q > 0]
    controls = centered_controls
    if not controls.index.is_unique or not controls.columns.is_unique or set(q.index)-set(controls.index):
        raise ValueError("MISSING_HELD_CONTROL_NOT_ZERO_EXPOSURE")
    prices = pd.Series(prices).reindex(q.index).astype(float)
    z = controls.reindex(q.index).astype(float)
    if q.empty or not np.isfinite(prices).all() or (prices <= 0).any() or not np.isfinite(z.to_numpy()).all():
        raise ValueError("MISSING_MARK_OR_CONTROL")
    values = q*prices
    return (values/values.sum()) @ z


def decompose_drift(open_after, close_frozen, close_updated):
    a, b, c = (pd.Series(x, dtype=float) for x in (open_after, close_frozen, close_updated))
    common = a.index.intersection(b.index).intersection(c.index)
    if not len(common) or not np.isfinite(pd.concat([a[common], b[common], c[common]]).to_numpy()).all():
        raise ValueError("NO_COMPARABLE_FINITE_FACTOR_COORDINATES")
    result = pd.DataFrame({"open_after": a[common], "close_frozen": b[common], "close_updated": c[common]})
    result["within_day_weight_effect"] = result.close_frozen-result.open_after
    # Combines updated factor observations, normalization, reference weights
    # and universe changes. This is not named-factor return attribution.
    result["control_and_reference_update_effect"] = result.close_updated-result.close_frozen
    result["identity_residual"] = result.close_updated-result.open_after-result.within_day_weight_effect-result.control_and_reference_update_effect
    return result


def repair_state(before, after, limit, *, next_session, plan_passed, tolerance=1e-6):
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("INVALID_MONITOR_LIMIT")
    if not next_session:
        return "PENDING_WINDOW_END"
    if not plan_passed:
        return "PENDING_BLOCKED_EXECUTION"
    if before is None or after is None or not np.isfinite([before, after]).all():
        return "PENDING_MISSING_EVIDENCE"
    if abs(after) > limit+tolerance:
        return "STILL_OUTSIDE_AFTER_EXECUTION"
    if abs(before) <= limit+tolerance:
        return "NATURAL_REENTRY_BEFORE_EXECUTION"
    return "REPAIRED_AFTER_EXECUTED_TRADES"
