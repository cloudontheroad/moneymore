"""Independent invariants; a successful solve is not strategy validation."""
import numpy as np


def constraint_violations(weights, benchmark, current, controls, limits,
                          maximum_weight=.15, turnover=.20, tolerance=1e-6):
    w = np.asarray(weights, float)
    b = np.asarray(benchmark, float)
    if not np.isfinite(w).all():
        return ["NONFINITE_WEIGHTS"]
    controls = np.asarray(controls, float)
    limits = np.asarray(limits, float)
    current = np.asarray(current, float)
    if (controls.ndim != 2 or controls.shape[0] != len(w)
            or limits.shape != (controls.shape[1],) or b.shape != w.shape or current.shape != w.shape
            or not np.isfinite(controls).all() or not np.isfinite(limits).all()
            or not np.isfinite(b).all() or not np.isfinite(current).all() or (limits < 0).any()):
        return ["INVALID_OR_MISSING_CONSTRAINT_INPUTS"]
    failures = []
    if abs(w.sum()-1) > tolerance or w.min() < -tolerance or w.max() > maximum_weight+tolerance:
        failures.append("WEIGHT_BUDGET_OR_BOUNDS")
    if np.abs(w-current).sum() > turnover+tolerance:
        failures.append("TURNOVER")
    if np.any(np.abs((w-b) @ controls) > limits+tolerance):
        failures.append("NAMED_EXPOSURE")
    return failures
