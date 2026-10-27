"""Convex phase-I feasibility restoration, not an alpha or allocation formula."""

import numpy as np


def attainable_style_retention(
    controls, current, *, maximum_weight=0.15, turnover=0.20, desired=0.75
):
    """Minimize worst retained active exposure under the SAME trading budget.

    This LP never observes future returns or scores. Qlib still chooses the final
    return/risk portfolio. The requested contraction is weakened explicitly when
    it is infeasible, rather than removing the turnover constraint silently.
    """
    import cvxpy as cp

    if not 0 <= desired <= 1 or not 0 <= turnover <= 2 or not 0 < maximum_weight <= 1:
        raise ValueError("INVALID_PHASE_ONE_POLICY")
    z = (controls - controls.mean()) / controls.std(ddof=0).replace(0, 1)
    w0 = current.reindex(z.index, fill_value=0).to_numpy(float)
    if (
        not np.isfinite(z.to_numpy()).all()
        or not np.isfinite(w0).all()
        or (w0 < 0).any()
        or abs(w0.sum() - 1) > 1e-6
    ):
        raise ValueError("INVALID_PHASE_ONE_INPUT")
    if set(current[current > 1e-10].index) - set(z.index):
        raise ValueError("PHASE_ONE_HELD_ASSET_MISSING")
    wb = np.full(len(w0), 1 / len(w0))
    exposure = z.to_numpy()
    initial = np.abs((w0 - wb) @ exposure)
    weights = cp.Variable(len(w0))
    retention = cp.Variable(nonneg=True)
    active = (weights - wb) @ exposure
    constraints = [
        weights >= 0,
        weights <= maximum_weight,
        cp.sum(weights) == 1,
        cp.norm1(weights - w0) <= turnover,
        active <= cp.multiply(initial, retention),
        active >= -cp.multiply(initial, retention),
    ]
    problem = cp.Problem(cp.Minimize(retention), constraints)
    problem.solve(solver=cp.ECOS, max_iters=200)
    if problem.status != "optimal" or weights.value is None or not np.isfinite(weights.value).all():
        raise ValueError("PHASE_ONE_INFEASIBLE_OR_INACCURATE")
    witness = weights.value
    minimum = float(retention.value)
    if (
        minimum > 1 + 1e-5
        or abs(witness.sum() - 1) > 1e-5
        or witness.min() < -1e-5
        or witness.max() > maximum_weight + 1e-5
        or np.abs(witness - w0).sum() > turnover + 1e-5
        or (np.abs((witness - wb) @ exposure) > initial * minimum + 1e-5).any()
    ):
        raise ValueError("PHASE_ONE_POLICY_CONFLICT_OR_INVALID_WITNESS")
    chosen = min(1.0, max(desired, minimum + 1e-3))
    return {
        "minimum_retention": minimum,
        "chosen_retention": chosen,
        "desired_retention": desired,
        "goal_relaxed": chosen > desired + 1e-8,
        "witness_turnover_l1": float(np.abs(witness - w0).sum()),
        "witness_weights": dict(zip(z.index, map(float, witness))),
        "method": "CVXPY_ECOS_MINIMAX_ACTIVE_EXPOSURE_PHASE_I",
        "future_returns_used": False,
    }


def classify_history_gaps(symbol, missing_dates, listing_date, full_suspensions):
    """Never call the first locally stored bar a listing date."""
    rows = []
    for date in missing_dates:
        reason = (
            "PRE_LISTING"
            if listing_date and date < listing_date
            else "OFFICIAL_FULL_SESSION_SUSPENSION"
            if (symbol, date) in full_suspensions
            else "UNEXPLAINED_DATA_GAP"
        )
        rows.append({"symbol": symbol, "date": date, "reason": reason})
    return rows
