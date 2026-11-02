"""Pre-registered research hypotheses, not production trading rules."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def alternative_label(frame, protocol):
    if protocol == "sector_total_return":
        return frame.future_total_return - frame.groupby(["datetime", "sector"]).future_total_return.transform("mean")
    if protocol == "absolute_total_return":
        return frame.future_total_return
    if protocol == "risk_adjusted_total_return":
        scale = frame.trailing_risk.where(frame.trailing_risk > 1e-8)
        return frame.future_total_return / scale
    raise ValueError("unknown label protocol")


def fit_rank_blend(predictions, labels, *, minimum_days=30):
    """Learn nonnegative sum-one weights on a separate chronological blend set."""
    scores = pd.concat(predictions, axis=1)
    aligned = scores.join(labels.rename("target")).replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.index.get_level_values("datetime").nunique() < minimum_days:
        raise ValueError("insufficient independent blend-validation days")
    ranks = aligned.groupby(level="datetime").rank(pct=True) - .5
    x = ranks[scores.columns].to_numpy()
    y = ranks.target.to_numpy()
    n = x.shape[1]
    # Every date has equal influence even if cross-section coverage changes.
    row_weight = 1 / aligned.groupby(level="datetime").target.transform("size").to_numpy()
    row_weight /= row_weight.sum()
    def loss(w):
        return float(np.sum(row_weight * (x @ w - y)**2) + .001 * np.sum((w - 1/n)**2))
    result = minimize(loss, np.full(n, 1/n), method="SLSQP", bounds=[(0., 1.)]*n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}], options={"maxiter": 300, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.x).all() or abs(result.x.sum()-1) > 1e-6:
        raise ValueError("blend solver failed")
    return {"weights": dict(zip(scores.columns, map(float, result.x))),
        "days": int(aligned.index.get_level_values("datetime").nunique()), "rows": len(aligned),
        "objective": "date_equal_rank_mse_nonnegative_simplex_ridge_0.001",
        "validation_start": str(aligned.index.get_level_values("datetime").min().date()),
        "validation_end": str(aligned.index.get_level_values("datetime").max().date()),
        "test_data_used": False}
