"""Native Qlib risk estimation and enhanced-index optimizer, offline and fail-closed.

No formula replacement, broker calls, model training or NAV writes occur here.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnhancedPolicy:
    lamb: float = 1.0
    delta: float = 0.20
    maximum_weight: float = 0.15
    num_factors: int = 10
    size_neutral: bool = False


def check_weights(w, w0, wb, exposures, factor_limits, policy, tolerance=1e-5):
    failures = []
    if not np.isfinite(w).all():
        return ["NONFINITE_WEIGHTS"]
    if abs(float(w.sum()) - 1) > tolerance:
        failures.append("GROSS_EXPOSURE")
    if w.min() < -tolerance or w.max() > policy.maximum_weight + tolerance:
        failures.append("WEIGHT_BOUNDS")
    if float(np.abs(w - w0).sum()) > policy.delta + tolerance:
        failures.append("TURNOVER_LIMIT")
    if factor_limits is not None and np.any(
        np.abs((w - wb) @ exposures) > factor_limits + tolerance
    ):
        failures.append("FACTOR_EXPOSURE")
    return failures


def optimize_snapshot(
    scores: pd.Series,
    returns: pd.DataFrame,
    size: pd.Series,
    current_weights: pd.Series,
    policy: EnhancedPolicy,
    *,
    audit_capture: dict | None = None,
    style_controls: pd.DataFrame | None = None,
    style_retention: float = 0.0,
    solver_max_iters: int = 200,
) -> dict:
    """Fully-invested STOCK sleeve; current_weights and benchmark both sum to one.

    The native optimizer's volatility score scaling is deliberately disclosed as
    uncalibrated. This experimental output is never eligible for live promotion.
    """
    import cvxpy as cp
    from qlib.contrib.strategy.optimizer import enhanced_indexing as native
    from qlib.model.riskmodel import StructuredCovEstimator

    symbols = list(returns.columns)
    if "ECOS" not in cp.installed_solvers():
        raise RuntimeError("Qlib native optimizer requires ECOS")
    if len(symbols) < 1 / policy.maximum_weight or len(returns) < max(30, policy.num_factors + 2):
        raise ValueError("INSUFFICIENT_RISK_SAMPLE")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("MISSING_RISK_RETURNS_NO_IMPLICIT_FILL")
    signal = scores.reindex(symbols).to_numpy(dtype=float)
    capital = size.reindex(symbols).to_numpy(dtype=float)
    if not np.isfinite(signal).all() or np.std(signal) <= 1e-12:
        raise ValueError("INVALID_OR_CONSTANT_SCORE")
    if not np.isfinite(capital).all() or (capital <= 0).any():
        raise ValueError("MISSING_MARKET_CAP_CONTROL")
    if not set(current_weights[current_weights > 1e-10].index).issubset(symbols):
        raise ValueError("HELD_ASSET_MISSING_FROM_RISK_UNIVERSE")
    w0 = current_weights.reindex(symbols, fill_value=0).to_numpy(dtype=float)
    if abs(w0.sum() - 1) > 1e-6 or (w0 < 0).any():
        raise ValueError("CURRENT_STOCK_SLEEVE_MUST_SUM_TO_ONE")
    wb = np.full(len(symbols), 1 / len(symbols))
    risk = StructuredCovEstimator(
        num_factors=min(policy.num_factors, len(symbols) - 1),
        factor_model="pca",
        scale_return=False,
    )
    F, covariance, specific = risk.predict(
        returns.copy(), is_price=False, return_decomposed_components=True
    )
    covariance = np.atleast_2d(covariance)
    if (
        not np.isfinite(covariance).all()
        or (specific < 0).any()
        or np.linalg.eigvalsh(covariance).min() < -1e-10
    ):
        raise ValueError("INVALID_RISK_DECOMPOSITION")
    log_size = np.log(capital)
    log_size = (log_size - log_size.mean()) / max(log_size.std(), 1e-12)
    factors = F
    cov = covariance
    limits = None
    if policy.size_neutral:
        # Last column is an explicit exposure constraint, not an extra risk
        # variance term: zero covariance prevents double counting PCA risk.
        factors = np.column_stack([F, log_size])
        cov = np.zeros((factors.shape[1], factors.shape[1]))
        cov[:-1, :-1] = covariance
        # PCA bounds are mathematically nonbinding for long-only unit portfolios.
        # Only the log-size active exposure is required to be zero.
        limits = np.r_[np.ptp(F, axis=0), 0.0]
    if style_controls is not None:
        if not 0 <= style_retention <= 1:
            raise ValueError("INVALID_STYLE_RETENTION")
        controls = style_controls.reindex(symbols)
        if controls.empty or not np.isfinite(controls.to_numpy()).all():
            raise ValueError("INVALID_NAMED_STYLE_CONTROLS")
        controls = (controls - controls.mean()) / controls.std(ddof=0).replace(0, 1)
        factors = np.column_stack([factors, controls.to_numpy()])
        extended = np.zeros((factors.shape[1], factors.shape[1]))
        extended[:cov.shape[0], :cov.shape[1]] = cov
        cov = extended
        control_limits = np.abs((w0 - wb) @ controls.to_numpy()) * style_retention
        limits = np.r_[np.ptp(F, axis=0), ([0.] if policy.size_neutral else []), control_limits]
    if audit_capture is not None:
        audit_capture.update(
            factor_exposure=factors,
            factor_covariance=cov,
            specific_variance=specific,
            current_weights=w0,
            benchmark_weights=wb,
            raw_scores=signal,
            log_size=log_size,
            symbols=np.array(symbols),
        )
    events = []

    class Capture(logging.Handler):
        def emit(self, record):
            events.append(record.getMessage())

    handler = Capture()
    native.logger.addHandler(handler)
    try:
        optimizer = native.EnhancedIndexingOptimizer(
            lamb=policy.lamb,
            delta=policy.delta,
            b_dev=policy.maximum_weight - 1 / len(symbols),
            f_dev=limits,
            scale_return=True,
            epsilon=0.0,
            solver_kwargs={"max_iters": solver_max_iters},
        )
        weights = optimizer(signal.copy(), factors, cov, specific, w0, wb)
    finally:
        native.logger.removeHandler(handler)
    failures = check_weights(weights, w0, wb, factors, limits, policy)
    if any("optimization failed, will return" in e for e in events):
        failures.append("NATIVE_SOLVER_FALLBACK")
    if any("optimization is inaccurate" in e for e in events):
        failures.append("INACCURATE_SOLVE")

    def metrics(w):
        active = w - wb
        loading = active @ F
        variance = float(loading @ covariance @ loading + specific @ (active**2))
        return {
            "annualized_tracking_error": float(np.sqrt(max(variance, 0) * 252)),
            "active_log_size_exposure": float(active @ log_size),
            "effective_holdings": float(1 / np.sum(w**2)),
            "maximum_weight": float(w.max()),
            "holdings_above_1bp": int((w > 0.0001).sum()),
        }

    path = inspect.getsourcefile(native.EnhancedIndexingOptimizer)
    from pathlib import Path

    return {
        "status": "PASS_OFFLINE_ONLY" if not failures else "BLOCKED",
        "failures": failures,
        "weights": dict(zip(symbols, map(float, weights))) if not failures else {},
        "stock_sleeve_gross": 1.0,
        "turnover_l1": float(np.abs(weights - w0).sum()),
        "turnover_constraint_relaxed_by_native": any("removing turnover" in e for e in events),
        "before": metrics(w0),
        "after": metrics(weights),
        "solver_events": events,
        "actual_size_control_column": "standardized_log_total_mv" if policy.size_neutral else None,
        "named_style_active_exposures": {
            name: {"before": float((w0 - wb) @ controls[name].to_numpy()),
                   "after": float((weights - wb) @ controls[name].to_numpy())}
            for name in (style_controls.columns if style_controls is not None else [])
        },
        "named_style_retention": style_retention if style_controls is not None else None,
        "benchmark_weights": dict(zip(symbols, map(float, wb))),
        "native_optimizer_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "risk_estimator": "qlib.StructuredCovEstimator.PCA",
        "risk_return_units": "daily_decimal",
        "signal_units": "QLIB_NATIVE_VOLATILITY_SCALING_NOT_EXPECTED_RETURN_CALIBRATION",
        "automatic_promotion": False,
        "orders_enabled": False,
    }


def carry_confirmed_suspensions(
    prices: pd.DataFrame, proof: pd.DataFrame
) -> tuple[pd.DataFrame, list]:
    """Only documented full-session suspensions may carry an earlier known price.

    Unknown gaps stay missing; this never writes synthetic bars into the market store.
    """
    result = prices.copy()
    carried = []
    if proof.empty:
        return result, carried
    if "verified_full_session" in proof:
        full = proof.loc[proof.suspend_type.eq("S") & proof.verified_full_session.eq(True)]
    elif "suspend_timing" in proof:
        full = proof.loc[
            proof.suspend_type.eq("S") & (proof.suspend_timing.isna() | proof.suspend_timing.eq(""))
        ]
    else:
        return result, carried
    for row in full.sort_values("trade_date").itertuples():
        if row.ts_code not in result or row.trade_date not in result.index:
            continue
        if pd.notna(result.loc[row.trade_date, row.ts_code]):
            continue
        earlier = result.loc[result.index < row.trade_date, row.ts_code]
        if earlier.empty or pd.isna(earlier.iloc[-1]):
            continue
        result.loc[row.trade_date, row.ts_code] = earlier.iloc[-1]
        carried.append(
            {
                "symbol": row.ts_code,
                "trade_date": row.trade_date,
                "reason": "OFFICIAL_FULL_SESSION_SUSPENSION",
            }
        )
    return result, carried
