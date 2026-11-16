"""Run native cvxportfolio in a separate dependency environment, without market I/O."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import cvxportfolio as cvx
import numpy as np
import pandas as pd


def solve(request, arrays):
    symbols = request["symbols"]
    if len(set(symbols)) != len(symbols) or "CASH" in symbols:
        raise ValueError("INVALID_SYMBOLS")
    covariance = arrays["covariance"]
    mu = np.asarray(request["mu"], dtype=float)
    initial = np.asarray(request["weights"], dtype=float)
    benchmark = np.asarray(request["benchmark"], dtype=float)
    exposure = np.asarray(request["industry_exposure"], dtype=float)
    industry_limit = np.asarray(request["industry_limits"], dtype=float)
    if (
        covariance.shape != (len(symbols), len(symbols))
        or len(mu) != len(symbols)
        or len(initial) != len(symbols) + 1
        or not np.isfinite(covariance).all()
        or not np.isfinite(mu).all()
        or not np.isfinite(initial).all()
        or abs(initial.sum() - 1) > 1e-6
        or initial.min() < -1e-6
    ):
        raise ValueError("INVALID_OPTIMIZATION_INPUT")
    sigma = pd.DataFrame(covariance, index=symbols, columns=symbols)
    objective = cvx.ReturnsForecast(pd.Series(mu, index=symbols)) - request[
        "risk_aversion"
    ] * cvx.FullCovariance(sigma)
    buy, sell = request["buy_rate"], request["sell_rate"]
    if request["include_cost"]:
        objective -= cvx.TransactionCost(a=(buy + sell) / 2, b=None, c=(buy - sell) / 2)
    constraints = [
        cvx.LongOnly(),
        cvx.FixedImbalance(request["stock_exposure"]),
        cvx.MaxWeights(pd.Series(request["upper_weights"], index=symbols)),
        cvx.MinWeights(pd.Series(request["lower_weights"], index=symbols)),
        cvx.TurnoverLimit(request["turnover_limit"]),
        cvx.FactorMaxLimit(pd.DataFrame(exposure, index=symbols), industry_limit),
    ]
    index = symbols + ["CASH"]
    policy = cvx.SinglePeriodOptimization(
        objective,
        constraints,
        include_cash_return=False,
        benchmark=cvx.FixedWeights(pd.Series(benchmark, index=index)),
        solver="CLARABEL",
        fallback_solver=None,
        max_iter=200,
        ignore_dpp=True,
    )
    value = float(request["equity"])
    trades, _, _ = policy.execute(
        pd.Series(initial * value, index=index),
        market_data=None,
        t=pd.Timestamp(request["signal_date"]),
    )
    weights = initial + trades.reindex(index).to_numpy() / value
    change = weights[:-1] - initial[:-1]
    max_industry = float(np.max(exposure.T @ weights[:-1] - industry_limit))
    violation = max(
        float(-weights.min()),
        abs(weights[:-1].sum() - request["stock_exposure"]),
        float(np.max(weights[:-1] - request["upper_weights"])),
        float(np.max(np.asarray(request["lower_weights"]) - weights[:-1])),
        float(abs(change).sum() / 2 - request["turnover_limit"]),
        max_industry,
    )
    if not np.isfinite(weights).all() or violation > 1e-5:
        raise ValueError("NATIVE_CVX_POSTCHECK_FAILED")
    after = weights[:-1] - benchmark[:-1]
    before = initial[:-1] - benchmark[:-1]
    friction = float(buy * np.maximum(change, 0).sum() + sell * np.maximum(-change, 0).sum())
    return {
        "status": "CONTINUOUS_PLAN_VERIFIED_NOT_EXECUTED",
        "weights": dict(zip(symbols, weights[:-1].tolist())),
        "cash_weight": float(weights[-1]),
        "one_way_turnover": float(abs(change).sum() / 2),
        "expected_return_increment": float(change @ mu),
        "estimated_linear_cost": friction,
        "active_variance_increment": float(
            after @ covariance @ after - before @ covariance @ before
        ),
        "max_constraint_violation": violation,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("cvxportfolio", "cvxpy", "clarabel", "numpy", "pandas")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    try:
        with np.load(args.request.with_suffix(".npz")) as arrays:
            result = solve(request, arrays)
    except Exception as error:
        args.output.write_text(
            json.dumps(
                {
                    "status": "SOLVER_FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
