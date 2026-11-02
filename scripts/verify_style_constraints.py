"""S4/S5 offline constraint mechanisms, strict benchmark neutrality test only."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from qlib.contrib.strategy.optimizer import enhanced_indexing as native
from moneymore.research.style_constraint_checks import constraint_violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("test evidence immutable")
    root = Path(__file__).resolve().parents[1]
    source = root / "state/style-balance/named-risk-20261102-b"
    controls = pd.read_parquet(source / "named-controls.parquet")
    raw = pd.read_parquet(source / "raw-exposures.parquet").reindex(controls.index)
    risk = np.load(source / "qlib-statistical-risk.npz")
    symbols = risk["symbols"].astype(str)
    if list(symbols) != list(controls.index):
        raise ValueError("risk/name ordering mismatch")
    for board in ("STAR", "CHINEXT"):
        controls["board_"+board] = raw.listing_board.eq(board).astype(float)
    C = controls.to_numpy(float)
    F, cov, specific = risk["loadings"], risk["factor_covariance"], risk["specific_variance"]
    wb = np.exp(raw.log_size.to_numpy(float)); wb /= wb.sum()
    w0 = wb.copy()
    max_weight = .15
    deviation = max_weight - wb.max()
    if deviation <= 0:
        raise ValueError("benchmark violates stock cap")
    tech = raw.industry.isin(["电子", "计算机", "通信"]).to_numpy()
    bank = raw.industry.eq("银行").to_numpy()
    growth = raw.listing_board.isin(["STAR", "CHINEXT"]).to_numpy()
    if min(tech.sum(), bank.sum(), growth.sum()) < 10:
        raise ValueError("synthetic scenarios lack enough securities")
    rank_tie = np.arange(len(wb))/len(wb)*1e-4
    cases = {"all_technology_high": tech.astype(float)+rank_tie,
        "all_banks_high": bank.astype(float)+rank_tie,
        "all_growth_boards_high": growth.astype(float)+rank_tie}
    scores_path = root / "state/qlib-challenger/candidates/20261031/latest-refit/current-score-comparison.parquet"
    if scores_path.exists():
        actual = pd.read_parquet(scores_path).latest_refit_score
        actual.index = actual.index.get_level_values("instrument")
        aligned = actual.reindex(symbols)
        if aligned.notna().all():
            cases["latest_completed_model"] = aligned.to_numpy()
    args.output.mkdir(parents=True)
    results = []
    for case, score in cases.items():
        raw_top = np.zeros(len(wb)); raw_top[np.argsort(score)[-10:]] = .1
        for constrained in (False, True):
            # Zero active exposure is a mechanism-test contract, not a fitted
            # trading risk budget. The benchmark always remains feasible.
            limits = np.zeros(C.shape[1])
            combined = np.column_stack([F, C]) if constrained else F
            factor_cov = np.zeros((combined.shape[1], combined.shape[1])); factor_cov[:len(cov), :len(cov)] = cov
            bound = np.r_[np.ptp(F, axis=0), limits] if constrained else None
            messages = []
            class Capture(logging.Handler):
                def emit(self, record): messages.append(record.getMessage())
            handler = Capture(); native.logger.addHandler(handler)
            try:
                solver = native.EnhancedIndexingOptimizer(lamb=1., delta=.20, b_dev=deviation,
                    f_dev=bound, epsilon=0., solver_kwargs={"max_iters": 300})
                weights = solver(score.copy(), combined, factor_cov, specific, w0, wb)
            finally:
                native.logger.removeHandler(handler)
            check_C = C if constrained else np.zeros_like(C)
            failures = constraint_violations(weights, wb, w0, check_C, limits)
            if any("removing turnover" in msg or "optimization failed" in msg or "inaccurate" in msg for msg in messages):
                failures.append("NATIVE_RELAXATION_FALLBACK_OR_INACCURACY")
            # Independent convex formulation with a different solver, using
            # the same risk/score scaling and all of the same constraints.
            q = cp.Variable(len(wb))
            active = q-wb
            scale = np.sqrt(np.mean(np.sum((F @ cov)*F, axis=1)+specific))
            calibrated = score/score.std()*scale
            objective = cp.Maximize(active @ calibrated - cp.quad_form(active @ F, cp.psd_wrap(cov)) - cp.sum(cp.multiply(specific, cp.square(active))))
            lb = np.maximum(0., wb-deviation); ub = np.minimum(1., wb+deviation)
            restrictions = [cp.sum(q) == 1, q >= lb, q <= ub, cp.norm1(q-w0) <= .20]
            if constrained:
                restrictions += [C.T @ active == 0]
            problem = cp.Problem(objective, restrictions)
            problem.solve(solver=cp.CLARABEL, max_iter=300)
            reference_failures = ["REFERENCE_SOLVE_FAILED"] if problem.status != "optimal" or q.value is None else constraint_violations(q.value, wb, w0, check_C, limits)
            def objective_value(w):
                a=w-wb; loading=a @ F
                return float(a @ calibrated-loading @ cov @ loading-specific @ (a*a))
            objective_gap = abs(objective_value(weights)-objective_value(q.value)) if q.value is not None else None
            if objective_gap is None or objective_gap > 1e-5:
                failures.append("CROSS_SOLVER_OBJECTIVE_MISMATCH")
            row = {"case": case, "constrained": constrained,
                "status": "PASS_MECHANISM_ONLY" if not failures and not reference_failures else "BLOCKED",
                "failures": failures, "reference_failures": reference_failures,
                "reference_solver": "CVXPY_CLARABEL_NOT_CVXPORTFOLIO", "objective_gap": objective_gap,
                "raw_top10_growth_share": float(raw_top @ growth), "raw_top10_technology_share": float(raw_top @ tech),
                "benchmark_growth_share": float(wb @ growth), "growth_share": float(weights @ growth),
                "benchmark_technology_share": float(wb @ tech), "technology_share": float(weights @ tech),
                "benchmark_bank_share": float(wb @ bank), "bank_share": float(weights @ bank),
                "max_named_active_exposure": float(np.abs((weights-wb) @ C).max()),
                "turnover_l1": float(np.abs(weights-w0).sum()), "effective_holdings": float(1/(weights @ weights)),
                "solver_messages": messages}
            results.append(row)
            pd.Series(weights, index=symbols, name="weight").to_frame().to_parquet(args.output / f"{case}-{constrained}.parquet")
    report = {"status": "COMPLETED_MECHANISM_TEST_NOT_STRATEGY_APPROVAL", "orders_enabled": False,
        "benchmark": "current_complete_risk_subset_total_cap_renormalized_not_full_top1000",
        "contract": "strict zero-active named industry/style/listing-board exposures, test-only",
        "production_risk_budget_frozen": False, "results": results,
        "source_risk_sha256": hashlib.sha256((source/"risk-model.json").read_bytes()).hexdigest(),
        "source_named_controls_sha256": hashlib.sha256((source/"named-controls.parquet").read_bytes()).hexdigest(),
        "source_statistical_risk_sha256": hashlib.sha256((source/"qlib-statistical-risk.npz").read_bytes()).hexdigest(),
        "source_scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest() if scores_path.exists() else None,
        "limitations": ["799-complete-subset coverage bias remains", "not a historical risk backtest",
            "zero active exposure used for falsification, not chosen production parameters",
            "predictive validity not established", "cvxportfolio integration remains pending"]}
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "solves": len(results), "blocked": sum(r["status"]=="BLOCKED" for r in results)}))


if __name__ == "__main__":
    main()
