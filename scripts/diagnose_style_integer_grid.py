"""Finite lot-grid proof versus bounded full-pool solve. Research only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.research.integer_execution import plan_integer_execution


def configuration(capital, cap):
    return BacktestConfig(initial_cash=capital, commission_rate=.00015, minimum_commission=5,
        stamp_duty_rate=.0005, transfer_fee_rate=.00001, slippage_bps=10, lot_size=100,
        max_position_weight=cap, max_gross_exposure=1., max_drawdown=.2)


def small_proof(reference, budget):
    config = configuration(10_000., 1.)
    # All affordable positive-investment integer portfolios, not a search
    # heuristic: 10 lots already exceed cash once slippage/fees are included.
    feasible = []
    for a in range(11):
        for b in range(11):
            if not a+b:
                continue
            notional = np.array([a, b])*1000*1.001
            fee = np.where(notional > 0, np.maximum(5., notional*.00015)+notional*.00001, 0).sum()
            if notional.sum()+fee <= 10000 and abs(b/(a+b)-reference) <= budget+1e-8:
                feasible.append({"A_lots": a, "B_lots": b, "cash": float(10000-notional.sum()-fee)})
    benchmark = {"A": 1-reference, "B": reference}
    result = plan_integer_execution(benchmark, {"A": 10., "B": 10.}, {}, {}, 10000., 10000., config,
        pd.DataFrame({"unknown_optional": [0., 1.]}, index=["A", "B"]), {"unknown_optional": budget},
        benchmark_weights=benchmark, style_controls_prestandardized=True, time_limit=5., solver_threads=2)
    matched = (result["status"] == "PASS_REFERENCE_QUOTES_ONLY") == bool(feasible)
    if not feasible:
        matched = matched and result.get("reason") == "INTEGER_FEASIBILITY_INFEASIBLE"
    if not matched:
        raise ValueError("INTEGER_SOLVER_DISAGREES_WITH_EXHAUSTIVE_GRID")
    return {"reference_unknown_weight": reference, "unknown_active_budget": budget,
        "exhaustively_feasible": bool(feasible), "feasible_portfolios": len(feasible),
        "witness": feasible[0] if feasible else None, "solver_matches_proof": matched,
        "solver_result": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mip-relative-gap", type=float, default=0.)
    parser.add_argument("--feasibility-only", action="store_true")
    parser.add_argument("--time-limit", type=float, default=10.)
    parser.add_argument("--full-pool-budget", type=float, choices=[0., .005, .01])
    parser.add_argument("--staged-warm-start", action="store_true")
    parser.add_argument("--target-time-limit", type=float)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if not 0 < args.time_limit <= 60:
        raise ValueError("bounded diagnostic requires 0 < seconds <=60")
    if args.output.exists():
        raise FileExistsError("immutable evidence")
    args.output.mkdir(parents=True)
    # Budgets are fixed before observing results; none is auto-selected or
    # used to retry/relax the previously failed production-like contract.
    report = {"status": "RUNNING_RESEARCH_ONLY", "orders_enabled": False,
        "predeclared_unknown_budgets": [0., .005, .01], "risk_budget_selected": False,
        "requested_mip_relative_gap": args.mip_relative_gap,
        "diagnostic_feasibility_only": args.feasibility_only,
        "research_staged_warm_start": args.staged_warm_start,
        "research_target_time_limit": args.target_time_limit,
        "small_cases": [], "full_pool_cases": []}
    path = args.output / "results.json"
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for reference, budget in ((.37, 0.), (.375, 0.), (.37, .005), (.37, .01)):
        report["small_cases"].append(small_proof(reference, budget)); save()
    root = Path(__file__).resolve().parents[1]
    source = args.source or root / "state/style-balance/coverage-rounding-20261102-a"
    source_metadata = json.loads((source/"results.json").read_text(encoding="utf-8"))
    styles = pd.read_parquet(source / "estimated-controls-with-unknown-flags.parquet")
    weights = pd.read_parquet(source / "weights-0.1.parquet").reindex(styles.index)
    target = weights.weight.clip(lower=0); target /= target.sum()
    if abs(target-weights.weight).max() > 1e-8:
        raise ValueError("NON_NUMERICAL_TARGET_CHANGE")
    prices = ParquetStore(root / "data").read("daily", columns=["ts_code", "close"],
        filters=[("trade_date", "==", "20261030"), ("ts_code", "in", list(styles.index))]).set_index("ts_code").close
    report.update({"source_weights_sha256": hashlib.sha256((source/"weights-0.1.parquet").read_bytes()).hexdigest(),
        "source_initial_state": source_metadata.get("initial_state", "legacy_benchmark_mechanism"),
        "source_unknown_band": source_metadata.get("unknown_active_band", 0.),
        "source_risk_decomposition_sha256": source_metadata.get("risk_decomposition_sha256"),
        "full_pool_assets": len(styles), "capital": 1_000_000., "full_pool_time_limit_each_seconds": args.time_limit,
        "integer_support_assets": int((target > 1e-6).sum()),
        "same_continuous_target_all_cases": True,
        "limitations": ["current subset, not PIT or predictive-efficacy evidence", "fresh-cash test, not current account",
            "finite missing-exposure bands are distinct experimental contracts, not permission to relax a failed contract",
            "no real/paper orders submitted; any reference orders here are non-executable diagnostics"]})
    for budget in ([args.full_pool_budget] if args.full_pool_budget is not None else (0., .005, .01)):
        limits = {c: budget if c.startswith("unknown_") else .03 if c.startswith(("industry_", "board_")) else .10 for c in styles.columns}
        if (abs((target-weights.benchmark).to_numpy() @ styles.to_numpy()) > pd.Series(limits).reindex(styles.columns).to_numpy()+1e-6).any():
            raise ValueError("CONTINUOUS_TARGET_VIOLATES_PREDECLARED_CONTRACT")
        start = time.monotonic()
        result = plan_integer_execution(target, prices, {}, {}, 1_000_000., 1_000_000., configuration(1_000_000., .15),
            styles, limits, turnover=.20, time_limit=args.time_limit, solver_threads=2, benchmark_weights=weights.benchmark,
            style_controls_prestandardized=True, mip_relative_gap=args.mip_relative_gap,
            diagnostic_feasibility_only=args.feasibility_only, research_staged_warm_start=args.staged_warm_start,
            research_target_time_limit=args.target_time_limit)
        row = {"unknown_active_budget": budget, "elapsed_seconds": time.monotonic()-start, "result": result}
        report["full_pool_cases"].append(row); save()
        print(json.dumps({"budget": budget, "status": result["status"], "reason": result.get("reason"), "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
    report["status"] = "COMPLETED_GRID_DIAGNOSIS_NOT_RELEASE"; save()


if __name__ == "__main__":
    main()
