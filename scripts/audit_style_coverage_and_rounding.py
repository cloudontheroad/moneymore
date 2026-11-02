"""Bounded isolated coverage/risk-band/lot-size diagnostics. No orders."""
import argparse
import hashlib
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from qlib.model.riskmodel import StructuredCovEstimator

from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore
from moneymore.research.style_missing_controls import uncertain_controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-state", choices=["benchmark", "cold_cash"], default="benchmark")
    parser.add_argument("--unknown-band", type=float, choices=[0., .005, .01], default=0.)
    parser.add_argument("--standard-band", type=float, choices=[0., .05, .10, .20])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable evidence")
    root = Path(__file__).resolve().parents[1]
    source = root / "state/style-balance/named-risk-20261102-b"
    raw = pd.read_parquet(source / "raw-exposures.parquet")
    core = ["log_size", "momentum60", "volatility20", "market_beta120", "residual_volatility120"]
    controls, eligible, missing_info = uncertain_controls(raw, core)
    selected = raw.loc[eligible]
    old = pd.read_parquet(source / "named-controls.parquet").index
    store = ParquetStore(root / "data")
    as_of = "20261030"
    days = [d for d in trading_dates(store, as_of) if d <= as_of][-121:]
    filters = [("ts_code", "in", list(selected.index)), ("trade_date", ">=", days[0]), ("trade_date", "<=", as_of)]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    bars = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    bars["total_price"] = bars.close * bars.adj_factor
    returns = bars.pivot(index="trade_date", columns="ts_code", values="total_price").reindex(days).pct_change(fill_method=None).iloc[1:].reindex(columns=selected.index)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("MISSING_RETURNS_CANNOT_BE_IMPUTED")
    np.random.seed(17)
    F, covariance, specific = StructuredCovEstimator(num_factors=10, factor_model="pca", scale_return=False).predict(returns, is_price=False, return_decomposed_components=True)
    wb = np.exp(selected.log_size.to_numpy()); wb /= wb.sum()
    score_path = root / "state/qlib-challenger/candidates/20261031/latest-refit/current-score-comparison.parquet"
    scores = pd.read_parquet(score_path).latest_refit_score
    scores.index = scores.index.get_level_values("instrument")
    scores = scores.reindex(selected.index)
    if scores.isna().any() or scores.std() <= 0:
        raise ValueError("NO_SCORES_FOR_REQUIRED_RISK_UNIVERSE")
    calibrated = scores.to_numpy()/scores.to_numpy().std() * np.sqrt(np.mean(np.sum((F @ covariance)*F, axis=1)+specific))
    C = controls.to_numpy()
    tech = selected.industry.isin(["电子", "计算机", "通信"]).to_numpy()
    growth = selected.listing_board.isin(["STAR", "CHINEXT"]).to_numpy()
    close = daily[daily.trade_date.eq(as_of)].set_index("ts_code").close.reindex(selected.index).to_numpy()
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("NO_SAME_DATE_REFERENCE_PRICES")
    # Protocol fixed before this run: zero then three nested budgets, never
    # selected by this snapshot's technology share or nonexistent returns.
    bands = [(0., 0.), (.05, .01), (.10, .03), (.20, .05)]
    if args.standard_band is not None:
        bands = [b for b in bands if b[0] == args.standard_band]
    args.output.mkdir(parents=True)
    controls.to_parquet(args.output / "estimated-controls-with-unknown-flags.parquet")
    selected.to_parquet(args.output / "raw-observations-unfilled.parquet")
    rows = []
    for standard_band, category_band in bands:
        limits = np.array([args.unknown_band if c.startswith("unknown_") else category_band if c.startswith(("industry_", "board_")) else standard_band for c in controls.columns])
        weights = cp.Variable(len(wb)); active = weights-wb
        constraints = [weights >= 0, weights <= .15, cp.sum(weights) == 1,
            C.T @ active <= limits, C.T @ active >= -limits]
        if args.initial_state == "benchmark":
            constraints.append(cp.norm1(active) <= .20)
        problem = cp.Problem(cp.Maximize(active @ calibrated-cp.quad_form(active @ F, cp.psd_wrap(covariance))-cp.sum(cp.multiply(specific, cp.square(active)))), constraints)
        problem.solve(solver=cp.CLARABEL, max_iter=300)
        row = {"standardized_band": standard_band, "category_band": category_band, "solver_status": problem.status}
        if problem.status != "optimal":
            rows.append(row); continue
        w = weights.value
        excess = np.max(np.abs((w-wb) @ C)-limits)
        if excess > 1e-6 or w.min() < -1e-6 or w.max() > .15+1e-6 or abs(w.sum()-1) > 1e-6 or (args.initial_state == "benchmark" and abs(w-wb).sum() > .20+1e-6):
            raise ValueError("CONTINUOUS_CONSTRAINT_BREACH")
        row.update({"effective_holdings": float(1/(w @ w)), "growth_weight": float(w @ growth), "technology_weight": float(w @ tech),
            "max_constraint_excess": float(excess), "turnover_from_test_benchmark_l1": float(abs(w-wb).sum()), "rounding": []})
        pd.DataFrame({"weight": w, "benchmark": wb}, index=selected.index).to_parquet(args.output / f"weights-{standard_band}.parquet")
        for capital in (100_000., 1_000_000.):
            # Conservative 100-share increments; STAR initial order >=200.
            quantity = np.floor(np.maximum(w, 0)*capital/close/100)*100
            star = selected.index.str.startswith(("688", "689"))
            quantity[star & (quantity < 200)] = 0
            value = quantity*close
            notional = value*1.001
            fees = np.where(quantity > 0, np.maximum(5., notional*.00015)+notional*.00001, 0.)
            cash = capital-notional.sum()-fees.sum()
            stock_weight = value/value.sum() if value.sum() else np.zeros(len(w))
            style_excess = np.abs((stock_weight-wb) @ C)-limits
            row["rounding"].append({"capital": capital, "method": "FLOOR_ONLY_NOT_INTEGER_OPTIMIZER_OR_TRADABLE_PLAN",
                "cash_after_cost": float(cash), "fees": float(fees.sum()), "slippage_cost": float(value.sum()*.001),
                "gross_at_reference": float(value.sum()/capital), "holdings": int((quantity > 0).sum()),
                "style_pass": bool(style_excess.max() <= 1e-6), "max_style_excess": float(style_excess.max()),
                "violated_controls": controls.columns[style_excess > 1e-6].tolist(),
                "cash_pass": bool(cash >= 0), "not_trade_authorization": True})
        rows.append(row)
    old_cap = np.exp(raw.loc[old].log_size)
    old_w = old_cap/old_cap.sum()
    report = {"status": "EXPLORATORY_COVERAGE_AND_ROUNDING_NOT_RELEASE", "orders_enabled": False, "as_of": as_of,
        "requested": len(raw), "old_complete": len(old), "expanded_core": len(selected), "required_still_missing": int((~eligible).sum()),
        "newly_included": sorted(set(selected.index)-set(old)), "optional_estimation": missing_info,
        "old_growth_count_share": float(raw.loc[old].listing_board.isin(["STAR", "CHINEXT"]).mean()),
        "expanded_growth_count_share": float(growth.mean()), "benchmark_growth_weight": float(wb @ growth),
        "old_benchmark_growth_weight": float(old_w @ raw.loc[old].listing_board.isin(["STAR", "CHINEXT"])),
        "benchmark_technology_weight": float(wb @ tech), "risk_sessions": len(returns), "cases": rows,
        "risk_budget_selected": False, "benchmark_initial_position_is_synthetic": args.initial_state == "benchmark",
        "initial_state": args.initial_state, "unknown_active_band": args.unknown_band,
        "risk_random_seed": 17,
        "risk_decomposition_sha256": hashlib.sha256(F.tobytes()+covariance.tobytes()+specific.tobytes()).hexdigest(),
        "source_raw_sha256": hashlib.sha256((source/"raw-exposures.parquet").read_bytes()).hexdigest(),
        "source_scores_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "limitations": ["not full1000/PIT/cost-return replay", "median estimates are uncertain, not financial observations",
            "unknown-field active exposures held to zero, not proof unknown values harmless", "no suspension/limit/quote-age or sell-side/T+1 acceptance",
            "floor rounding is diagnostic only, not an executable plan; fees are specified test assumptions, not broker tariff validation",
            "all stock capital is hypothetical; no orders or historical returns changed"]}
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"expanded": len(selected), "old": len(old), "solves": len(rows), "rounding_cases": sum(len(r.get('rounding', [])) for r in rows)}))


if __name__ == "__main__":
    main()
