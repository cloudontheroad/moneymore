"""As-of inputs and distinct entry/rebalance constraints for isolated replay."""
import numpy as np
import pandas as pd
import cvxpy as cp
from qlib.model.riskmodel import StructuredCovEstimator

from moneymore.research.baseline_v2_financials import common_period_features, required_period
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.style_risk_inputs import signed_inverse


def prospective_capture(signal_date, created_at, execution_date):
    captured = pd.to_datetime(created_at, utc=True, errors="coerce")
    return bool(not pd.isna(captured) and signal_date < execution_date
        and pd.Timestamp(signal_date+" 15:00", tz="Asia/Shanghai").tz_convert("UTC") <= captured
        < pd.Timestamp(execution_date+" 09:30", tz="Asia/Shanghai").tz_convert("UTC"))


def measured_style(weights, controls, F, covariance, specific, wb):
    """Stock-sleeve active controls and statistical risk, not named attribution."""
    weights = weights.reindex(controls.index, fill_value=0).astype(float)
    if weights.sum() <= 0:
        return {"invested": False, "active_controls": None, "active_tracking_volatility": None}
    weights /= weights.sum()
    active = (weights-wb).to_numpy()
    factor_variance = float((active @ F) @ covariance @ (active @ F))
    specific_variance = float(specific @ (active*active))
    contribution = active*(F @ covariance @ (active @ F)+specific*active)
    if not np.isfinite(contribution).all() or factor_variance+specific_variance < -1e-10:
        raise ValueError("NONFINITE_ACTIVE_RISK")
    return {"invested": True, "active_controls": ((weights-wb) @ controls).to_dict(),
        "active_tracking_volatility": float(np.sqrt(max(0., factor_variance+specific_variance)*252)),
        "statistical_factor_variance": factor_variance, "specific_variance": specific_variance,
        "asset_active_variance_contributions": dict(zip(controls.index, map(float, contribution))),
        "not_named_factor_return_attribution": True}


def daily_risk_snapshot(date, scores, held, calendar, daily, basic, finances, membership, *, estimate_covariance=True):
    symbols = sorted(set(scores.index) | set(held))
    days = [d for d in calendar if d <= date][-121:]
    history = daily.loc[daily.trade_date.isin(days) & daily.ts_code.isin(symbols)]
    prices = history.pivot(index="trade_date", columns="ts_code", values="total_price").reindex(index=days, columns=symbols)
    current = basic.loc[basic.trade_date.eq(date)].set_index("ts_code").reindex(symbols)
    raw, _ = named_price_risk_exposures(prices, current.total_mv)
    raw["book_to_price"] = signed_inverse(current.pb)
    raw["earnings_yield"] = signed_inverse(current.pe_ttm)
    raw["dividend_yield"] = current.dv_ttm/100
    raw["turnover_rate"] = current.turnover_rate/100
    financial = common_period_features(finances, date).set_index("ts_code")
    for field, name in (("roe", "quality_roe"), ("q_sales_yoy", "growth_sales_yoy")):
        raw[name] = financial[field].where(financial[field+"_end_date"].astype(str).eq(required_period(date))).reindex(symbols)/100
    amount = history.pivot(index="trade_date", columns="ts_code", values="amount").reindex(index=days[-20:], columns=symbols)
    raw["log_amount20"] = np.log(amount.mean().where(amount.notna().sum().eq(20)).where(lambda x: x > 0)*1000)
    raw["industry"] = raw.index.map(industry_as_of(membership, date).l1_name)
    raw["listing_board"] = ["STAR" if s.startswith("68") else "CHINEXT" if s.startswith("30") else "MAIN" for s in raw.index]
    raw = raw.replace([np.inf, -np.inf], np.nan)
    core = ["log_size", "momentum60", "volatility20", "market_beta120", "residual_volatility120"]
    controls, eligible, missing = uncertain_controls(raw, core)
    if set(held)-set(controls.index):
        raise ValueError("HELD_RISK_DATA_MISSING_NOT_A_POOL_EXIT:"+",".join(sorted(set(held)-set(controls.index))))
    selected = raw.loc[eligible]
    returns = prices.pct_change(fill_method=None).iloc[1:].reindex(columns=selected.index)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("MISSING_RISK_HISTORY")
    F = covariance = specific = None
    if estimate_covariance:
        np.random.seed(17)
        F, covariance, specific = StructuredCovEstimator(num_factors=10, factor_model="pca", scale_return=False).predict(returns, is_price=False, return_decomposed_components=True)
    wb = np.exp(selected.log_size).where(selected.index.isin(scores.index), 0.)
    wb /= wb.sum()
    limits = pd.Series({c: .01 if c.startswith("unknown_") else .03 if c.startswith(("industry_", "board_")) else .10 for c in controls.columns})
    return selected, controls, F, covariance, specific, wb, limits, missing


def optimize_daily(scores, selected, controls, F, covariance, specific, wb, limits, held_values, *, lower_limits=None, upper_limits=None, research_quality_envelope=None, candidate_symbols=None):
    observed = scores.reindex(selected.index)
    if candidate_symbols is None:
        exits = ~selected.index.isin(scores.index)
    else:
        from .holding_eligibility import explicit_score_membership
        exits = explicit_score_membership(selected.index, scores, candidate_symbols)
    if observed.loc[~exits].isna().any():
        raise ValueError("MISSING_SCORES_NOT_A_POOL_EXIT")
    score_scale=observed.loc[~exits].std()
    if not np.isfinite(score_scale) or score_scale<=0:
        raise ValueError('NONINFORMATIVE_SCORE_SCALE')
    calibrated = observed.where(~exits, 0.).fillna(0.).to_numpy()/observed.loc[~exits].to_numpy().std() * np.sqrt(np.mean(np.sum((F @ covariance)*F, axis=1)+specific))
    weights = cp.Variable(len(selected)); active = weights-wb.to_numpy()
    C = controls.to_numpy()
    if research_quality_envelope is not None:
        if lower_limits is not None or upper_limits is not None:
            raise ValueError("QUALITY_CONTRACT_CANNOT_MIX_LEGACY_OVERRIDE")
        from .quality_contract import quality_upper_only_bounds
        lower,upper=quality_upper_only_bounds(controls,wb,limits,research_quality_envelope)
        if (lower>upper).any():raise ValueError("QUALITY_CONTRACT_INFEASIBLE_INTERVAL")
    elif (lower_limits is None) != (upper_limits is None):
        raise ValueError("BOTH_ASYMMETRIC_LIMITS_REQUIRED")
    else:
        lower = -limits.reindex(controls.columns) if lower_limits is None else lower_limits.reindex(controls.columns)
        upper = limits.reindex(controls.columns) if upper_limits is None else upper_limits.reindex(controls.columns)
    if research_quality_envelope is None and (not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower>upper).any()
            or (lower < -limits.reindex(controls.columns)-1e-12).any() or (upper > limits.reindex(controls.columns)+1e-12).any()):
        raise ValueError("ASYMMETRIC_LIMITS_MISSING_INFEASIBLE_OR_RELAXED")
    constraints = [weights >= 0, weights <= .15, cp.sum(weights) == 1,
        C.T @ active <= upper.to_numpy(), C.T @ active >= lower.to_numpy()]
    if exits.any():
        constraints.append(weights[exits] == 0)
    if held_values.sum() > 0:
        w0 = held_values.reindex(selected.index, fill_value=0)/held_values.sum()
        constraints.append(cp.norm1(weights-w0.to_numpy()) <= .20)
    problem = cp.Problem(cp.Maximize(active @ calibrated-cp.quad_form(active @ F, cp.psd_wrap(covariance))-cp.sum(cp.multiply(specific, cp.square(active)))), constraints)
    problem.solve(solver=cp.CLARABEL, max_iter=300, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    if problem.status != "optimal":
        raise ValueError("CONTINUOUS_SOLVER_"+problem.status.upper())
    raw_weights = pd.Series(weights.value, index=selected.index)
    target = raw_weights.clip(lower=0); target /= target.sum()
    repair = float(abs(target-raw_weights).max())
    exposure = (target-wb).to_numpy() @ C
    excess = float(max(np.max(exposure-upper.to_numpy()), np.max(lower.to_numpy()-exposure)))
    if repair > 1e-8 or target.max() > .15+1e-6 or excess > 1e-6:
        raise ValueError(f"CONTINUOUS_NUMERICAL_REPAIR_OR_CONSTRAINT_BREACH:repair={repair:.9g},excess={excess:.9g}")
    if held_values.sum() > 0 and abs(target-w0).sum() > .20+1e-6:
        raise ValueError("ACTUAL_HOLDING_TURNOVER_BREACH")
    return target
