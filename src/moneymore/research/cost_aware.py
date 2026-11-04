"""Versioned incremental portfolio utility, not a new stock prediction model."""
import numpy as np
import pandas as pd

def calibrated_returns(scores,calibration):
    if calibration.get('schema')!='COST_CALIBRATION_V1':raise ValueError('INVALID_CALIBRATION')
    scores=pd.Series(scores,dtype=float)
    if not scores.index.is_unique or not np.isfinite(scores).all() or scores.std(ddof=0)<=1e-12:raise ValueError('INVALID_RAW_SCORE')
    return (scores-scores.mean())/scores.std(ddof=0)*float(calibration['beta_conservative'])

def transaction_rates(config):
    base=config.commission_rate+config.transfer_fee_rate+config.slippage_bps/10000
    return base,base+config.stamp_duty_rate

def five_session_risk(returns):
    from qlib.model.riskmodel import StructuredCovEstimator
    compounded=(1+returns).rolling(5).apply(np.prod,raw=True).dropna()
    if len(compounded)<30 or not np.isfinite(compounded).all().all():raise ValueError('FIVE_SESSION_RISK_INCOMPLETE')
    model=StructuredCovEstimator(factor_model='pca',num_factors=min(10,len(returns.columns)-1),scale_return=False)
    f,c,s=model.predict(compounded.copy(),is_price=False,return_decomposed_components=True)
    return {'symbols':list(returns.columns),'factor_exposure':np.asarray(f).tolist(),'factor_covariance':np.atleast_2d(c).tolist(),
        'specific_variance':np.asarray(s).tolist(),'horizon':5}

def variance(weights,risk,benchmark):
    if risk is None:return 0.
    w=pd.Series(weights).reindex(risk['symbols'],fill_value=0).to_numpy()-pd.Series(benchmark).reindex(risk['symbols'],fill_value=0).to_numpy()
    f=np.asarray(risk['factor_exposure']);c=np.asarray(risk['factor_covariance']);s=np.asarray(risk['specific_variance'])
    v=w@f
    return float(v@c@v+s@(w*w))

def utility_change(current,proposed,mu,risk,benchmark,friction_fraction):
    before=pd.Series(current,dtype=float);after=pd.Series(proposed,dtype=float);expected=pd.Series(mu,dtype=float)
    index=pd.Index(sorted(set(before.index)|set(after.index)|set(expected.index)))
    before=before.reindex(index,fill_value=0);after=after.reindex(index,fill_value=0)
    alpha=float((after-before)@expected.reindex(index,fill_value=0))
    risk_cost=variance(after,risk,benchmark)-variance(before,risk,benchmark)
    return {'expected_return_increment':alpha,'risk_cost_increment':risk_cost,'friction_fraction':float(friction_fraction),
        'net_utility_increment':alpha-risk_cost-friction_fraction}

def risk_compliant(weights,styles,limits,benchmark,maximum_weight=.15,tolerance=1e-5):
    w=pd.Series(weights,dtype=float).reindex(styles.index,fill_value=0)
    if w.sum()<=0:return False
    maximum_nav_weight=float(w.max())
    w=w/w.sum();reference=pd.Series(benchmark,dtype=float).reindex(styles.index,fill_value=0)
    stock_active_ok=(abs(w-reference)<=max(0.,maximum_weight-float(reference.max()))+tolerance).all() if abs(float(reference.sum())-1)<1e-6 else True
    return bool(np.isfinite(w).all() and w.ge(-tolerance).all() and maximum_nav_weight<=maximum_weight+tolerance
        and stock_active_ok and (abs((w-reference)@styles)<=pd.Series(limits).reindex(styles.columns)+tolerance).all())

def optimize_cost_target(mu,returns,current,styles,limits,benchmark,config,forced_sell=(),forced_hold=()):
    import cvxpy as cp
    risk=five_session_risk(returns);symbols=risk['symbols'];m=pd.Series(mu).reindex(symbols,fill_value=0).to_numpy()
    w0=pd.Series(current,dtype=float).reindex(symbols,fill_value=0).to_numpy();wb=pd.Series(benchmark).reindex(symbols).to_numpy()
    f=np.asarray(risk['factor_exposure']);c=np.asarray(risk['factor_covariance']);s=np.asarray(risk['specific_variance'])
    w=cp.Variable(len(symbols));d=w-wb;z=styles.reindex(symbols).to_numpy();bounds=pd.Series(limits).reindex(styles.columns).to_numpy()
    active_stock=max(0.,.15-float(wb.max()))
    constraints=[w>=0,w<=.15,cp.sum(w)==1,d<=active_stock,d>=-active_stock,(w-wb)@z<=bounds,(w-wb)@z>=-bounds]
    if w0.sum()>0:constraints.append(cp.norm1(w-w0)<=.20)
    for i,symbol in enumerate(symbols):
        if symbol in forced_sell:constraints.append(w[i]==0)
        if symbol in forced_hold:constraints.append(w[i]==w0[i])
    buy,sell=transaction_rates(config);cost=buy*cp.sum(cp.pos(w-w0))+sell*cp.sum(cp.pos(w0-w))
    objective=m@(w-w0)-cp.quad_form(d@f,cp.psd_wrap(c))-cp.sum_squares(cp.multiply(np.sqrt(np.maximum(s,0)),d))-cost
    problem=cp.Problem(cp.Maximize(objective*10000),constraints)
    problem.solve(solver=cp.CLARABEL,max_iter=1000,tol_gap_abs=1e-8,tol_feas=1e-9)
    if problem.status not in ['optimal','optimal_inaccurate'] or w.value is None:
        return {'status':'PROVEN_CONTINUOUS_INFEASIBLE' if problem.status.startswith('infeasible') else 'CONTINUOUS_SOLVER_FAILED','solver_status':problem.status}
    result=np.asarray(w.value).flatten();result[np.abs(result)<1e-9]=0
    if result.min()<-1e-6:raise ValueError('NEGATIVE_COST_WEIGHT')
    result=np.maximum(result,0);result/=result.sum();weights=dict(zip(symbols,map(float,result)))
    if not risk_compliant(weights,styles,limits,benchmark):raise ValueError('COST_CONTINUOUS_RISK_POSTCHECK')
    if w0.sum()>0 and abs(result-w0).sum()>.20001:raise ValueError('COST_CONTINUOUS_TURNOVER_POSTCHECK')
    gradient=m-2*(f@c@(f.T@(result-wb))+s*(result-wb))
    return {'status':'PASS_COST_CONTINUOUS','weights':weights,'risk_model':risk,'mu':dict(zip(symbols,map(float,m))),
        'economic_tracking_weights':dict(zip(symbols,map(float,np.abs(gradient)))),'solver_status':problem.status,
        'cost_contract':'expected_five_session_return_minus_active_variance_minus_one_time_transaction_cost',
        'objective':float(problem.value/10000),'turnover_l1':float(abs(result-w0).sum())}
