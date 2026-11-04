"""Same frozen predictor, explicit calibration mode and incremental targets."""
import pandas as pd
from .cost_aware import calibrated_returns,optimize_cost_target
from .mature_multifactor import topk_dropout_targets

def cost_targets(scores,values,equity,returns,styles,limits,benchmark,config,calibration,kind,exits=(),frozen=(),size=None):
    mu=calibrated_returns(scores,calibration)
    values=pd.Series(values,dtype=float)
    current=values/values.sum() if values.sum()>0 else values
    if values.empty and kind!='raw':
        from .enhanced_indexing import optimize_snapshot,EnhancedPolicy
        if size is None:raise ValueError('INITIAL_CAPITALIZATION_REQUIRED')
        initial=optimize_snapshot(scores.reindex(returns.columns),returns,size,values,
            EnhancedPolicy(objective_preconditioning=True),style_controls=styles,style_limits=limits,benchmark_weights=benchmark,
            controls_prestandardized=True,allow_cash_bootstrap=True,reject_turnover_relaxation=True,solver_max_iters=1000)
        if initial['status']!='PASS_OFFLINE_ONLY':return {'status':'INITIAL_CONFIGURATION_BLOCKED','reason':initial.get('failures')}
        return {'status':'PASS_COST_CONTINUOUS','weights':initial['weights'],'risk_model':None,'mu':mu.to_dict(),
            'economic_tracking_weights':mu.abs().to_dict(),'alpha_enabled':calibration['optional_alpha_enabled'],'mode':'SHARED_INITIAL_CONFIGURATION'}
    if kind=='raw':
        protected=values.reindex(sorted(set(frozen)-set(exits))).dropna()/equity
        free=1-float(protected.sum())
        eligible=scores.drop(index=list(set(exits)|set(protected.index)),errors='ignore')
        current_absolute=values.drop(index=list(set(exits)|set(protected.index)),errors='ignore')/equity/free
        if calibration['optional_alpha_enabled'] or current_absolute.empty:
            selected=topk_dropout_targets(eligible,current_absolute,topk=50-len(protected),n_drop=5)
        else:
            selected=current_absolute.to_dict()
            additions=eligible.loc[~eligible.index.isin(current_absolute.index)].sort_values(ascending=False).head(max(0,50-len(protected)-len(current_absolute)))
            if len(additions):selected.update({s:max(0.,1-current_absolute.sum())/len(additions) for s in additions.index})
        targets={s:w*free for s,w in selected.items()}
        targets.update(protected.to_dict())
        return {'status':'PASS_COST_CONTINUOUS','weights':targets,'risk_model':None,'mu':mu.to_dict(),
            'economic_tracking_weights':mu.abs().to_dict(),'alpha_enabled':calibration['optional_alpha_enabled'],
            'mode':'COST_AWARE_ALPHA' if calibration['optional_alpha_enabled'] else 'COST_PROTECTION_NO_RELIABLE_ALPHA',
            'preserve_quantities':not calibration['optional_alpha_enabled'] and bool(values.sum()) and not set(exits)}
    # A missing forecast is a frozen dimension, not a zero-valued sell signal.
    risk_mu=mu.reindex(returns.columns,fill_value=0.)
    result=optimize_cost_target(risk_mu,returns,current,styles,limits,benchmark,config,forced_sell=exits,forced_hold=frozen)
    result.update(alpha_enabled=calibration['optional_alpha_enabled'],mode='COST_AWARE_ALPHA' if calibration['optional_alpha_enabled'] else 'RISK_AND_COST_PROTECTION_NO_RELIABLE_ALPHA')
    return result
