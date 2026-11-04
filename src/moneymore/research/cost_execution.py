"""Exact, opt-in cost decision on a physically validated integer plan."""
import numpy as np
import pandas as pd
from .cost_aware import utility_change,risk_compliant

def holding_plan(prices,held,cash,equity,styles,limits,benchmark,reason='COST_NO_TRADE',locked_shares=None):
    quantities=pd.Series(held,dtype=float).add(pd.Series(locked_shares or {},dtype=float),fill_value=0)
    values=pd.Series({s:q*prices[s] for s,q in quantities.items() if q>0},dtype=float)
    if values.empty:raise ValueError('CANNOT_HOLD_EMPTY_BOOTSTRAP')
    weights=values/values.sum();active=(weights.reindex(styles.index,fill_value=0)-pd.Series(benchmark).reindex(styles.index,fill_value=0))@styles
    return {'status':'PASS_REFERENCE_QUOTES_ONLY','orders_enabled':False,'orders':[],'target_quantities':{s:int(q) for s,q in held.items() if q>0},
        'estimated_cash':cash,'estimated_fees':0.,'estimated_gross':float(values.sum()/equity),'turnover_l1':0.,
        'active_style_exposures':active.to_dict(),'cost_decision':reason,'fresh_quote_revalidated':False}

def finalize_cost_decision(plan,prices,held,cash,equity,styles,limits,benchmark,economics,forced_exit=(),locked_shares=None):
    if plan.get('status')!='PASS_REFERENCE_QUOTES_ONLY':return plan
    if not held:
        return {**plan,'cost_decision':'INITIAL_CONFIGURATION','alpha_evidence_enabled':economics.get('alpha_enabled',False)}
    before_quantities=pd.Series(held,dtype=float).add(pd.Series(locked_shares or {},dtype=float),fill_value=0)
    before=pd.Series({s:q*prices[s]/equity for s,q in before_quantities.items() if q>0},dtype=float);before_absolute=before.copy()
    quantities=pd.Series(plan['target_quantities'],dtype=float).add(pd.Series(locked_shares or {},dtype=float),fill_value=0)
    after=pd.Series({s:q*prices[s]/equity for s,q in quantities.items() if q>0},dtype=float)
    friction=float(plan['estimated_fees'])
    for order in plan['orders']:
        friction+=abs(order['estimated_execution_price']-prices[order['symbol']])*order['quantity']
    change=utility_change(before,after,economics['mu'],economics.get('risk_model'),benchmark,friction/equity)
    necessary=bool(set(forced_exit)&set(held)) or not risk_compliant(before_absolute,styles,limits,benchmark)
    if not necessary and change['net_utility_increment']<=1e-10:
        return {**holding_plan(prices,held,cash,equity,styles,limits,benchmark,locked_shares=locked_shares),
            'candidate_net_utility':change,'rejected_orders':len(plan['orders']),'alpha_evidence_enabled':economics.get('alpha_enabled',False)}
    return {**plan,'cost_decision':'NECESSARY_RISK_OR_POOL_EXIT' if necessary else 'POSITIVE_NET_UTILITY',
        'candidate_net_utility':change,'alpha_evidence_enabled':economics.get('alpha_enabled',False)}
