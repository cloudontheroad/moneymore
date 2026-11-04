"""Seed a rolling risk window from prior *proven* suspension anchors only."""
import numpy as np
import pandas as pd
from .mature_multifactor import current_risk_inputs
from moneymore.challenger_diagnostics import trading_dates

def seed_confirmed_window_start(bars,symbols,calendar,proof,all_calendar):
    if len(calendar)!=121:raise ValueError('INCOMPLETE_COST_RISK_CALENDAR')
    eligible=proof.loc[proof.suspend_type.eq('S')&(proof.suspend_timing.isna()|proof.suspend_timing.eq(''))] if not proof.empty else proof
    confirmed=set(zip(eligible.ts_code,eligible.trade_date.astype(str))) if not eligible.empty else set()
    frame=bars.loc[bars.ts_code.isin(symbols)&bars.trade_date.astype(str).le(calendar[-1])].copy()
    frame['day']=frame.trade_date.astype(str);added=[]
    for symbol,g in frame.groupby('ts_code'):
        observed=set(g.loc[g.close.notna()&g.adj_factor.notna(),'day'])
        if calendar[0] in observed:continue
        prior=g.loc[g.day.lt(calendar[0])].sort_values('day')
        if prior.empty or prior.iloc[-1].day not in all_calendar:continue
        chain=[d for d in all_calendar if prior.iloc[-1].day<d<calendar[0]]
        if any((symbol,d) not in confirmed for d in chain):continue
        anchor=float(prior.iloc[-1].close*prior.iloc[-1].adj_factor)
        if not np.isfinite(anchor) or anchor<=0:continue
        for day in calendar:
            if day in observed:break
            if (symbol,day) not in confirmed:break
            added.append({'ts_code':symbol,'trade_date':day,'close':anchor,'adj_factor':1.,'risk_only_confirmed_halt_seed':True,
                'source_anchor_date':prior.iloc[-1].day,'prior_halt_chain':','.join(chain)})
    if not added:return bars,[]
    # Risk-only adjusted-value anchors: never export these as execution OHLC or forecasts.
    overlay=pd.DataFrame(added)
    result=pd.concat([bars,overlay],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
    return result,added

def cost_risk_inputs(store,bars,scores,day,**kwargs):
    proof=kwargs.get('suspension_proof',pd.DataFrame())
    try:historical=store.read('suspend_d')
    except FileNotFoundError:historical=pd.DataFrame()
    if not historical.empty:proof=pd.concat([proof,historical],ignore_index=True).drop_duplicates()
    if not proof.empty:proof=proof.loc[proof.trade_date.astype(str).le(day)]
    all_dates=[d for d in trading_dates(store,day) if d<=day];calendar=all_dates[-121:]
    seeded,rows=seed_confirmed_window_start(bars,set(scores.index),calendar,proof,all_dates)
    kwargs['suspension_proof']=proof
    result=current_risk_inputs(store,seeded,scores,day,**kwargs)
    result[-1]['cost_window_start_proven_halt_seeds']=rows
    result[-1]['risk_seed_is_not_execution_quote_or_forecast']=True
    return result
