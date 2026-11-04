import numpy as np
import pandas as pd
import pytest
from moneymore.config import BacktestConfig
from moneymore.research.cost_aware import calibrated_returns,utility_change,risk_compliant
from moneymore.research.cost_execution import holding_plan,finalize_cost_decision
from moneymore.research.cost_integer import plan_integer_execution

def test_calibration_is_not_rank_probability_or_invented_alpha():
    scores=pd.Series([1.,2.,3.],index=['a','b','c'])
    assert calibrated_returns(scores,{'schema':'COST_CALIBRATION_V1','beta_conservative':0.}).eq(0).all()
    result=calibrated_returns(scores,{'schema':'COST_CALIBRATION_V1','beta_conservative':.001})
    assert result.mean()==pytest.approx(0) and result.std(ddof=0)==pytest.approx(.001)
    with pytest.raises(ValueError):calibrated_returns(pd.Series([1,1]),{'schema':'COST_CALIBRATION_V1'})

def test_net_value_compares_increment_not_absolute_high_score():
    result=utility_change({'a':1.},{'b':1.},{'a':.01,'b':.011},None,{},.002)
    assert result['net_utility_increment']==pytest.approx(-.001)

def test_economic_integer_objective_does_not_rebalance_a_tiny_drift():
    symbols=[f'{600000+i}.SH' for i in range(10)]
    config=BacktestConfig(initial_cash=10000,commission_rate=.00015,minimum_commission=5,stamp_duty_rate=.0005,transfer_fee_rate=.00001,
        slippage_bps=10,lot_size=100,max_position_weight=.15,max_gross_exposure=1,max_drawdown=.15)
    styles=pd.DataFrame({'none':0.},index=symbols);weights={s:.1 for s in symbols};weights[symbols[0]]+=.0001;weights[symbols[1]]-=.0001
    plan=plan_integer_execution(weights,{s:10 for s in symbols},{s:100 for s in symbols},{s:100 for s in symbols},0,10000,config,styles,{'none':0},
        economic_tracking_weights={s:.0001 for s in symbols},time_limit=5,solver_threads=4,accept_validated_incumbent=True)
    assert plan['status']=='PASS_REFERENCE_QUOTES_ONLY' and plan['orders']==[]
    assert plan['estimated_cash']==pytest.approx(0)

def test_no_trade_keeps_actual_shares_and_has_explicit_reason():
    styles=pd.DataFrame({'none':[0.,0.]},index=['a','b']);benchmark={'a':.5,'b':.5}
    old={'a':50,'b':50};prices={'a':10.,'b':10.}
    plan={'status':'PASS_REFERENCE_QUOTES_ONLY','target_quantities':{'a':40,'b':60},'estimated_fees':10.,
        'orders':[{'symbol':'a','quantity':10,'estimated_execution_price':9.99},{'symbol':'b','quantity':10,'estimated_execution_price':10.01}]}
    # Use 100% cap to isolate the utility gate in this small fixture.
    from unittest.mock import patch
    with patch('moneymore.research.cost_execution.risk_compliant',return_value=True):
        result=finalize_cost_decision(plan,prices,old,100,1100,styles,{'none':0},benchmark,{'mu':{'a':0.,'b':0.},'risk_model':None})
    assert result['cost_decision']=='COST_NO_TRADE' and result['target_quantities']==old and not result['orders']

def test_holding_is_not_a_valid_bootstrap():
    with pytest.raises(ValueError):holding_plan({}, {},100,100,pd.DataFrame(),{}, {})

def test_zero_alpha_cannot_keep_an_exited_member_by_optimizing_only_fees():
    symbols=[f'{600000+i}.SH' for i in range(11)]
    config=BacktestConfig(initial_cash=11000,commission_rate=.00015,minimum_commission=5,stamp_duty_rate=.0005,transfer_fee_rate=.00001,
        slippage_bps=10,lot_size=100,max_position_weight=.15,max_gross_exposure=1,max_drawdown=.15)
    held=dict.fromkeys(symbols[:10],100);weights=dict.fromkeys(symbols[1:],.1)
    result=plan_integer_execution(weights,dict.fromkeys(symbols,10),held,held,1000,11000,config,pd.DataFrame({'none':0.},index=symbols),{'none':0},
        economic_tracking_weights=dict.fromkeys(symbols,0.),forced_sell_symbols=[symbols[0]],accept_validated_incumbent=True,time_limit=5,solver_threads=4)
    assert result['status']=='PASS_REFERENCE_QUOTES_ONLY',result
    assert result['target_quantities'][symbols[0]]==0 and any(o['side']=='SELL' and o['symbol']==symbols[0] for o in result['orders'])

def test_risk_window_start_only_uses_contiguous_proven_halt_anchor():
    from moneymore.research.cost_risk import seed_confirmed_window_start
    dates=pd.bdate_range('20260101',periods=122).strftime('%Y%m%d').tolist();calendar=dates[1:]
    bars=pd.DataFrame([{'ts_code':'a','trade_date':d,'close':10.,'adj_factor':2.} for d in [dates[0],*calendar[3:]]])
    proof=pd.DataFrame([{'ts_code':'a','trade_date':d,'suspend_type':'S','suspend_timing':None} for d in calendar[:3]])
    seeded,added=seed_confirmed_window_start(bars,{'a'},calendar,proof,dates)
    assert len(added)==3 and all(r['close']*r['adj_factor']==20 for r in added)
    assert not any('open' in r for r in added) and len(bars)==119
    partial=proof.iloc[:2]
    _,added=seed_confirmed_window_start(bars,{'a'},calendar,partial,dates);assert len(added)==2
    _,added=seed_confirmed_window_start(bars,{'a'},calendar,proof,calendar);assert not added
    shifted=[*calendar[1:],pd.bdate_range(calendar[-1],periods=2)[-1].strftime('%Y%m%d')]
    _,added=seed_confirmed_window_start(bars,{'a'},shifted,proof,[*dates,shifted[-1]]);assert len(added)==2
    _,added=seed_confirmed_window_start(bars,{'a'},shifted,proof.iloc[1:],[*dates,shifted[-1]]);assert not added

def test_optional_trade_must_beat_its_exact_cost_not_just_have_a_high_score():
    symbols=[str(i) for i in range(15)];held=dict.fromkeys(symbols,100);prices=dict.fromkeys(symbols,1.)
    quantities={**held,'0':0,'1':200};styles=pd.DataFrame({'none':0.},index=symbols)
    plan={'status':'PASS_REFERENCE_QUOTES_ONLY','target_quantities':quantities,'estimated_fees':2.,
        'orders':[{'symbol':'0','side':'SELL','quantity':100,'estimated_execution_price':1.},{'symbol':'1','side':'BUY','quantity':100,'estimated_execution_price':1.}]}
    economics={'mu':{**dict.fromkeys(symbols,0.),'1':.05},'risk_model':None,'alpha_enabled':True}
    assert finalize_cost_decision(plan,prices,held,0,1500,styles,{'none':0.},None,economics)['cost_decision']=='POSITIVE_NET_UTILITY'
    plan['estimated_fees']=20.
    assert finalize_cost_decision(plan,prices,held,0,1500,styles,{'none':0.},None,economics)['cost_decision']=='COST_NO_TRADE'

def test_calibrated_edge_and_friction_use_same_nav_unit_with_cash():
    plan={'status':'PASS_REFERENCE_QUOTES_ONLY','target_quantities':{'a':40,'b':60},'estimated_fees':.5,
        'orders':[{'symbol':'a','quantity':10,'estimated_execution_price':1.},{'symbol':'b','quantity':10,'estimated_execution_price':1.}]}
    result=finalize_cost_decision(plan,{'a':1.,'b':1.},{'a':50,'b':50},900,1000,pd.DataFrame({'none':[0.,0.]},index=['a','b']),{'none':0},None,
        {'mu':{'a':0.,'b':.1},'risk_model':None,'alpha_enabled':True})
    assert result['candidate_net_utility']['expected_return_increment']==pytest.approx(.001)
    assert result['candidate_net_utility']['friction_fraction']==pytest.approx(.0005)
