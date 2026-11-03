import numpy as np
import pandas as pd
import pytest
from moneymore.config import BacktestConfig
from moneymore.mature_multifactor_runtime import mature_candidates,is_mature,reprice_reference_plan
from moneymore.research.style_v2_experiment import candidate_execution_config


def config(slippage=2):
    return BacktestConfig(initial_cash=1_000_000,commission_rate=.00015,minimum_commission=5,
        stamp_duty_rate=.0005,transfer_fee_rate=.00001,slippage_bps=slippage,lot_size=100,
        max_position_weight=.15,max_gross_exposure=1.,max_drawdown=.15)


def test_mature_capital_and_cost_do_not_modify_shared_default():
    original=config()
    candidate={'portfolio_variant':'mature_alpha158_balanced','initial_cash':10_000_000,
        'execution_policy':{'version':'mature_v1','real_orders_enabled':False,'slippage_bps':10}}
    updated=candidate_execution_config(original,candidate)
    assert updated.initial_cash==10_000_000 and updated.slippage_bps==10
    assert original.initial_cash==1_000_000 and original.slippage_bps==2
    assert is_mature(candidate)


def test_absent_registry_does_not_create_old_or_new_accounts(tmp_path):
    assert mature_candidates(tmp_path)==[]
    assert not (tmp_path/'state/paper_orders.sqlite3').exists()


def reprice_inputs():
    names=[str(i) for i in range(10)]
    plan={'status':'PASS_REFERENCE_QUOTES_ONLY','target_quantities':dict.fromkeys(names,900),
        'orders':[{'symbol':s,'side':'BUY','quantity':900} for s in names]}
    styles=pd.DataFrame({'execution_only':0.},index=names)
    return plan,dict.fromkeys(names,100.),styles,dict.fromkeys(names,.1),dict.fromkeys(names,True)


def test_quote_after_solver_rechecks_prices_fees_and_cash():
    plan,prices,styles,reference,permissions=reprice_inputs()
    execution_config=config(10)
    updated=reprice_reference_plan(plan,prices,{},1_000_000,execution_config,styles,{'execution_only':0.},reference,permissions,permissions)
    assert updated['fresh_quote_revalidated'] and updated['estimated_cash']>0
    assert updated['orders'][0]['estimated_execution_price']==pytest.approx(100.1)
    prices['0']=np.nan
    with pytest.raises(ValueError,match='INVALID_FRESH'):
        reprice_reference_plan(plan,prices,{},1_000_000,execution_config,styles,{'execution_only':0.},reference,permissions,permissions)


def test_post_solver_quote_change_cannot_overspend_or_relax_risk():
    plan,prices,styles,reference,permissions=reprice_inputs();prices=dict.fromkeys(prices,200.)
    with pytest.raises(ValueError,match='FRESH_QUOTE_CASH_RISK'):
        reprice_reference_plan(plan,prices,{},1_000_000,config(),styles,{'execution_only':0.},reference,permissions,permissions)
