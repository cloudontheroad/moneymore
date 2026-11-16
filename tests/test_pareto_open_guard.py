import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_ashare_pareto_qlib import exact_open_guard
from moneymore.config import BacktestConfig


def example():
    names=[f'600{i:03d}.SH' for i in range(10)]
    desired=dict.fromkeys(names,1000)
    snap={'cash':1_000_000.,'equity':1_000_000.,'positions':[],'unsettled_stock_rights':{},'receivable_cash':0.}
    controls=pd.DataFrame({'factor':0.},index=names)
    config=BacktestConfig.from_yaml(Path(__file__).resolve().parents[1]/'configs/default.yaml')
    return names,desired,snap,controls,config.model_copy(update={'initial_cash':1_000_000,'slippage_bps':10,'max_position_weight':.15})


def test_cash_clipping_keeps_postcheck_and_does_not_add_stocks():
    names,desired,snap,styles,config=example()
    result,reason=exact_open_guard(desired,{},snap,dict.fromkeys(names,100.),dict.fromkeys(names,True),{},config,styles,{'factor':.1},pd.Series(.1,index=names))
    assert reason is None and set(result)<=set(desired)
    assert sum(result.values())<sum(desired.values())
    assert all(0<=q<=desired[s] and q%100==0 for s,q in result.items())


def test_cash_clipping_cannot_relax_style_budget():
    names,desired,snap,styles,config=example();styles.loc[names[-1],'factor']=1.
    result,reason=exact_open_guard(desired,{},snap,dict.fromkeys(names,100.),dict.fromkeys(names,True),{},config,styles,{'factor':.001},pd.Series(.1,index=names))
    assert reason=='OPEN_STYLE_POSTCHECK' and result=={}


def test_missing_permission_stops_plan():
    names,desired,snap,styles,config=example()
    result,reason=exact_open_guard(desired,{},snap,dict.fromkeys(names,100.),{}, {},config,styles,{'factor':.1},pd.Series(.1,index=names))
    assert reason=='OPEN_PRICE_OR_PERMISSION' and result=={}
