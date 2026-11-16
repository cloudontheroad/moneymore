import numpy as np
import pandas as pd
import pytest
from moneymore.research.pareto_rebuild import pareto_front,mature_training_mask,residualize_cross_section


def row(name,ret=.1,sharpe=1.,dd=.1,turn=2.,**kw):
    return dict(strategy=name,annualized_return=ret,sharpe=sharpe,maximum_drawdown=dd,annualized_one_way_turnover=turn,window_start='20260105',window_end='20260930',sessions=181,initial_cash=10_000_000,slippage_bps=10,execution_version='same',complete=True,**kw)


def test_strict_domination_and_tradeoff():
    out={r['strategy']:r for r in pareto_front([row('old'),row('better',ret=.2),row('lowrisk',ret=.05,dd=.01)])}
    assert out['old']['dominated_by']==['better']
    assert out['lowrisk']['pareto_status']=='NONDOMINATED'


def test_equal_points_do_not_dominate_each_other():
    assert all(r['pareto_status']=='NONDOMINATED' for r in pareto_front([row('a'),row('b')]))


def test_cost_window_and_incomplete_are_not_mixed():
    other=row('b');other['slippage_bps']=20
    with pytest.raises(ValueError,match='INCOMPARABLE'):pareto_front([row('a'),other])
    other=row('b',ret=10);other['complete']=False
    out=pareto_front([row('a'),other]);assert out[0]['pareto_status']=='NONDOMINATED' and out[1]['pareto_status']=='INELIGIBLE'
    other=row('b');other['sharpe']=None
    assert pareto_front([other])[0]['pareto_status']=='INELIGIBLE'


def test_label_end_must_be_strictly_before_training_cutoff():
    frame=pd.DataFrame({'date':['20250101']*3,'label_end_20':['20250330','20250401',None]})
    assert mature_training_mask(frame,'20250401',20).tolist()==[True,False,False]


def test_residual_controls_are_orthogonal_and_missing_is_not_raw():
    idx=range(30);groups=pd.Series(['A']*15+['B']*15,index=idx);size=pd.Series(np.exp(np.linspace(1,3,30)),index=idx)
    value=pd.Series(np.arange(30)+np.sin(np.arange(30)),index=idx)
    residual=residualize_cross_section(value,groups,size)
    assert abs(residual.groupby(groups).sum()).max()<1e-9
    assert abs(residual@np.log(size))<1e-8
    groups.loc[0]=None
    assert pd.isna(residualize_cross_section(value,groups,size).loc[0])
