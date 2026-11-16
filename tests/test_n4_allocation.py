import numpy as np
import pandas as pd
import pytest
from moneymore.research.n4_allocation import past_returns, validate_weights, constrain_mix, netting_audit


def test_returns_are_strictly_lagged():
    dates=pd.date_range('2025-01-01',periods=150).strftime('%Y%m%d')
    x=pd.DataFrame(np.random.default_rng(1).normal(size=(150,4))*.01,index=dates)
    expected=past_returns(x,dates[119])
    x.loc[dates[120]:]=999
    pd.testing.assert_frame_equal(expected,past_returns(x,dates[119]))
    assert expected.index[-1]==dates[119]
    with pytest.raises(ValueError,match='INSUFFICIENT'):
        past_returns(x,dates[60])


@pytest.mark.parametrize('failure',['missing','duplicate','constant','unsorted'])
def test_bad_historical_series_rejected(failure):
    x=pd.DataFrame(np.random.default_rng(1).normal(size=(120,4)),index=[f'{i:04}' for i in range(120)])
    if failure=='missing': x.iloc[2,2]=np.nan
    if failure=='duplicate': x.index=[*x.index[:-1],x.index[-2]]
    if failure=='constant': x.iloc[:,0]=0
    if failure=='unsorted': x=x.iloc[::-1]
    with pytest.raises(ValueError): past_returns(x,'9999')


def test_weight_bounds_and_core_only_fallback():
    assert validate_weights({'core':1},['core','new']).tolist()==[1,0]
    with pytest.raises(ValueError): validate_weights({'a':.7,'b':.3},['a','b'])
    with pytest.raises(ValueError): validate_weights({'a':.5,'b':np.nan},['a','b'])
    with pytest.raises(ValueError): validate_weights({'a':.5,'b':.3},['a','b'])


def test_mix_enforces_sector_and_cash_by_reduction_only():
    w,a=constrain_mix({'a':.3,'b':.3,'c':.3},{'a':'one','b':'one','c':'two'},{'one':.2,'two':.2},{},.6,.5,.08)
    assert sum(w.values())<=.6
    assert w['a']+w['b']==pytest.approx(.28)
    assert w['c']==pytest.approx(.28)
    assert a['reduced_exposure']==pytest.approx(.34)


def test_frozen_not_silently_reduced():
    with pytest.raises(ValueError,match='LOCKED_INDUSTRY'):
        constrain_mix({'a':.3},{'a':'one'},{'one':.1},{'a':.3},1,.5,.08)
    with pytest.raises(ValueError,match='LOCKED_BUDGET'):
        constrain_mix({'a':.3},{'a':'one'},{'one':.9},{'a':.3},1,.05,.08)
    w,_=constrain_mix({'a':.1,'b':.3},{'a':'one','b':'two'},{'one':.4,'two':.4},{'a':.2},.5,.5,.08)
    assert w['a']==.2


def test_netting_cancels_crossed_sleeves_not_fake_fills():
    a=netting_audit({'one':{'s':200},'two':{}},{'s':{'one':1,'two':1}},{'s':200},{'s':10})
    assert a['virtual_gross_notional']==2000
    assert a['net_notional_before_lot_cash_and_permissions']==0
    assert a['crossed_notional_both_sides']==2000


def test_netting_keeps_external_net_and_unassigned_positions():
    a=netting_audit({'one':{'s':300},'two':{}},{'s':{'one':1,'two':1}},{'s':200},{'s':10})
    assert a['net_notional_before_lot_cash_and_permissions']==1000
    b=netting_audit({'one':{},'two':{}},{},{'s':200},{'s':10})
    assert b['net_notional_before_lot_cash_and_permissions']==2000


def test_n4_horizons_and_feature_families_are_disjoint_and_bounded():
    import json
    from pathlib import Path
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/ashare_n4_v1.json').read_bytes())
    assert c['real_orders_enabled'] is False and c['production_enabled'] is False
    assert [s['horizon'] for s in c['sources'].values()]==[60,20,5]
    groups=[]
    for spec in c['sources'].values():
        groups.append(set(spec['features']+[p+str(w) for p in spec.get('prefixes',[]) for w in spec['windows']]))
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
