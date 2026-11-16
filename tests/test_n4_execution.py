import json
from pathlib import Path
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('split',[0.,1.])
def test_source_intent_never_uses_next_open_fill(tmp_path,monkeypatch,split):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import replay_ashare_n4_sources as n4
    p=n4.SleevePlanner.__new__(n4.SleevePlanner)
    p.n4=json.loads(n4.CONFIG.read_bytes())
    p.contract={'rebalance_sessions':20}
    p.output=tmp_path
    kind='short--rank_cap'
    def fake_parent(self,**c):
        path=tmp_path/'decisions'/kind/(c['day']+'.json')
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'status':'OPEN_PLAN_VERIFIED','target_quantities':{'A':int(100*(1+split))}}))
        return {'A':0},'opening_permission_blocked'
    monkeypatch.setattr(n4.old.RankPlanner,'__call__',fake_parent)
    c={'kind':kind,'day':'20260105','signal':'20251231','index':0,'night':{'equity':10000,'unsettled_stock_rights':{}},'snapshot':{'unsettled_stock_rights':{}},'night_holdings':{},'last':{'A':10},'members':{'A'},'actions':pd.DataFrame([{'symbol':'A','ex_date':pd.Timestamp('20260105'),'stock_ratio':split}])}
    p(**c)
    intent=json.loads((tmp_path/'intents/short/20260105.json').read_bytes())
    assert intent['weights']['A']==pytest.approx(.1)
    assert intent['next_open_fills_used_for_selection'] is False
    assert p.contract=={'rebalance_sessions':20}


@pytest.mark.parametrize('future_field',['return_end','signal_date'])
def test_blend_rejects_future_weight_record_before_any_trading(tmp_path,monkeypatch,future_field):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import replay_ashare_n4_blends as n4
    p=n4.BlendPlanner.__new__(n4.BlendPlanner)
    p.output=tmp_path
    record={'method':'hrp','execution_date':'20260105','signal_date':'20251231','return_end':'20251231'}
    record[future_field]='20260106'
    p.weight_records=[record]
    with pytest.raises(ValueError,match='LOOKAHEAD'):
        p(kind='hrp',day='20260105',signal='20251231',held={},members=set())


def test_failed_source_parent_restores_cadence(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import replay_ashare_n4_sources as n4
    p=n4.SleevePlanner.__new__(n4.SleevePlanner)
    p.n4=json.loads(n4.CONFIG.read_bytes())
    p.contract={'rebalance_sessions':20}
    def fail(*args,**kwargs): raise RuntimeError('test')
    monkeypatch.setattr(n4.old.RankPlanner,'__call__',fail)
    with pytest.raises(RuntimeError): p(kind='fundamental--rank_cap')
    assert p.contract=={'rebalance_sessions':20}


def test_known_merger_exclusion_is_common_past_available_and_nonmutating(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import replay_ashare_n4_sources as n4
    p=n4.SleevePlanner.__new__(n4.SleevePlanner)
    attrs=pd.DataFrame({'industry':['ship','bank'],'total_mv':[100,200]},index=['601989.SH','A'])
    covariance=pd.DataFrame([[1,0],[0,1]],index=attrs.index,columns=attrs.index)
    monkeypatch.setattr(n4.old.RankPlanner,'risk',lambda self,signal:(covariance,attrs,{'eligible':2}))
    _,before,audit=p.risk('20240918')
    assert len(before)==2 and audit=={'eligible':2}
    _,after,audit=p.risk('20250630')
    assert after.index.tolist()==['A']
    assert audit['known_unsupported_merger_exclusions']==['601989.SH']
    assert len(attrs)==2


def test_transparent_style_controls_lock_direction_and_ignore_future(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from replay_ashare_n4_style_controls import fixed_scores
    frame=pd.DataFrame({'date':['20260105']*3+['20260106']*3,'symbol':['A','B','C']*2,'mom6_1':[.5,.1,-.2]*2,'mom12_1':[.7,.3,-.4]*2,'ROC5':[.8,1.,1.2]*2})
    first=fixed_scores(frame)
    assert first[('factor_momentum','20260105')].idxmax()=='A'
    assert first[('factor_reversal','20260105')].idxmax()=='C'
    frame.loc[frame.date.eq('20260106'),['mom6_1','mom12_1','ROC5']]=999
    second=fixed_scores(frame)
    for name in ['factor_momentum','factor_reversal']:
        pd.testing.assert_series_equal(first[(name,'20260105')],second[(name,'20260105')])


def test_momentum_missing_horizon_does_not_double_other_weight(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from replay_ashare_n4_style_controls import fixed_scores
    frame=pd.DataFrame({'date':['20260105']*2,'symbol':['A','B'],'mom6_1':[.5,.1],'mom12_1':[float('nan'),.3],'ROC5':[.8,1.]})
    result=fixed_scores(frame)[('factor_momentum','20260105')]
    assert result['A']==.75


def test_style_nonfinite_features_use_explicit_neutral_scores(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from replay_ashare_n4_style_controls import fixed_scores
    frame=pd.DataFrame({'date':['20260105']*3,'symbol':['A','B','C'],'mom6_1':[float('nan'),float('inf'),.5],'mom12_1':[float('nan'),float('-inf'),.3],'ROC5':[float('nan'),float('inf'),1.1]})
    scores=fixed_scores(frame)
    assert scores[('factor_momentum','20260105')].to_dict()=={'A':.5,'B':.5,'C':1.}
    assert scores[('factor_reversal','20260105')].to_dict()=={'A':1.,'B':1.,'C':1.1}
