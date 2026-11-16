from types import SimpleNamespace
import pytest
import moneymore.qlib_candidate_observer as observer


def test_old_candidate_failure_does_not_skip_newer_plans(tmp_path,monkeypatch):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/qlib_challenger.yaml').write_text('candidate_observation_policy: {enabled: true}')
    monkeypatch.setattr(observer,'candidate_catalog',lambda _: [{'candidate_tag':'20261104'},{'candidate_tag':'20261103'}])
    monkeypatch.setattr(observer,'run_lifecycle_review',lambda *a:{})
    calls=[]
    def run(**kw):
        tag=kw['candidate']['candidate_tag'];calls.append(tag)
        if tag=='20261103':raise ValueError('HELD_RISK_MISSING')
        return SimpleNamespace(status='COMPLETED')
    monkeypatch.setattr(observer,'run_candidate_daily',run)
    with pytest.raises(RuntimeError,match='20261103: ValueError: HELD_RISK_MISSING'):
        observer.run_candidate_queue_daily(root=tmp_path,store=None,broker=None,config=None,trade_date='20261110')
    assert calls==['20261103','20261104']
