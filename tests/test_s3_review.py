import hashlib
import json
from moneymore.research.s3_review import STAGES,review_decision,read_public_review


def test_missing_or_failed_stage_never_authorizes_observation():
    assert not review_decision({})['observation_activation_authorized']
    reports={s:{'status':'PASS'} for s in STAGES};reports['prediction']={'status':'FAIL'}
    assert not review_decision(reports)['observation_activation_authorized']
    assert not review_decision(reports)['real_orders_enabled']


def test_observation_does_not_require_its_own_future_days_but_promotion_does():
    reports={s:{'status':'PASS'} for s in STAGES[:-1]}
    decision=review_decision(reports)
    assert decision['observation_activation_authorized'] and not decision['promotion_authorized']
    reports['forward_observation']={'status':'PASS','mature_prospective_days':60,'backfilled_days_counted':False}
    assert review_decision(reports)['promotion_authorized']
    reports['forward_observation']['backfilled_days_counted']=True
    assert not review_decision(reports)['promotion_authorized']


def test_corrupt_or_changed_evidence_is_fail_closed(tmp_path):
    folder=tmp_path/'state'/'style-balance';folder.mkdir(parents=True);report=folder/'review.json'
    assert read_public_review(report) is None
    report.write_text('{')
    assert read_public_review(report)['status']=='UNAVAILABLE'
    report.write_text('[]')
    assert read_public_review(report)['status']=='UNAVAILABLE'
    evidence=folder/'evidence.json';evidence.write_text('{}')
    reports={s:{'status':'PASS'} for s in STAGES[:-1]}
    report.write_text(json.dumps({'schema':'S3_REVIEW_V1','reports':reports,'public_summary':{'title':'Review','comparison':[]},
        'input_hashes':{str(evidence):hashlib.sha256(evidence.read_bytes()).hexdigest()}}))
    assert read_public_review(report)['observation_authorized']
    evidence.write_text('{"changed":true}')
    assert not read_public_review(report)['observation_authorized']
