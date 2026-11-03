"""Read-only, fail-closed S3 review using the existing release gate engine."""
import hashlib
import json
from pathlib import Path
from .baseline_v2_release import release_decision

STAGES=['prediction','risk','execution','runtime','forward_observation']
LABELS={'prediction':'A · 独立预测与成本证据','risk':'B · 风格与风险证据','execution':'C · 离线执行与对账',
        'runtime':'运行发布验收','forward_observation':'真实前瞻观察'}
STATUS_LABELS={'PASS':'通过','FAIL':'未通过','INCOMPLETE':'证据不足','NOT_TRIGGERED':'条件未触发','NOT_STARTED':'未开始'}


def review_decision(reports,*,evidence_valid=True):
    protocol={'real_orders_enabled':False,'required_stages':STAGES,'forward_review_days':60,
              'observation_required_stages':STAGES[:-1]}
    return release_decision(protocol,reports,evidence_valid=evidence_valid)


def read_public_review(path):
    path=Path(path)
    if not path.exists():return None
    try:
        if path.stat().st_size>256000:raise ValueError('REVIEW_TOO_LARGE')
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('schema')!='S3_REVIEW_V1':raise ValueError('WRONG_REVIEW_SCHEMA')
        allowed=path.resolve().parents[2]
        for name,digest in report['input_hashes'].items():
            file=Path(name).resolve()
            if not file.is_relative_to(allowed) or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise ValueError('REVIEW_EVIDENCE_CHANGED')
        if not report['input_hashes']:raise ValueError('NO_REVIEW_EVIDENCE')
        decision=review_decision(report['reports']);public=dict(report['public_summary'])
        public.update(status=decision['status'],observation_authorized=decision['observation_activation_authorized'],
                      promotion_authorized=decision['promotion_authorized'],real_orders_enabled=False)
        public['gates']=[{'stage':LABELS[s],'status':STATUS_LABELS.get(report['reports'].get(s,{}).get('status'),'未通过 / 未完成'),
                         'reason':report['reports'].get(s,{}).get('reason','缺少验收证据')} for s in STAGES]
        return public
    except (OSError,ValueError,KeyError,TypeError,AttributeError,RuntimeError):
        return {'status':'UNAVAILABLE','title':'研究验收报告暂不可用','reason':'报告缺失、损坏或证据版本已变化；未授权新模型启用',
                'gates':[],'comparison':[],'observation_authorized':False,'promotion_authorized':False,'real_orders_enabled':False}
