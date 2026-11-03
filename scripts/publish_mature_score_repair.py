"""Audited prospective PAPER-only score repair of the two still-empty accounts."""
from pathlib import Path
from datetime import datetime,UTC
import hashlib,json,sqlite3,shutil,os
import pandas as pd
from threadpoolctl import threadpool_limits
from moneymore.candidate_lifecycle import lifecycle_rows,trading_allowed
from moneymore.execution.paper import PaperBroker
from moneymore.research.prediction_quality import prediction_quality

def main():
    root=Path.cwd();model=root/'state/mature-multifactor/p2-ridge-repair-20261103-b'
    fixture=root/'state/mature-multifactor/ridge-engine-20261103-b/results.json'
    meta=json.loads((model/'results.json').read_bytes());proof=json.loads(fixture.read_bytes())
    scores=pd.read_parquet(model/'latest-predictions.parquet')['score'];quality=prediction_quality(scores)
    assert len(scores)==1000 and scores.index.is_unique and quality['passed'] and meta['reload_max_abs_error']==0
    assert proof['status']=='P4_ENGINE_FIXTURE_LOOP_VERIFIED' and all(r['idempotent_retry'] and r['reconciliation']['matched'] for r in proof['results'])
    for name,digest in meta['artifact_hashes'].items():assert hashlib.sha256((model/name).read_bytes()).hexdigest()==digest
    registry_path=root/'state/mature-multifactor/registry.json';registry=json.loads(registry_path.read_bytes())
    tags={'20261103A','20261103B'};assert {r['candidate_tag'] for r in registry['accounts']}==tags
    broker=PaperBroker(root/'state/paper_orders.sqlite3');states=lifecycle_rows(root)
    for row in registry['accounts']:
        snapshot=broker.account_snapshot({},row['account_id'])
        assert snapshot['cash']==10_000_000 and not snapshot['positions'] and not broker.fills(row['account_id'])
        assert states[row['candidate_tag']].get('model_quality_hold'), 'REPAIR_REQUIRES_EXPLICIT_HOLD'
    stamp=datetime.now(UTC).strftime('%Y%m%d-%H%M%S');out=root/'state/mature-multifactor'/('score-release-'+stamp);out.mkdir()
    shutil.copy2(registry_path,out/'previous-registry.json')
    for tag in tags:
        for kind in ['qlib-portfolio-plans','qlib-candidate-shadow','qlib-order-previews']:
            path=root/'state'/kind/tag/'20261103.json'
            if path.exists():shutil.copy2(path,out/(tag+'-'+kind+'.json'))
    # Old LGB replay numbers must NOT be presented as performance of this Ridge model.
    (out/'historical-replay.json').write_text(json.dumps({'cases':[],'scope':'NO_HISTORICAL_RETURNS_CLAIMED_FOR_REPAIRED_MODEL','old_lgb_replay':registry['historical_replay']}))
    registry.update(version='mature_alpha158_ridge_pair_v2',model_dir=str(model),model_manifest=str(model/'results.json'),
        model_display_name='Qlib Alpha158 + Ridge（连续多因子）',historical_rank_ic=meta['test_rank_ic'],
        historical_replay=str(out/'historical-replay.json'),previous_registry=str(out/'previous-registry.json'),
        admission_scope='PAPER_SCORE_AND_EXECUTION_REPAIR_NOT_PREDICTIVE_SUPERIORITY',validation_rank_ic=meta['valid_rank_ic'],
        prediction_quality=quality,repaired_at=datetime.now(UTC).isoformat(),real_orders_enabled=False,promotion_authorized=False)
    for row in registry['accounts']:row.update(model_id=meta['model_id'],strategy_id='mature_ridge_v2_'+row['portfolio_variant'].rsplit('_',1)[-1])
    sources=['src/moneymore/mature_multifactor_runtime.py','src/moneymore/research/mature_multifactor.py',
        'src/moneymore/research/enhanced_indexing.py','src/moneymore/research/prediction_quality.py','scripts/build_mature_alpha158.py','configs/mature_multifactor_v1.yaml']
    files=[root/s for s in sources]+[model/n for n in ['results.json','model.pkl','processor.pkl']]+[fixture]
    registry['evidence_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    temp=registry_path.with_suffix('.json.tmp');temp.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,registry_path)
    # Both accounts remain held until both replacement plans and previews pass.
    from moneymore.mature_multifactor_runtime import mature_candidates,prepare_mature_daily
    from moneymore.mature_order_preview import build_preview
    from moneymore.data.store import ParquetStore
    from moneymore.config import BacktestConfig
    reports=[]
    for row in mature_candidates(root):
        assert row['research_gate_passed']
        prepare_mature_daily(root=root,store=ParquetStore(root/'data'),broker=broker,config=BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate=row,trade_date='20261103')
        report=json.loads((root/'state/qlib-candidate-shadow'/row['candidate_tag']/'20261103.json').read_bytes())
        assert prediction_quality([r['score'] for r in report['scores']])['passed']
        preview=build_preview(root,row,'20261103');assert preview['status']=='REFERENCE_CLOSE_PREVIEW'
        assert preview['planned_trade_date']=='20261104' and preview['orders'] and preview['real_orders_enabled'] is False
        reports.append({'tag':row['candidate_tag'],'orders':len(preview['orders']),'boards':preview['board_weights'],'planned_trade_date':'20261104'})
    with sqlite3.connect(root/'state/candidate-lifecycle.sqlite3') as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS model_quality_releases (tag TEXT, released_at TEXT, previous_payload TEXT, evidence TEXT)')
        for tag in tags:
            previous=lifecycle_rows(root)[tag]
            connection.execute('INSERT INTO model_quality_releases VALUES (?,?,?,?)',(tag,stamp,json.dumps(previous),str(out)))
            state={**previous,'status':'OBSERVING','model_quality_hold':False,'reason':'连续预测与模拟执行修复通过；历史预测指标偏弱，仅独立模拟观察，不代表收益优势','model_id':meta['model_id']}
            connection.execute('UPDATE candidate_state SET payload=? WHERE candidate_tag=?',(json.dumps(state,ensure_ascii=False),tag))
    assert all(trading_allowed(root,tag,'20261104') for tag in tags)
    result={'status':'PAPER_SCORE_REPAIR_RELEASED','prediction_quality':quality,'plans':reports,
        'valid_rank_ic':meta['valid_rank_ic'],'test_rank_ic':meta['test_rank_ic'],'profitability_proven':False,
        'real_orders_enabled':False,'production_fills_created':0,'previous_plans_preserved':True}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
