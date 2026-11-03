"""Hold only the explicitly diagnosed pair; no holdings or history changes."""
from pathlib import Path
from datetime import datetime,UTC
import json,sqlite3
from moneymore.candidate_lifecycle import lifecycle_rows
from moneymore.execution.paper import PaperBroker
root=Path(__file__).resolve().parents[1];states=lifecycle_rows(root);now=datetime.now(UTC).isoformat()
with sqlite3.connect(root/'state/candidate-lifecycle.sqlite3') as connection:
    connection.execute('CREATE TABLE IF NOT EXISTS model_quality_holds (tag TEXT, held_at TEXT, previous_payload TEXT, reason TEXT)')
    for tag in ['20261103A','20261103B']:
        reason='1103模型仅1棵树，122只并列最高分；暂停新增模拟交易，等待预测区分度修复验收'
        connection.execute('INSERT INTO model_quality_holds VALUES (?,?,?,?)',(tag,now,json.dumps(states.get(tag,{})),reason))
        state={**states.get(tag,{}),'candidate_tag':tag,'status':'REVIEW_HOLD','review_date':'20261103','reason':reason,'history_preserved':True,'model_quality_hold':True}
        connection.execute('INSERT INTO candidate_state VALUES (?,?) ON CONFLICT(candidate_tag) DO UPDATE SET payload=excluded.payload',(tag,json.dumps(state,ensure_ascii=False)))
for tag in ['20261103A','20261103B']:PaperBroker(root/'state/paper_orders.sqlite3').cancel_pending('qlib_candidate_'+tag,'MODEL_DISCRIMINATION_HOLD')
print('PAIR_HELD_NO_FILLS_OR_HISTORY_REWRITE')
