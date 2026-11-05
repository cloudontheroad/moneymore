"""Offline cost-bootstrap feasibility diagnosis at observed paper-fill marks.

No order submission, no account/history changes, no use of future prices.
"""
import json
from pathlib import Path
import sqlite3
import pandas as pd
from threadpoolctl import threadpool_limits
from moneymore.config import BacktestConfig
from moneymore.research.style_v2_experiment import candidate_execution_config
from moneymore.cost_ab_runtime import execute_cost_plan
from moneymore.research.cost_aware import risk_compliant
from moneymore.qlib_candidate_observer import candidate_catalog

root=Path(__file__).resolve().parents[1]
candidate=next(r for r in candidate_catalog(root) if r['candidate_tag']=='20261104CB')
config=candidate_execution_config(BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate)
plan=json.loads((root/'state/qlib-portfolio-plans/20261104CB/20261104.json').read_bytes())
with sqlite3.connect(f"file:{(root/'state/paper_orders.sqlite3').as_posix()}?mode=ro",uri=True) as c:
    fills=c.execute('SELECT symbol,quantity,price,fee FROM fills WHERE account_id=? AND trade_date=?',('qlib_candidate_20261103B','20261105')).fetchall()
marks={s:price/(1+config.slippage_bps/10000) for s,q,price,fee in fills}
required={s for s,w in plan['target_weights'].items() if w>1e-6}
assert required<=marks.keys(), sorted(required-marks.keys())
styles=pd.DataFrame.from_dict(plan['styles'],orient='index')
weights={s:q*marks[s]/10_000_000 for s,q,price,fee in fills}
print('actual_original_B_risk_compliant',risk_compliant(weights,styles,plan['style_limits'],plan['benchmark_weights']),flush=True)
with threadpool_limits(limits=4):
    result=execute_cost_plan(plan,marks,{}, {},10_000_000,10_000_000,config,dict.fromkeys(marks,True),dict.fromkeys(marks,True))
out=root/'state/cost-aware-ab-v2/execution-hotfix-20261105/bootstrap-diagnostic.json'
out.write_text(json.dumps({'price_source':'OBSERVED_ORIGINAL_B_PAPER_FILL_PRE_SLIPPAGE_NOT_LIVE','result':result},default=str,indent=2))
print('cost_result',result['status'],result.get('reason'),flush=True)
