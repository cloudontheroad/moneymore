"""Post-result diagnosis, NOT a new searched strategy: freeze 10bps weights at 20bps execution."""
import json
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
import replay_ashare_n4_blends as blend
from prepare_four_strategy_2026 import save
from report_ashare_mainline_replay import audit_ledger
from report_ashare_n4 import attribution
from moneymore.research.ashare_mainline import sha256
from moneymore.research.period_statistics import period_statistics

ROOT,BASE=blend.ROOT,blend.BASE
RUNS=BASE/'n4-matrix-v2'
OUT=RUNS/'fixed10-allocation-execution20'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    loaded=blend.sources.load_inputs('evaluation')
    allocation=RUNS/'allocation-cost10.json'
    planner=blend.BlendPlanner(OUT,loaded,RUNS/'evaluation-cost20',allocation)
    engine=blend.sources.old.engine
    engine.load_inputs=lambda:loaded
    engine.KINDS=['hrp_positive']
    engine.OUT=OUT
    save(OUT/'contract.json',{'role':'AFTER_RESULT_COST_VS_POLICY_DIAGNOSIS_NOT_STRATEGY_SELECTION','allocation_history_cost_bps':10,'execution_cost_bps':20,'fixed_allocation_sha256':sha256(allocation),'source_runner_sha256':sha256(Path(blend.__file__))})
    engine.run(20,execution_planner=planner,config_overrides={'initial_cash':10_000_000,'max_position_weight':.05})
    folder=OUT/'bps20'
    case='hrp_positive'
    equity=pd.read_parquet(folder/case/'equity.parquet')
    fills=pd.read_parquet(folder/case/'fills.parquet')
    ledger=audit_ledger(folder/case/'paper.sqlite3',fills)
    if len(equity)!=181:raise ValueError('INCOMPLETE_DIAGNOSTIC')
    records=json.loads(allocation.read_bytes())['records']
    verified=0
    for p in sorted((OUT/'decisions'/case).glob('*.json')):
        d=json.loads(p.read_bytes())
        if d.get('reason'):raise ValueError('PROTECTED_DIAGNOSTIC_REQUIRES_REVIEW')
        if d['status']=='OPEN_PLAN_VERIFIED':
            expected=[r for r in records if r['method']==case and r['execution_date']<=p.stem][-1]
            if d['allocation']!=expected or not d['open_postcheck']['passed']:raise ValueError('COST_POLICY_NOT_FIXED')
            verified+=1
    h=pd.read_parquet(folder/'holdings.parquet')
    panel=pd.read_parquet(BASE/'panel-v2/panel.parquet',columns=['date','symbol','industry'])
    pnl=attribution(folder,case,equity,fills,h,panel)
    comparison=json.loads((RUNS/'final/results.json').read_bytes())
    original=next(r for r in comparison['cases'] if r['period']=='blends' and r['bps']==10 and r['case']==case)
    adaptive=next(r for r in comparison['cases'] if r['period']=='blends' and r['bps']==20 and r['case']==case)
    stats=period_statistics(equity.equity,10_000_000)
    result={'status':'FIXED_POLICY_COST_DIAGNOSIS_COMPLETE','ledger':ledger,'stats':stats,'fees':float(fills.fee.sum()),'slippage':float(fills.slippage.sum()),'fills':len(fills),'attribution':pnl,'verified_fixed_policy_trade_dates':verified,'return_difference_vs_10bps_execution_same_weights':stats['net_return']-original['net_return'],'return_difference_vs_20bps_adaptive_weights':stats['net_return']-adaptive['net_return'],'both_policy_and_cost_change_in_original_stress_matrix':True,'production_modified':False}
    save(OUT/'results.json',result)
    print(json.dumps({'status':result['status'],**stats}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
