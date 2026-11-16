"""R5 first-close native Qlib + integer acceptance; not a performance backtest."""
import json
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_ashare_pareto_inputs import ROOT, OUT
from replay_four_strategy_2026 import StudyStore
from moneymore.config import BacktestConfig
from moneymore.research.cost_risk import cost_risk_inputs
from moneymore.research.enhanced_indexing import EnhancedPolicy, optimize_snapshot
from moneymore.research.cost_integer import plan_integer_execution


def main():
    selection=json.loads((OUT/'blend-selection/results.json').read_bytes())
    selected=selection['selected']
    if selected is None:raise ValueError('NO_VALIDATED_SIGNAL_BLEND')
    predictions=pd.read_parquet(OUT/'walkforward-repaired/predictions.parquet')
    signal='20251231'
    part=predictions.loc[predictions.date.eq(signal)&predictions.strategy.isin(['tree_residual20','nonnegative_residual20'])]
    matrix=part.pivot(index='symbol',columns='strategy',values='score')
    scores=(selected['tree_weight']*matrix.tree_residual20+selected['nonnegative_weight']*matrix.nonnegative_residual20).dropna()
    store=StudyStore();store.day=signal
    basic=[pd.read_parquet(p) for p in (ROOT/'state/four-strategy-2026-jan-sep-v1/official').glob('daily_basic-202512*.parquet')]
    store.cache['daily_basic']=pd.concat([store.read('daily_basic'),*basic],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
    bars=pd.read_parquet(ROOT/'state/mature-multifactor/p1-native-20261103-a/bars-input.parquet')
    bars.trade_date=bars.trade_date.astype(str)
    member=pd.read_parquet(ROOT/'state/mature-multifactor/p1-native-20261103-a/membership.parquet')
    members=set(member.loc[pd.to_datetime(member.datetime).eq(pd.Timestamp(signal)),'instrument'])
    returns,size,raw,controls,benchmark,limits,audit=cost_risk_inputs(store,bars,scores,signal,candidate_members=members)
    dest=OUT/'qlib-first-close';dest.mkdir(exist_ok=True)
    solved=optimize_snapshot(scores.reindex(controls.index),returns,size,pd.Series(dtype=float),EnhancedPolicy(objective_preconditioning=True),style_controls=controls,style_limits=limits,benchmark_weights=benchmark,controls_prestandardized=True,allow_cash_bootstrap=True,reject_turnover_relaxation=True,solver_max_iters=1000)
    (dest/'continuous.json').write_text(json.dumps(solved,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'risk-input-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    if not solved.get('weights'):raise ValueError('QLIB_CONTINUOUS_BLOCKED')
    prices=bars.loc[bars.trade_date.eq(signal)].set_index('ts_code').close.dropna().to_dict()
    results=[]
    for bps in (10,20):
        config=BacktestConfig.model_validate({**BacktestConfig.from_yaml(ROOT/'configs/default.yaml').model_dump(),'initial_cash':10_000_000,'slippage_bps':bps,'max_position_weight':.15})
        integer=plan_integer_execution(solved['weights'],prices,{}, {},10_000_000,10_000_000,config,controls,limits,benchmark_weights=benchmark,style_controls_prestandardized=True,time_limit=60,solver_threads=2,mip_relative_gap=.01,research_secondary_time_limit=10,accept_validated_incumbent=True,stock_active_bound=max(0.,.15-float(benchmark.max())))
        (dest/f'integer-bps{bps}.json').write_text(json.dumps(integer,ensure_ascii=False,indent=2),encoding='utf-8')
        results.append({'bps':bps,'status':integer['status'],'reason':integer.get('reason'),'orders':len(integer.get('orders',[]))})
        print(results[-1],flush=True)
    (dest/'results.json').write_text(json.dumps({'signal_date':signal,'selection':selected,'risk_assets':len(controls),'results':results,'reference_close_only':True,'future_open_not_used':True,'backtest_complete':False,'real_orders':False},indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
