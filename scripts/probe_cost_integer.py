"""Bounded exact-state execution probe, isolated and read-only to all ledgers."""
from pathlib import Path
import json,sqlite3,argparse
import pandas as pd
from threadpoolctl import threadpool_limits
from replay_cost_ab_v2 import ROOT,OUT,STUDY,StudyStore
from moneymore.config import BacktestConfig
from moneymore.research.replay_attribution import reconstruct_positions,reconstruct_cash
from moneymore.research.mature_multifactor import current_risk_inputs
from moneymore.research.cost_integer import plan_integer_execution
from moneymore.research.quote_permissions import quote_permissions
from prepare_four_strategy_2026 import save

def probe(case,day):
    folder=OUT/'t5'/case;inner=folder/case
    with sqlite3.connect(f'file:{(inner/"paper.sqlite3").as_posix()}?mode=ro',uri=True) as db:
        fills=pd.read_sql_query('SELECT * FROM fills WHERE trade_date<?',db,params=[day]);ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger WHERE trade_date<=?',db,params=[day])
    dates=sorted(set(fills.trade_date)|set(ledger.trade_date)|{day})
    held=reconstruct_positions(fills,ledger,dates)[day];cash=reconstruct_cash(fills,ledger,dates,1e7)[day]
    events=json.loads((folder/'checkpoint.json').read_bytes())['events'][case];event=next(e for e in events if e['date']==day);signal=event['signal_date']
    source=ROOT/'state/mature-multifactor/p1-native-20261103-a';bars=pd.read_parquet(source/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    membership=pd.read_parquet(source/'membership.parquet');members=set(membership.loc[pd.to_datetime(membership.datetime).eq(pd.Timestamp(signal)),'instrument'])
    scores=pd.read_parquet(STUDY/'model/predictions.parquet').score.xs(pd.Timestamp(signal),level='datetime').reindex(sorted(members)).dropna()
    store=StudyStore();store.day=signal;store.basic=[pd.read_parquet(p) for p in sorted((STUDY/'official').glob('daily_basic-*.parquet')) if p.stem.split('-')[-1]<=signal]
    proof=pd.concat([pd.read_parquet(p) for p in sorted((STUDY/'official').glob('suspend_d-*.parquet')) if p.stem.split('-')[-1]<=signal],ignore_index=True)
    returns,size,raw,styles,benchmark,limits,audit=current_risk_inputs(store,bars,scores.reindex(sorted(set(scores.index)|set(held))),signal,suspension_proof=proof,candidate_members=members)
    economics=json.loads((inner/(day+'-economic.json')).read_bytes());marks=bars.loc[bars.trade_date.eq(day)].set_index('ts_code').open.dropna().to_dict()
    gate=pd.read_parquet(STUDY/'official'/('stk_limit-'+day+'.parquet')).set_index('ts_code');susp=pd.read_parquet(STUDY/'official'/('suspend_d-'+day+'.parquet'))
    halted=set(susp.loc[susp.suspend_type.eq('S')&(susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code'])
    buy,sell=quote_permissions(marks,halted,gate,10)
    config=BacktestConfig.model_validate({**BacktestConfig.from_yaml(ROOT/'configs/default.yaml').model_dump(),'initial_cash':1e7,'slippage_bps':10})
    result=plan_integer_execution(economics['weights'],marks,held,held,cash,cash+sum(q*marks[s] for s,q in held.items()),config,styles,limits,
        can_buy=buy,can_sell=sell,benchmark_weights=benchmark,style_controls_prestandardized=True,solver_threads=4,time_limit=15,mip_relative_gap=.01,
        research_objective_scale=10000,economic_tracking_weights=economics['economic_tracking_weights'],
        accept_validated_incumbent=True,stock_active_bound=max(0.,.15-benchmark.max()))
    save(OUT/'t4'/('scaled-'+case+'-'+day+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['target_quantities','orders','active_style_exposures']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',default='b_cost_exec');parser.add_argument('--date',default='20260121');args=parser.parse_args()
    with threadpool_limits(limits=4):probe(args.case,args.date)
