"""Validated signal -> native Qlib -> prior-close integer plan -> next-open guard."""
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_ashare_pareto_inputs import ROOT, OUT
from replay_four_strategy_2026 import StudyStore
from moneymore.costs import execution_price,transaction_fee
from moneymore.models import Side
from moneymore.research.cost_risk import cost_risk_inputs
from moneymore.research.cost_aware import risk_compliant
from moneymore.research.enhanced_indexing import EnhancedPolicy,optimize_snapshot
from moneymore.research.cost_integer import plan_integer_execution
import research_ashare_paper_routes as engine


def exact_open_guard(desired,held,snapshot,opened,buy,sell,config,controls,limits,benchmark):
    """Only physical cash/lot clipping at open, then recheck the complete portfolio."""
    cash=snapshot['cash'];available={p['symbol']:int(p['available_quantity']) for p in snapshot['positions']}
    feasible=dict(held)
    requests=[(s,int(desired.get(s,0))-held.get(s,0)) for s in sorted(set(desired)|set(held))]
    requests.sort(key=lambda x:(x[1]>0,x[0]))
    for symbol,delta in requests:
        if not delta:continue
        side=Side.BUY if delta>0 else Side.SELL
        if symbol not in opened or not (buy if delta>0 else sell).get(symbol,False):return held,'OPEN_PRICE_OR_PERMISSION'
        q=abs(delta);price=execution_price(opened[symbol],side,config.slippage_bps)
        if delta<0:
            if q>available.get(symbol,0) or (q<held.get(symbol,0) and q%100):return held,'OPEN_AVAILABILITY_OR_LOT'
            cash+=q*price-transaction_fee(q*price,side,config)
            feasible[symbol]=held.get(symbol,0)-q
        else:
            q=q//100*100
            cap=max(0.,config.max_position_weight*snapshot['equity']-held.get(symbol,0)*opened[symbol])
            q=min(q,int(cap/price)//100*100)
            q=min(q,max(0,int((cash-config.minimum_commission)/(price*(1+config.commission_rate+config.transfer_fee_rate))))//100*100)
            if symbol.startswith('688') and not held.get(symbol,0) and q<200:q=0
            while q>0 and q*price+transaction_fee(q*price,side,config)>cash+1e-8:q-=100
            if q>0:cash-=q*price+transaction_fee(q*price,side,config)
            feasible[symbol]=held.get(symbol,0)+q
    rights=snapshot['unsettled_stock_rights']
    values={s:(feasible.get(s,0)+rights.get(s,0))*opened[s] for s in set(feasible)|set(rights) if feasible.get(s,0)+rights.get(s,0)>0}
    equity=cash+sum(values.values())+snapshot.get('receivable_cash',0.)
    if equity<=0 or not risk_compliant({s:v/equity for s,v in values.items()},controls,limits,benchmark):return held,'OPEN_STYLE_POSTCHECK'
    return feasible,None


class QlibPlan:
    def __init__(self,bps,interval):
        self.interval=interval;self.store=StudyStore()
        basics=[pd.read_parquet(p) for p in (ROOT/'state/four-strategy-2026-jan-sep-v1/official').glob('daily_basic-*.parquet')]
        self.store.cache['daily_basic']=pd.concat([self.store.base.read('daily_basic'),*basics],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
        suspensions=[pd.read_parquet(p) for p in (ROOT/'state/four-strategy-2026-jan-sep-v1/official').glob('suspend_d-*.parquet')]
        self.suspension_proof=pd.concat(suspensions,ignore_index=True).drop_duplicates()
        self.bars=pd.read_parquet(ROOT/'state/mature-multifactor/p1-native-20261103-a/bars-input.parquet');self.bars.trade_date=self.bars.trade_date.astype(str)
        proof=pd.read_parquet(OUT/'label-repair-v2/source-overlap-audit.parquet')
        allowed=set(proof.loc[proof.accepted,'source'])
        extras=[pd.read_parquet(p) for p in (OUT/'endpoint-cache').glob('*.parquet') if str(p.relative_to(ROOT)) in allowed]
        if extras:self.bars=pd.concat([self.bars,*extras],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='first')
        self.forecasts=pd.read_parquet(OUT/'walkforward-repaired/predictions.parquet')
        self.selected=json.loads((OUT/'blend-selection/results.json').read_bytes())['selected']
        if self.selected is None:raise ValueError('NO_2025_VALIDATED_BLEND')
        self.dest=engine.OUT/f'bps{bps}/plans';self.dest.mkdir(parents=True,exist_ok=True)

    def __call__(self,**c):
        day=c['day'];signal=c['signal'];held=c['held'];nh=c['night_holdings'];members=c['members'];night=c['night'];snap=c['snapshot'];exits=set(held)-members
        if c['index']%self.interval and held and not exits:return held,None
        self.store.day=signal
        f=self.forecasts.loc[self.forecasts.date.eq(signal)&self.forecasts.strategy.isin(['tree_residual20','nonnegative_residual20'])].pivot(index='symbol',columns='strategy',values='score')
        scores=(self.selected['tree_weight']*f.tree_residual20+self.selected['nonnegative_weight']*f.nonnegative_residual20).reindex(sorted(members)).dropna()
        missing_forecast=(set(held)&members)-set(scores.index)
        scores=scores.reindex(sorted(set(scores.index)|set(held)))
        scores.loc[list(exits|missing_forecast)]=0.
        returns,size,raw,controls,benchmark,limits,audit=cost_risk_inputs(self.store,self.bars,scores,signal,candidate_members=members,suspension_proof=self.suspension_proof)
        # Freeze an interior planning band, retaining the original execution limit.
        # Otherwise a solution exactly on the close boundary fails after tiny gaps.
        planning_limits={name:bound*.8 for name,bound in limits.items()}
        audit['planning_limit_fraction']=.8
        audit['execution_limits_unchanged']=True
        def save(name,value):
            (self.dest/f'{day}-{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        save('risk',audit)
        if set(held)-set(controls.index):return held,'HELD_RISK_MISSING'
        values=pd.Series({s:(q+night['unsettled_stock_rights'].get(s,0))*c['last'][s] for s,q in nh.items()},dtype=float)
        w0=values/values.sum() if len(values) else values
        frozen=(set(held)&c['halted'])|missing_forecast
        solved=optimize_snapshot(scores.reindex(controls.index),returns,size,w0,EnhancedPolicy(objective_preconditioning=True),style_controls=controls,style_limits=planning_limits,benchmark_weights=benchmark,controls_prestandardized=True,allow_cash_bootstrap=True,reject_turnover_relaxation=True,forced_sell_symbols=exits-c['halted'],forced_hold_symbols=frozen,solver_max_iters=1000)
        save('continuous',solved)
        if not solved.get('weights'):return held,'QLIB_CONTINUOUS_BLOCKED'
        integer=plan_integer_execution(solved['weights'],c['last'],nh,nh,night['cash'],night['equity'],c['config'],controls,planning_limits,benchmark_weights=benchmark,style_controls_prestandardized=True,time_limit=60,solver_threads=2,mip_relative_gap=.01,research_secondary_time_limit=10,accept_validated_incumbent=True,stock_active_bound=max(0.,.15-float(benchmark.max())),locked_shares=night['unsettled_stock_rights'],forced_sell_symbols=exits-c['halted'],fixed_quantities={s:nh.get(s,0) for s in frozen})
        save('integer',integer)
        if integer['status']!='PASS_REFERENCE_QUOTES_ONLY':return held,'QLIB_INTEGER_BLOCKED:'+str(integer.get('reason'))
        desired={}
        for s,q in integer['target_quantities'].items():
            economic=q+night['unsettled_stock_rights'].get(s,0)
            for event in c['actions'].loc[c['actions'].symbol.eq(s)&c['actions'].ex_date.eq(pd.Timestamp(day))].itertuples():economic*=1+event.stock_ratio
            adjusted=economic-snap['unsettled_stock_rights'].get(s,0)
            desired[s]=int(q) if abs(adjusted-q)<1e-8 else max(0,int(np.floor(adjusted/100))*100)
            if s in frozen:desired[s]=held.get(s,0)
        feasible,reason=exact_open_guard(desired,held,snap,c['opened'],c['buy'],c['sell'],c['config'],controls,limits,benchmark)
        save('open-guard',{'reason':reason,'close_plan_quantities':integer['target_quantities'],'corporate_action_adjusted_quantities':desired,'cash_lot_checked_quantities':feasible,'next_open_used_for_selection':False,'actual_orders_sent':False})
        return feasible,reason


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bps',type=int,choices=[10,20],default=10);parser.add_argument('--interval',type=int,choices=[15,20,25],default=20);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    engine.OUT=OUT/f'qlib-replays/interval{args.interval}'/('smoke-proof-v4' if args.smoke else 'full-proof-v4');engine.KINDS=['validated_qlib']
    with threadpool_limits(limits=2):engine.run(args.bps,smoke=args.smoke,execution_planner=QlibPlan(args.bps,args.interval))
