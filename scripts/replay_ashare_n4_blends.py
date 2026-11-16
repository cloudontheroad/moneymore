"""Netted stock portfolios, not NAV averaging; fixed source intents + past-return HRP."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import replay_ashare_n4_sources as sources
from moneymore.research.ashare_mainline import sha256
from moneymore.research.mainline_execution import physical_open_plan
from moneymore.research.n4_allocation import constrain_mix, netting_audit, validate_weights
from prepare_four_strategy_2026 import save

ROOT, BASE, CONFIG = sources.ROOT, sources.BASE, sources.CONFIG


class BlendPlanner(sources.old.RankPlanner):
    def __init__(self, output, loaded, source_run, allocation_file):
        self.n4 = json.loads(CONFIG.read_bytes())
        super().__init__(output,json.loads(sources.old.CONFIG.read_bytes()),loaded)
        self.source_run = source_run
        allocation = json.loads(allocation_file.read_bytes())
        if allocation['config_sha256']!=sha256(CONFIG):
            raise ValueError('ALLOCATION_CONTRACT_CHANGED')
        for p,h in allocation['input_hashes'].items():
            if sha256(Path(p))!=h:
                raise ValueError('ALLOCATION_RETURN_INPUT_CHANGED')
        self.weight_records = allocation['records']
        self.ownership = {}
        self.input_hashes = {}

    def __call__(self, **c):
        method = c['kind']
        day,signal,held = c['day'],c['signal'],c['held']
        destination = self.output / 'decisions' / method / (day+'.json')
        exposures = None

        def record(status, reason=None, **extra):
            save(destination,{'status':status,'reason':reason,'signal_date':signal,'execution_date':day,**extra})

        def protect(reason, **extra):
            feasible,_,clips = physical_open_plan({s:q for s,q in held.items() if s in c['members']},c['snapshot'],c['opened'],c['buy'],c['sell'],c['config'])
            record('BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED',reason,clips=clips,**extra)
            return feasible,reason

        exits=set(held)-c['members']
        records=[r for r in self.weight_records if r['method']==method and r['execution_date']<=day]
        if not records:
            raise ValueError('MISSING_HISTORICAL_ALLOCATION')
        allocation=records[-1]
        if allocation['return_end']>allocation['signal_date'] or allocation['signal_date']>=day:
            raise ValueError('ALLOCATION_LOOKAHEAD')
        weights=validate_weights(allocation['weights'],sources.SOURCES)
        cadence={sources.CORE:20,**{s:spec['cadence'] for s,spec in self.n4['sources'].items()}}
        scheduled=c['index']%self.n4['allocation']['rebalance_sessions']==0 or any(weights[s]>1e-10 and c['index']%cadence[s]==0 for s in sources.SOURCES)
        previous = sorted(p for p in destination.parent.glob('*.json') if p.stem<day) if destination.parent.exists() else []
        retry=bool(previous and json.loads(previous[-1].read_bytes()).get('reason'))
        if not scheduled and held and not retry:
            if exits:
                feasible,_,clips=physical_open_plan({s:q for s,q in held.items() if s not in exits},c['snapshot'],c['opened'],c['buy'],c['sell'],c['config'])
                record('MANDATORY_EXITS_ONLY',clips=clips,exit_symbols=sorted(exits))
                return feasible,None
            record('NOT_SCHEDULED')
            return held,None
        intents={}
        for source in sources.SOURCES:
            path=self.source_run/'intents'/source/(day+'.json')
            intent=json.loads(path.read_bytes())
            if intent['signal_date']!=signal or intent['execution_date']!=day or intent['next_open_fills_used_for_selection']:
                raise ValueError('SOURCE_PLAN_LOOKAHEAD')
            if intent['source_status'] not in {'OPEN_PLAN_VERIFIED','NOT_SCHEDULED','MANDATORY_EXITS_ONLY'}:
                return protect('SOURCE_PREOPEN_INTENT_UNAVAILABLE',source=source,source_status=intent['source_status'])
            intents[source]=intent
            self.input_hashes[str(path)]=sha256(path)
        covariance,attrs,risk_audit=self.risk(signal)
        rights=c['night']['unsettled_stock_rights']
        economic_held=set(c['night_holdings'])|set(rights)
        if economic_held-set(covariance.index):
            return protect('HELD_RISK_HISTORY_MISSING',missing_symbols=sorted(economic_held-set(covariance.index)))
        frozen=(economic_held & c['members'])-set(attrs.index)
        frozen|={s for s in exits if not c['sell'].get(s,False)}
        equity=c['night']['equity']
        night_weights={s:(c['night_holdings'].get(s,0)+rights.get(s,0))*c['last'][s]/equity for s in economic_held}
        component_weights={source:{s:float(weights[source])*w for s,w in intent['weights'].items() if s in attrs.index} for source,intent in intents.items()}
        mixed=pd.DataFrame(component_weights).fillna(0).sum(axis=1).to_dict()
        industries=attrs.industry.to_dict()
        for s in economic_held-set(industries):
            past=self.industry_history.get(s,pd.DataFrame(columns=['date','industry']))
            past=past.loc[past.date.le(signal)]
            if past.empty:
                return protect('HELD_INDUSTRY_MISSING',symbol=s)
            industries[s]=past.iloc[-1].industry
        stock_exposure=min(self.n4['stock_exposure'],1-c['night'].get('receivable_cash',0)/equity)
        benchmark=(attrs.total_mv/attrs.total_mv.sum()*stock_exposure).groupby(attrs.industry).sum().to_dict()
        try:
            targets,projection=constrain_mix(mixed,industries,benchmark,{s:night_weights[s] for s in frozen},stock_exposure,self.n4['max_stock_weight'],self.n4['industry_active_upper_planning'])
        except ValueError as error:
            if not str(error).startswith('LOCKED_'):
                raise
            return protect(str(error))
        desired={}
        desired_by_source={source:{} for source in sources.SOURCES}
        for s,w in targets.items():
            if s not in c['last'] or c['last'][s]<=0:
                return protect('TARGET_PRICE_MISSING',symbol=s)
            q=w*equity/c['last'][s]
            # Netting is audited in pre-corporate-action share units at the signal close.
            total=mixed.get(s,0)
            portions={source:(component_weights[source].get(s,0)/total if total>0 else (1. if source==sources.CORE else 0.)) for source in sources.SOURCES}
            for source,fraction in portions.items():
                desired_by_source[source][s]=q*fraction
            if s in frozen:
                desired[s]=held.get(s,0)
                continue
            for event in c['actions'].loc[c['actions'].symbol.eq(s)&c['actions'].ex_date.eq(pd.Timestamp(day))].itertuples():
                q*=1+event.stock_ratio
            q=max(0,q-c['snapshot']['unsettled_stock_rights'].get(s,0))
            desired[s]=int(np.floor(q/100+1e-9))*100
        netting=netting_audit(desired_by_source,self.ownership.get(method,{}),{s:c['night_holdings'].get(s,0)+rights.get(s,0) for s in economic_held},c['last'])
        feasible,cash,clips=physical_open_plan(desired,c['snapshot'],c['opened'],c['buy'],c['sell'],c['config'])
        post_rights=c['snapshot']['unsettled_stock_rights']
        values=pd.Series({s:(feasible.get(s,0)+post_rights.get(s,0))*c['opened'][s] for s in set(feasible)|set(post_rights)},dtype=float)
        actual=values/(cash+values.sum()+c['snapshot'].get('receivable_cash',0))
        actual_industry=actual.groupby(pd.Series(industries).reindex(actual.index)).sum()
        violation=float((actual_industry-pd.Series(benchmark).reindex(actual_industry.index,fill_value=0)-self.n4['industry_active_upper_execution']).max()) if len(actual) else 0.
        postcheck={'passed':bool(violation<=1e-5 and (not len(actual) or actual.max()<=self.n4['max_stock_weight']+1e-5)), 'maximum_industry_violation':violation,'maximum_stock_weight':float(actual.max()) if len(actual) else 0.,'industry_weights':actual_industry.to_dict(),'industry_limits':{s:benchmark.get(s,0)+self.n4['industry_active_upper_execution'] for s in actual_industry.index}}
        if not postcheck['passed']:
            return protect('INTEGER_OPEN_RISK_POSTCHECK',open_postcheck=postcheck)
        # Ownership is only an attribution of the actual shares, not extra executable books.
        ownership={}
        for s in feasible:
            portions={source:desired_by_source[source].get(s,0) for source in sources.SOURCES}
            if sum(portions.values())<=0:
                portions=self.ownership.get(method,{}).get(s,{sources.CORE:1.})
            ownership[s]=portions
        self.ownership[method]=ownership
        record('OPEN_PLAN_VERIFIED',allocation=allocation,risk_audit=risk_audit,projection=projection,netting=netting,target_weights=targets,target_quantities=desired,executable_quantities=feasible,open_clips=clips,open_postcheck=postcheck,next_open_used_for_selection=False)
        return feasible,None


def run(runs,bps,output,smoke=False):
    output.mkdir(parents=True,exist_ok=False)
    loaded=sources.load_inputs('evaluation')
    source_run=runs/f'evaluation-cost{bps}'
    allocation=runs/f'allocation-cost{bps}.json'
    planner=BlendPlanner(output,loaded,source_run,allocation)
    save(output/'contract.json',planner.n4)
    sources.old.engine.load_inputs=lambda:loaded
    sources.old.engine.KINDS=planner.n4['allocation']['methods']
    sources.old.engine.OUT=output
    hashes={str(p):sha256(p) for p in sources.old.source_files()+[Path(__file__),Path(sources.__file__),ROOT/'src/moneymore/research/n4_allocation.py',CONFIG,allocation]}
    save(output/'runtime.json',{'status':'RUNNING','source_hashes':hashes})
    sources.old.engine.run(bps,smoke=smoke,execution_planner=planner,config_overrides={'initial_cash':planner.n4['initial_cash'],'max_position_weight':planner.n4['max_stock_weight']})
    save(output/'completed.json',{'status':'COMPLETE','source_hashes':hashes,'intent_hashes':planner.input_hashes,'production_modified':False})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--bps',type=int,choices=[10,20],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    with threadpool_limits(limits=2):
        run(a.runs,a.bps,a.output,a.smoke)
