"""Six pre-registered simple-rule books using the existing A-share execution ledger."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import replay_ashare_rank_preserving as old
from prepare_four_strategy_2026 import save
from moneymore.research.simple_rules import conservative_targets,fcf_targets,cni_rebalance_days,timing_filter,financial_snapshot
from moneymore.research.mainline_execution import physical_open_plan
from moneymore.research.ashare_mainline import sha256

ROOT=old.ROOT
BASE=ROOT/'state/ashare-simple-rules-20261113'
DATA=BASE/'data-v1'
CONFIG=ROOT/'configs/ashare_simple_rules_v1.json'
CORE='doubleensemble_full--rank_cap'


class RulePlanner(old.RankPlanner):
    def __init__(self,output,loaded,include_fcf=False):
        super().__init__(output,json.loads(old.CONFIG.read_bytes()),loaded)
        self.spec=json.loads(CONFIG.read_bytes())
        self.official=loaded[7]
        self.sessions=loaded[2]
        self.pending={}
        self.cni_days=cni_rebalance_days(self.sessions)
        self.quarter_days={d for j,d in enumerate(self.sessions) if j==0 or pd.Timestamp(d).quarter!=pd.Timestamp(self.sessions[j-1]).quarter}
        panel=pd.read_parquet(ROOT/'state/ashare-mainline-20261112/panel-v2/panel.parquet',columns=['date','symbol','industry','ROC5'])
        announcements=pd.read_parquet(ROOT/'data/processed/fina_indicator.parquet',columns=['ts_code','ann_date']).dropna().drop_duplicates()
        announcements.ann_date=announcements.ann_date.astype(str)
        self.announcement_map={}
        self.adjusted={}
        self.invalid={}
        for signal in loaded[3].values():
            p=panel.loc[panel.date.eq(signal)].set_index('symbol')
            raw=p.ROC5.replace([np.inf,-np.inf],np.nan)
            ret=1/raw-1
            residual=ret-ret.groupby(p.industry).transform('mean')
            dates=[d for d in self.calendar if d<=signal][-5:]
            recent=set(announcements.loc[announcements.ann_date.between(dates[0],signal),'ts_code'])
            invalid=set(p.index[residual.isna()])|recent
            self.announcement_map[signal]=recent
            self.adjusted[signal]=residual
            self.invalid[signal]=invalid
            score=(-residual).replace([np.inf,-np.inf],np.nan).fillna(-1e9)
            score.loc[score.index.intersection(list(recent))]=-1e9
            self.scores[('raw_reversal',signal)]=raw.fillna(1.)
            self.scores[('adjusted_reversal',signal)]=score
            self.scores[('core_timing',signal)]=self.scores[('doubleensemble_full',signal)]
        monthly=[]
        for path in sorted(DATA.glob('monthly-*.parquet')):
            day=path.stem.split('-')[1]
            m=pd.read_parquet(path)
            adj=pd.read_parquet(DATA/f'adj_factor-{day}.parquet')
            m=m.merge(adj[['ts_code','adj_factor']],on='ts_code',validate='one_to_one')
            m['p']=m.close*m.adj_factor
            monthly.append(m[['ts_code','trade_date','p']])
        self.months=pd.concat(monthly).pivot(index='trade_date',columns='ts_code',values='p').sort_index()
        self.amount=loaded[0].pivot(index='trade_date',columns='ts_code',values='amount').sort_index()
        self.financial={}
        if include_fcf:
            status=json.loads((DATA/'financial-progress.json').read_bytes())
            if status['status']!='COMPLETE':raise ValueError('FINANCIAL_FETCH_NOT_COMPLETE')
            frames={n:pd.concat([pd.read_parquet(p) for p in DATA.glob(n+'-*.parquet')],ignore_index=True) for n in ['cashflow','fina_indicator','income']}
            for day in self.cni_days:
                signal=loaded[3][day]
                f=financial_snapshot(frames['cashflow'],frames['fina_indicator'],frames['income'],signal,self.spec['financial_report_max_age_days'])
                self.financial[signal]=f
                f.to_parquet(output/f'financial-visible-{signal}.parquet')

    def risk(self,signal):
        cov,attrs,audit=super().risk(signal)
        scope=json.loads((ROOT/'configs/ashare_n4_execution_scope_v1.json').read_bytes())
        excluded=[e['symbol'] for e in scope['events'] if e['available_from']<=signal and e['symbol'] in attrs.index]
        return cov,attrs.drop(index=excluded),{**audit,'known_merger_exclusions':excluded}

    def _record(self,c,**details):
        save(self.output/'decisions'/c['kind']/(c['day']+'.json'),{'signal_date':c['signal'],'execution_date':c['day'],'next_open_used_for_selection':False,**details})

    def _feasible(self,c,desired):
        return physical_open_plan(desired,c['snapshot'],c['opened'],c['buy'],c['sell'],c['config'])

    def _simple(self,c):
        held=c['held'];case=c['kind'].split('--')[0]
        exits=set(held)-c['members']
        scheduled=c['day'] in (self.quarter_days if case=='conservative_dividend' else self.cni_days)
        if not scheduled:
            feasible,_,clips=self._feasible(c,{s:q for s,q in held.items() if s not in exits})
            self._record(c,status='MANDATORY_EXITS_ONLY' if exits else 'NOT_SCHEDULED',open_clips=clips)
            return feasible,None
        attrs=self.attributes[c['signal']]
        members=set(attrs.index)&c['members']
        scope=json.loads((ROOT/'configs/ashare_n4_execution_scope_v1.json').read_bytes())
        members-={e['symbol'] for e in scope['events'] if e['available_from']<=c['signal']}
        basic=self.official[c['signal']]['daily_basic'].set_index('ts_code')
        if case=='conservative_dividend':
            weights,audit=conservative_targets(self.months,basic.dv_ttm,members,c['signal'])
            p=self.months.loc[self.months.index<=c['signal']].tail(37)
            complete=set(p.columns[p.notna().all()])&set(basic.index[basic.dv_ttm.notna()])
        else:
            finance=self.financial[c['signal']]
            history=self.amount.loc[self.amount.index<=c['signal']].tail(126)
            amount=history.mean().where(history.notna().sum().ge(100))
            weights,audit=fcf_targets(attrs.loc[sorted(members)],finance,amount)
            complete=set(finance.index[np.isfinite(finance[['fcf','netdebt','roe_stability','cash_conversion']]).all(axis=1)&finance.annual_ocf_complete])
        # Missing input of an existing member is frozen, not a data-induced exit.
        frozen=((set(held)&c['members'])-complete) | {s for s in exits if not c['sell'].get(s,False)}
        rights=c['night']['unsettled_stock_rights']
        equity=c['night']['equity']
        frozen_value=sum((c['night_holdings'].get(s,0)+rights.get(s,0))*c['last'][s] for s in frozen)
        reserve=c['night'].get('receivable_cash',0)/equity
        budget=max(0,min(.995,1-reserve)-frozen_value/equity)
        weights=weights.drop(index=list(frozen),errors='ignore')
        weights*=min(1.,budget/max(float(weights.sum()),1e-12))
        desired={s:held[s] for s in frozen}
        for symbol,weight in weights.items():
            q=weight*equity/c['last'][symbol]
            for event in c['actions'].loc[c['actions'].symbol.eq(symbol)&c['actions'].ex_date.eq(pd.Timestamp(c['day']))].itertuples():q*=1+event.stock_ratio
            q=max(0,q-c['snapshot']['unsettled_stock_rights'].get(symbol,0))
            desired[symbol]=int(q/100)*100
        feasible,cash,clips=self._feasible(c,desired)
        self._record(c,status='SIMPLE_RULE_PLAN_VERIFIED',coverage=audit,target_weights=weights.to_dict(),frozen_symbols=sorted(frozen),target_quantities=desired,executable_quantities=feasible,cash_after=cash,open_clips=clips)
        return feasible,None

    def _timing(self,c):
        scheduled=c['index']%20==0
        if scheduled or c['kind'] not in self.pending:
            result=super().__call__(**c)
            decision=json.loads((self.output/'decisions'/c['kind']/(c['day']+'.json')).read_bytes())
            if decision['status']!='OPEN_PLAN_VERIFIED':return result
            q=decision['target_quantities']
            rights=c['snapshot']['unsettled_stock_rights']
            self.pending[c['kind']]={'created':c['index'],'economic_target':{s:q.get(s,0)+rights.get(s,0) for s in set(q)|set(rights)},'last_adjusted':c['day']}
        state=self.pending[c['kind']]
        age=c['index']-state['created']
        if age>3:
            # No stale intent survives its finite timing window; ordinary exits remain live.
            desired={s:q for s,q in c['held'].items() if s in c['members']}
            feasible,_,clips=self._feasible(c,desired)
            self._record(c,status='NOT_SCHEDULED',open_clips=clips)
            return feasible,None
        economic=dict(state['economic_target'])
        if state['last_adjusted']!=c['day']:
            for event in c['actions'].loc[c['actions'].ex_date.eq(pd.Timestamp(c['day']))].itertuples():
                if event.symbol in economic:economic[event.symbol]*=1+event.stock_ratio
            state['economic_target']=dict(economic);state['last_adjusted']=c['day']
        desired={s:max(0,int(q-c['snapshot']['unsettled_stock_rights'].get(s,0))) for s,q in economic.items()}
        exits=set(c['held'])-c['members']
        desired={s:q for s,q in desired.items() if s in c['members']}
        screened,delayed=timing_filter(desired,c['held'],self.adjusted[c['signal']],age,exits,self.announcement_map[c['signal']])
        feasible,cash,clips=self._feasible(c,screened)
        # Timing is subordinate to the original industry and stock controls.
        _,attrs,_=self.risk(c['signal'])
        industry=attrs.industry.to_dict()
        for symbol in set(feasible)-set(industry):
            past=self.industry_history.get(symbol,pd.DataFrame(columns=['date','industry']))
            past=past.loc[past.date.le(c['signal'])]
            if not past.empty:industry[symbol]=past.iloc[-1].industry
        benchmark=(attrs.total_mv/attrs.total_mv.sum()*.995).groupby(attrs.industry).sum()
        def valid(q,cash):
            rights=c['snapshot']['unsettled_stock_rights']
            values=pd.Series({s:(q.get(s,0)+rights.get(s,0))*c['opened'][s] for s in set(q)|set(rights)})
            nav=cash+values.sum()+c['snapshot'].get('receivable_cash',0)
            w=values/nav
            sectors=w.groupby(pd.Series(industry)).sum()
            return bool((w.max() if len(w) else 0)<=.05+1e-5 and (sectors-benchmark.reindex(sectors.index,fill_value=0)-.10).max()<=1e-5)
        override=None
        if not valid(feasible,cash):
            feasible,cash,clips=self._feasible(c,desired)
            override='TIMING_BYPASSED_FOR_RISK'
            delayed=[]
            if not valid(feasible,cash):
                feasible,cash,clips=self._feasible(c,{s:q for s,q in c['held'].items() if s not in exits})
                override='OPTIONAL_BLOCKED_BY_OPEN_RISK'
        self._record(c,status='TIMING_PLAN_VERIFIED',age=age,delayed_symbols=delayed,override=override,target_quantities=desired,executable_quantities=feasible,open_clips=clips)
        return feasible,None

    def __call__(self,**c):
        case=c['kind'].split('--')[0]
        if case in ['conservative_dividend','fcf_rules']:return self._simple(c)
        if case=='core_timing':return self._timing(c)
        previous=self.contract
        self.contract={**previous,'rebalance_sessions':5 if case.endswith('reversal') else 20}
        try:
            result=super().__call__(**c)
        finally:self.contract=previous
        if case=='adjusted_reversal':
            additions={s for s,q in result[0].items() if q>c['held'].get(s,0)}
            if additions&self.invalid[c['signal']]:raise ValueError('INVALID_REVERSAL_SELECTED')
        return result


def run(output,bps,group,smoke=False):
    output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(CONFIG.read_bytes())
    save(output/'contract.json',spec)
    loaded=old.engine.load_inputs()
    planner=RulePlanner(output,loaded,group in ['all','fcf'])
    kinds=spec['cases'] if group=='all' else [s for s in spec['cases'] if (s.startswith('fcf_rules'))==(group=='fcf')]
    sources=list(dict.fromkeys(old.source_files()+[Path(__file__),CONFIG,ROOT/'src/moneymore/research/simple_rules.py']))
    hashes={str(p):sha256(p) for p in sources}
    for path in sources:
        dest=output/'sources'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
    inputs=[ROOT/'state/mature-multifactor/p1-native-20261103-a/bars-input.parquet',
        ROOT/'state/mature-multifactor/p1-native-20261103-a/membership.parquet',
        ROOT/'state/ashare-mainline-20261112/panel-v2/panel.parquet',
        ROOT/'data/processed/fina_indicator.parquet',ROOT/'data/processed/instruments.parquet']
    inputs+=list(DATA.glob('monthly-*.parquet'))+list(DATA.glob('adj_factor-*.parquet'))
    if group in ['all','fcf']:
        for endpoint in ['cashflow','fina_indicator','income']:inputs+=list(DATA.glob(endpoint+'-*.parquet'))
    save(output/'inputs.json',{'sha256':{str(p):sha256(p) for p in inputs},'read_only_inputs':True})
    save(output/'runtime.json',{'status':'RUNNING','pid':os.getpid(),'bps':bps,'group':group,'source_hashes':hashes})
    old.engine.load_inputs=lambda:loaded
    old.engine.OUT=output;old.engine.KINDS=kinds
    try:
        old.engine.run(bps,smoke=smoke,execution_planner=planner,config_overrides={'initial_cash':10_000_000,'max_position_weight':.05})
        save(output/'completed.json',{'status':'COMPLETE','bps':bps,'group':group,'source_hashes':hashes,'production_modified':False})
    except Exception as e:
        save(output/'failure.json',{'status':'FAILED','error_type':type(e).__name__,'error':str(e)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--bps',type=int,choices=[10,20],required=True);p.add_argument('--group',choices=['all','nonfcf','fcf'],default='all');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=2):run(a.output.resolve(),a.bps,a.group,a.smoke)
