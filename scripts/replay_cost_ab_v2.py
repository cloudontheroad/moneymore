"""Four frozen strategies, Jan-Sep 2026; isolated PaperBroker accounts only."""
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import json,time,sqlite3,hashlib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits
from moneymore.data.store import ParquetStore
from moneymore.config import BacktestConfig
from moneymore.execution.paper import PaperBroker,ExecutionBar
from moneymore.research.replay_actions import official_replay_actions,economic_snapshot
from moneymore.research.period_statistics import period_statistics
from moneymore.research.integer_execution import plan_integer_execution,paper_intents_from_integer_plan
from moneymore.research.quote_permissions import quote_permissions
from moneymore.research.mature_multifactor import current_risk_inputs,topk_dropout_targets
from moneymore.research.enhanced_indexing import optimize_snapshot,EnhancedPolicy
from moneymore.research.baseline_v2_model import build_v2_snapshot
from moneymore.research.baseline_v2_daily_readiness import construct_daily_targets
from moneymore.research.sector_model import build_global_factor_snapshot
from moneymore.portfolio_constructor import global_topk_portfolio,trailing_return_correlation
from moneymore.challenger_diagnostics import trading_dates
from prepare_four_strategy_2026 import ROOT,OUT as STUDY,save
from moneymore.research.cost_rebalance import cost_targets
from moneymore.research.cost_aware import risk_compliant
from moneymore.research.cost_execution import finalize_cost_decision,holding_plan
from moneymore.research.cost_integer import plan_integer_execution as cost_integer
from cost_study_quotes import ensure_quotes
from moneymore.research.cost_risk import cost_risk_inputs
OUT=ROOT/'state/cost-aware-ab-v2'

KINDS=['a_reference','b_reference','a_cost','b_exec','b_cost','b_cost_exec']

class StudyStore:
    def __init__(self):
        frozen=ROOT/'state/cost-aware-ab-v2/frozen-data'
        self.base=ParquetStore(frozen if (frozen/'processed/fina_indicator_research_v2.parquet').exists() else ROOT/'data');self.curated=self.base.curated;self.cache={};self.day='20251231';self.basic=[]
    def read(self,table,columns=None,filters=None):
        if table not in self.cache:self.cache[table]=self.base.read(table)
        frame=self.cache[table]
        if table=='daily_basic' and self.basic:
            frame=pd.concat([frame,*self.basic],ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')
        if 'trade_date' in frame:frame=frame.loc[frame.trade_date.astype(str).le(self.day)]
        for key,op,value in filters or []:
            if op=='in':frame=frame.loc[frame[key].isin(value)]
            elif op in ('=','=='):frame=frame.loc[frame[key].eq(value)]
            elif op=='<':frame=frame.loc[frame[key].lt(value)]
            elif op=='<=':frame=frame.loc[frame[key].le(value)]
            elif op=='>':frame=frame.loc[frame[key].gt(value)]
            elif op=='>=':frame=frame.loc[frame[key].ge(value)]
            else:raise ValueError('UNSUPPORTED_FILTER')
        return frame[columns].copy() if columns else frame.copy()

def wait_files(day):
    names=[STUDY/'official'/(endpoint+'-'+day+'.parquet') for endpoint in ['daily_basic','stk_limit','suspend_d','dividend']]
    while not all(p.exists() for p in names):
        if (STUDY/'data-results.json').exists():raise ValueError('OFFICIAL_INPUT_FETCH_INCOMPLETE:'+day)
        save(OUT/'replay-progress.json',{'status':'WAITING_FOR_OFFICIAL_INPUTS','waiting_date':day})
        time.sleep(15)
    return [pd.read_parquet(p) for p in names]

def run(strategy=None,smoke=False):
    if not (STUDY/'model/results.json').exists():raise ValueError('PRE2026_FROZEN_MODEL_REQUIRED')
    kinds=[strategy] if strategy else KINDS
    if strategy is None:raise ValueError('ONE_FIXED_COMPONENT_CASE_REQUIRED')
    cost_mode=strategy in ['a_cost','b_cost','b_cost_exec']
    execution_fix=strategy in ['a_cost','b_exec','b_cost_exec']
    base_kind='raw' if strategy.startswith('a_') else 'balanced'
    calibration=json.loads((OUT/'t2/historical.json').read_bytes())
    contract_path=OUT/'t5-frozen-inputs.json'
    if not smoke:
        contract=json.loads(contract_path.read_bytes())
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest for p,digest in contract['hashes'].items()):raise ValueError('T5_FROZEN_INPUT_OR_POLICY_CHANGED')
    output=OUT/('t5-smoke2' if smoke else 't5')/strategy;output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():print('ALREADY_COMPLETE');return
    store=StudyStore();source=ROOT/'state/mature-multifactor/p1-native-20261103-a'
    bars=pd.read_parquet(source/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    # Proven exit quotes are execution/valuation inputs, not fabricated scores.
    extra=OUT/'official/held-exit-002405-202609.parquet'
    if extra.exists():
        evidence=json.loads(extra.with_suffix('.json').read_bytes())
        assert hashlib.sha256(extra.read_bytes()).hexdigest()==evidence['sha256']
        supplement=pd.read_parquet(extra)
        bars=pd.concat([bars,supplement],ignore_index=True).drop_duplicates(['trade_date','ts_code'],keep='first')
    quotes=bars.set_index(['trade_date','ts_code']);prices=bars.assign(price=bars.close*bars.adj_factor).pivot(index='trade_date',columns='ts_code',values='price');prices.index=pd.to_datetime(prices.index)
    membership=pd.read_parquet(source/'membership.parquet');membership.datetime=pd.to_datetime(membership.datetime)
    study_assets=set(membership.instrument)
    predictions=pd.read_parquet(STUDY/'model/predictions.parquet').score
    instruments=store.read('instruments').drop_duplicates('ts_code').set_index('ts_code')
    candidate=json.loads((ROOT/'state/factor-baseline-candidates/20261029.json').read_bytes());policy=candidate['factor_policy']
    weight_source=Path(candidate['publication']['source_weights']) if 'publication' in candidate and 'source_weights' in candidate['publication'] else ROOT/'state/baseline-v2-release/corrected-refits/20261029T092148390912Z/state/baseline-v2-validation/weight-validation.json'
    weight_report=json.loads(weight_source.read_bytes());assert weight_report['protocol']['as_of']=='20251231'
    coefficients=weight_report['diagnostic_coefficients_not_released'];assert coefficients==policy['experimental_linear_model']['coefficients']
    build_policy={k:v for k,v in policy.items() if k not in ['experimental_linear_model','input_overlay','validated_linear_model']}
    selection=yaml.safe_load((ROOT/'configs/sector_models.yaml').read_text(encoding='utf-8'))['global_selection']
    config=BacktestConfig.model_validate({**BacktestConfig.from_yaml(ROOT/'configs/default.yaml').model_dump(),'initial_cash':10_000_000,'slippage_bps':10,'max_position_weight':.15})
    protocol={'window':['20260101','20260930'],'initial_cash':config.initial_cash,'slippage_bps_each_side':10,
        'annualization':252,'annual_risk_free':0,'commission_rate':config.commission_rate,'minimum_commission':config.minimum_commission,
        'stamp_duty_rate':config.stamp_duty_rate,'transfer_fee_rate':config.transfer_fee_rate,
        'same_historical_membership':True,'timeline':'previous_close_target_then_next_open_execution_for_all_six_cases',
        'component':strategy,'cost_mode':cost_mode,'execution_fix':execution_fix,'calibration':calibration,'membership_limitation':'existing snapshot history includes 2026 carried intervals; not original-vintage daily Top1000',
        'weight_source':str(weight_source),'v2_training_cutoff':'20251231','real_orders_enabled':False,
        'platform_outages_and_live_account_circuit_breakers':'not simulated; compare frozen strategy constructors under common clean execution'}
    save(output/'protocol.json',protocol)
    calendar=trading_dates(store.base,'20260930');sessions=[d for d in calendar if '20260101'<=d<='20260930']
    if smoke:sessions=sessions[:3]
    books={};histories={k:[] for k in kinds};events={k:[] for k in kinds}
    checkpoint=output/'checkpoint.json'
    for kind in kinds:
        directory=output/kind;directory.mkdir(exist_ok=True);database=directory/'paper.sqlite3'
        if checkpoint.exists():
            with sqlite3.connect(directory/'resume.sqlite3') as old,sqlite3.connect(database) as new:old.backup(new)
        elif database.exists():
            blank=PaperBroker(database)
            assert not blank.fills('isolated') and not blank.account_snapshot({},'isolated')['positions'], 'PARTIAL_FIRST_DAY_REQUIRES_NEW_RUN_DIRECTORY'
        books[kind]=PaperBroker(database);books[kind].initialize_account(config.initial_cash,'isolated')
    completed=0
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_bytes());histories=saved['histories'];events=saved['events'];completed=saved['completed']
    else:
        for kind,broker in books.items():
            with sqlite3.connect(broker.database) as old,sqlite3.connect(output/kind/'resume.sqlite3') as new:old.backup(new)
        save(checkpoint,{'completed':0,'histories':histories,'events':events})
    # Warm-up metadata needed for a strictly prior-day baseline valuation.
    for d in ['20251230','20251231']:store.basic.append(wait_files(d)[0])
    # Empty initial holdings: only official record-date events in the replay window
    # can create rights. Do not mix unrelated legacy action rows into this ledger.
    source_actions=pd.DataFrame();proofs=[];last_marks=quotes.loc['20251231'].close.dropna().to_dict()
    for i,day in enumerate(sessions):
        signal=calendar[calendar.index(day)-1];basic,gate,susp,div=wait_files(day)
        store.basic.append(basic);proofs.append(susp)
        if not div.empty:source_actions=pd.concat([source_actions,div],ignore_index=True).drop_duplicates()
        if i<completed:
            last_marks.update(quotes.loc[day].close.dropna().to_dict());continue
        store.day=signal;members=set(membership.loc[membership.datetime.eq(pd.Timestamp(signal)),'instrument'])
        if len(members)!=1000:raise ValueError('INCOMPLETE_SHARED_MEMBERSHIP:'+signal)
        universe=instruments.reindex(sorted(members)).copy();universe['symbol']=universe.index;universe['effective_date']=signal
        universe['board']=['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in universe.index]
        if 'baseline' in kinds:
            baseline=build_global_factor_snapshot(store,sorted(members),selection['factors'])
        if 'v2' in kinds:
            v2_input_fault=None
            try:
                with patch('moneymore.research.baseline_v2_model.active_strategy_universe',return_value=universe.reset_index(drop=True)):
                    _,variants,_,_=build_v2_snapshot(ROOT,store,signal,policy=build_policy)
                v2=variants['industry_size_comparable'].copy();v2['score']=v2[list(coefficients)].to_numpy()@np.array(list(coefficients.values()))
            except ValueError as error:
                if str(error) not in {'EMPTY_OR_DUPLICATE_V2_CROSS_SECTION','INSUFFICIENT_V2_CROSS_SECTION'}:raise
                # Preserve the existing policy's hold, e.g. calendar-age expiry
                # of previous-session metadata after the long Spring Festival.
                # Do not loosen eligibility or remove this session from NAV.
                v2_input_fault=str(error);v2=pd.DataFrame(columns=['date','symbol','score'])
        signal_scores=predictions.xs(pd.Timestamp(signal),level='datetime').reindex(sorted(members)).dropna()
        market_open=quotes.loc[day].open.dropna().to_dict();close=quotes.loc[day].close.dropna().to_dict()
        actions=official_replay_actions(source_actions.loc[source_actions.ts_code.isin(study_assets)] if not source_actions.empty else source_actions)
        halted=set(susp.loc[susp.suspend_type.eq('S')&(susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code']) if not susp.empty else set()
        for symbol in halted-set(market_open):
            if symbol not in last_marks:continue
            mark=last_marks[symbol]
            for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():mark=(mark-action.cash_per_share)/(1+action.stock_ratio)
            market_open[symbol]=mark;close[symbol]=mark
        buy,sell=quote_permissions(market_open,halted,gate.set_index('ts_code'),10)
        for kind,broker in books.items():
            night=economic_snapshot(broker,broker.account_snapshot(last_marks,'isolated'),last_marks,'isolated',signal,actions)
            night_held={p['symbol']:p['quantity'] for p in night['positions'] if p['quantity']>0}
            night_values=pd.Series({s:q*last_marks[s] for s,q in night_held.items()},dtype=float)
            for s,q in night['unsettled_stock_rights'].items():night_values[s]=night_values.get(s,0)+q*last_marks[s]
            broker.settle_corporate_actions('isolated',day);broker.settle_available_positions('isolated',day)
            snap=economic_snapshot(broker,broker.account_snapshot(market_open,'isolated'),market_open,'isolated',day,actions)
            held={p['symbol']:int(p['quantity']) for p in snap['positions'] if p['quantity']>0};available={p['symbol']:int(p['available_quantity']) for p in snap['positions']}
            missing=set(held)-set(market_open)|set(held)-set(close)
            if missing:
                bars=ensure_quotes(ROOT,bars,missing,day);quotes=bars.set_index(['trade_date','ts_code'])
                market_open.update(quotes.loc[day].open.dropna().to_dict());close.update(quotes.loc[day].close.dropna().to_dict())
                if set(held)-set(market_open) or set(held)-set(close):raise ValueError('UNPROVEN_HELD_MARK_MISSING:'+day)
            event={'date':day,'signal_date':signal,'status':'UNPREPARED'};targets=None
            old_values=night_values;benchmark=None;economics=None
            execution_styles=pd.DataFrame({'execution_only':0.},index=sorted(members|set(held)));limits={'execution_only':0.}
            if kind=='baseline':
                scores=baseline.loc[baseline.symbol.isin(members)]
                if (set(held)&members)-set(scores.symbol):event.update(status='INPUT_HOLD',reason='MEMBER_SCORE_MISSING')
                else:
                    _,targets,_=global_topk_portfolio(scores,set(held),symbol_column='symbol',top_k=selection['top_k'],exit_rank=selection['exit_rank'],
                        max_replacements=selection['max_replacements'],minimum_weight=selection['minimum_weight'],maximum_weight=selection['maximum_weight'],
                        correlation=trailing_return_correlation(prices,pd.Timestamp(signal),selection['correlation_lookback']),
                        correlation_penalty=selection['correlation_penalty'],cluster_correlation_threshold=selection['cluster_correlation_threshold'],maximum_cluster_members=selection['maximum_cluster_members'])
            elif kind=='v2':
                if v2_input_fault:event.update(status='INPUT_HOLD',reason=v2_input_fault)
                elif (set(held)&members)-set(v2.symbol):event.update(status='INPUT_HOLD',reason='V2_MEMBER_SCORE_MISSING')
                else:_,targets,_=construct_daily_targets(v2,held,prices,signal,day,policy,universe_members=members)
            else:
                risk_scores=signal_scores.reindex(sorted(set(signal_scores.index)|set(held)))
                risk_builder=cost_risk_inputs if execution_fix else current_risk_inputs
                returns,size,raw,controls,benchmark,limits_current,audit=risk_builder(store,bars,risk_scores,signal,
                    suspension_proof=pd.concat(proofs,ignore_index=True),candidate_members=members)
                eligible=signal_scores.reindex(controls.index.intersection(signal_scores.index))
                exits=set(held)-members
                frozen_forecast=(set(held)&members)-set(eligible.index)
                if set(held)-set(controls.index) or (frozen_forecast and not execution_fix) or (snap['unsettled_stock_rights'] and not execution_fix):
                    event.update(status='INPUT_HOLD',reason='HELD_RISK_OR_SCORE_MISSING')
                elif cost_mode:
                    selected_styles=execution_styles if base_kind=='raw' else controls
                    selected_limits=limits if base_kind=='raw' else limits_current
                    economics=cost_targets(eligible,old_values,night['equity'],returns,selected_styles,selected_limits,benchmark,config,calibration,base_kind,exits=exits-halted,frozen=frozen_forecast|(set(held)&halted),size=size)
                    save(output/kind/(day+'-economic.json'),economics)
                    if economics['status']=='PASS_COST_CONTINUOUS':
                        targets=economics['weights'];execution_styles=selected_styles;limits=selected_limits
                        if base_kind=='raw':benchmark=None
                    else:event.update(status='OPTIMIZER_HOLD',reason=economics['status'])
                elif base_kind=='raw':
                    weights=old_values/night['equity'];locked=weights.reindex(sorted(exits&halted)).dropna();free=1-float(locked.sum())
                    targets={s:w*free for s,w in topk_dropout_targets(eligible,weights.drop(index=list(exits),errors='ignore')/free,topk=50-len(locked)).items()};targets.update(locked.to_dict());benchmark=None
                else:
                    signal_risk=risk_scores.reindex(controls.index);signal_risk.loc[list(exits&set(signal_risk.index))]=0.
                    if execution_fix:signal_risk.loc[list(frozen_forecast)]=0.
                    solved=optimize_snapshot(signal_risk,returns,size,old_values/old_values.sum() if len(old_values) else old_values,
                        EnhancedPolicy(objective_preconditioning=True),style_controls=controls,style_limits=limits_current,benchmark_weights=benchmark,
                        controls_prestandardized=True,allow_cash_bootstrap=True,reject_turnover_relaxation=True,forced_sell_symbols=exits-halted,forced_hold_symbols=(set(held)&halted)|(frozen_forecast if execution_fix else set()),solver_max_iters=1000)
                    save(output/kind/(day+'-continuous.json'),solved);targets=solved.get('weights')
                    execution_styles=controls;limits=limits_current
                    if not targets:event.update(status='OPTIMIZER_HOLD',reason=solved.get('failures'))
            if targets:
                planner=cost_integer if execution_fix or cost_mode else plan_integer_execution
                kwargs={}
                if execution_fix:kwargs.update(accept_validated_incumbent=True,locked_shares=snap['unsettled_stock_rights'],forced_sell_symbols=exits-halted)
                elif cost_mode:kwargs.update(forced_sell_symbols=exits-halted)
                if planner==cost_integer and base_kind=='balanced':kwargs['stock_active_bound']=max(0.,.15-float(benchmark.max()))
                if cost_mode:kwargs.update(economic_tracking_weights=economics['economic_tracking_weights'])
                if cost_mode and base_kind=='balanced':kwargs['research_objective_scale']=10000
                if execution_fix and frozen_forecast:kwargs['fixed_quantities']={s:held[s] for s in frozen_forecast}
                economic_quantities=pd.Series(held,dtype=float).add(pd.Series(snap['unsettled_stock_rights'],dtype=float),fill_value=0)
                current_values={s:q*market_open[s]/snap['equity'] for s,q in economic_quantities.items() if q>0}
                if not held and strategy not in ['a_reference','b_reference'] and not smoke:
                    reference='a_reference' if base_kind=='raw' else 'b_reference'
                    shared=OUT/'t5'/reference/reference/(day+'-integer.json')
                    deadline=time.monotonic()+300
                    while not shared.exists() and time.monotonic()<deadline:time.sleep(1)
                    if not shared.exists():raise ValueError('SHARED_INITIAL_CONFIGURATION_NOT_READY')
                    integer=json.loads(shared.read_bytes())
                    if integer['status']!='PASS_REFERENCE_QUOTES_ONLY':raise ValueError('SHARED_INITIAL_CONFIGURATION_FAILED')
                    integer.update(shared_initial_configuration=str(shared),shared_initial_sha256=hashlib.sha256(shared.read_bytes()).hexdigest())
                elif cost_mode and economics.get('preserve_quantities') and risk_compliant(current_values,execution_styles,limits,benchmark):
                    integer=holding_plan(market_open,held,snap['cash'],snap['equity'],execution_styles,limits,benchmark,locked_shares=snap['unsettled_stock_rights'])
                else:
                    integer=planner(targets,market_open,held,available,snap['cash'],snap['equity'],config,execution_styles,limits,
                        can_buy=buy,can_sell=sell,benchmark_weights=benchmark,style_controls_prestandardized=True,time_limit=60,solver_threads=4,mip_relative_gap=.01,research_secondary_time_limit=10,**kwargs)
                if cost_mode:
                    integer=finalize_cost_decision(integer,market_open,held,snap['cash'],snap['equity'],execution_styles,limits,benchmark,economics,forced_exit=exits-halted,locked_shares=snap['unsettled_stock_rights'])
                save(output/kind/(day+'-integer.json'),integer);event.update(status=integer['status'],reason=integer.get('reason'),cost_decision=integer.get('cost_decision'))
                if integer['status']=='PASS_REFERENCE_QUOTES_ONLY':
                    for intent in paper_intents_from_integer_plan(integer,'nine_month_'+kind,signal):
                        assert broker.submit(intent,'isolated')=='PENDING'
                    for order in integer['orders']:
                        symbol=order['symbol'];broker.execute_pending(ExecutionBar(symbol,day,market_open[symbol],market_open[symbol],buy.get(symbol,False),sell.get(symbol,False)),config,'isolated')
                    actual=broker.account_snapshot(market_open,'isolated')
                    assert {p['symbol']:int(p['quantity']) for p in actual['positions'] if p['quantity']>0}=={s:q for s,q in integer['target_quantities'].items() if q>0}
                    assert abs(actual['cash']-integer['estimated_cash'])<.01
                    event['orders']=len(integer['orders'])
            owned={p['symbol'] for p in broker.account_snapshot(close,'isolated')['positions'] if p['quantity']>0}
            for action in actions.loc[actions.symbol.isin(owned)&actions.record_date.eq(pd.Timestamp(day))].itertuples():
                broker.register_corporate_entitlement(account_id='isolated',symbol=action.symbol,action_key=action.action_key,record_date=day,cash_per_share=action.cash_per_share,stock_ratio=action.stock_ratio,
                    cash_pay_date=None if pd.isna(action.cash_pay_date) else action.cash_pay_date.strftime('%Y%m%d'),stock_list_date=None if pd.isna(action.stock_list_date) else action.stock_list_date.strftime('%Y%m%d'))
            final=economic_snapshot(broker,broker.account_snapshot(close,'isolated'),close,'isolated',day,actions)
            assert broker.reconcile('isolated').matched
            histories[kind].append({'date':day,'equity':final['equity'],'cash':final['cash'],'gross_exposure':final['market_value']/final['equity']})
            events[kind].append(event);pd.DataFrame(histories[kind]).to_parquet(output/kind/'equity.parquet',index=False)
        last_marks.update(close)
        for kind,broker in books.items():
            with sqlite3.connect(broker.database) as old,sqlite3.connect(output/kind/'resume.sqlite3') as new:old.backup(new)
        save(checkpoint,{'completed':i+1,'histories':histories,'events':events})
        save(output/'progress.json',{'status':'RUNNING','completed_sessions':i+1,'total_sessions':len(sessions),'latest_date':day,'accounts':{k:events[k][-1] for k in kinds}})
        print(json.dumps({'date':day,'completed':i+1,'accounts':{k:events[k][-1]['status'] for k in kinds}}),flush=True)
    results=[]
    for kind,broker in books.items():
        frame=pd.DataFrame(histories[kind]);fills=broker.fills('isolated')
        results.append({'strategy':kind,**period_statistics(frame.equity,config.initial_cash),
            'fills':len(fills),'fees':sum(float(r['fee']) for r in fills),'mean_gross_exposure':float(frame.gross_exposure.mean()),
            'executed_rebalances':sum(r['status']=='PASS_REFERENCE_QUOTES_ONLY' for r in events[kind]),
            'hold_sessions':sum(r['status']!='PASS_REFERENCE_QUOTES_ONLY' for r in events[kind])})
    save(output/'results.json',{'status':'COMPLETED_RETROSPECTIVE_FROZEN_PRE2026_NOT_LIVE_EVIDENCE','window':['20260101','20260930'],'smoke_only':smoke,
        'cases':results,'protocol':protocol,'real_orders_enabled':False,'production_ledgers_modified':False})
    print(json.dumps(results),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--strategy',choices=KINDS);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.strategy,args.smoke)
