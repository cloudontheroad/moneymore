"""Opt-in cost-policy paper branches; immutable legacy model/runtime untouched."""
from datetime import datetime,UTC
from pathlib import Path
import hashlib,json,sqlite3
import numpy as np
import pandas as pd

VERSION='cost_aware_ab_v2'

def reserve_execution_attempt(root,tag,day,timestamp):
    """Persistent bounded compensation; process restarts cannot reset the budget."""
    from filelock import FileLock
    from .challenger_diagnostics import write_diagnostic
    path=root/'state/cost-aware-ab-v2/execution-attempts'/tag/(day+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+'.lock',timeout=1):
        state=json.loads(path.read_bytes()) if path.exists() else {'attempts':0,'last_attempt':None}
        # A documented maintenance extension can add at most one further
        # three-attempt tranche. Never reset consumed attempts or cooldown.
        maximum=min(6,max(3,int(state.get('max_attempts',3))))
        if state['attempts']>=maximum:return 'COST_EXECUTION_BUDGET_EXHAUSTED'
        if state['last_attempt'] is not None and timestamp-state['last_attempt']<300:return 'COST_EXECUTION_COMPENSATION_COOLDOWN'
        write_diagnostic(path,{**state,'attempts':state['attempts']+1,'last_attempt':timestamp,'max_attempts':maximum,'cooldown_seconds':300})
    return None

def account_actions(store,broker,account):
    from .research.replay_actions import official_replay_actions
    with sqlite3.connect(broker.database) as db:
        symbols={r[0] for r in db.execute('SELECT symbol FROM positions WHERE account_id=? UNION SELECT symbol FROM corporate_entitlements WHERE account_id=?',(account,account))}
    try:raw=store.read('dividend')
    except FileNotFoundError:
        if symbols:raise ValueError('COST_CORPORATE_ACTION_REFERENCE_MISSING')
        raw=pd.DataFrame()
    if not raw.empty:raw=raw.loc[raw.ts_code.isin(symbols)]
    return official_replay_actions(raw)

def cost_candidates(root):
    from .execution.paper import PaperBroker
    from .candidate_lifecycle import lifecycle_rows,ARCHIVED
    path=root/'state/cost-aware-ab-v2/registry.json'
    if not path.exists():return []
    registry=json.loads(path.read_bytes())
    if registry.get('schema')!='COST_AWARE_PAPER_BRANCH_V1' or registry.get('real_orders_enabled') is not False:raise ValueError('INVALID_COST_BRANCH_SCOPE')
    valid=all(Path(p).resolve().is_relative_to(root.resolve()) and Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in registry['evidence_hashes'].items())
    broker=PaperBroker(root/'state/paper_orders.sqlite3');states=lifecycle_rows(root);rows=[]
    for entry in registry['accounts']:
        tag=entry['candidate_tag'];reports=sorted((root/'state/qlib-candidate-shadow'/tag).glob('*.json'))
        latest=json.loads(reports[-1].read_bytes()) if reports else {'status':'AWAITING_FIRST_PLAN','portfolio':broker.account_snapshot({},entry['account_id'])}
        executions=sorted((root/'state/qlib-portfolio-execution'/tag).glob('*.json'))
        execution=json.loads(executions[-1].read_bytes()) if executions else {}
        latest['portfolio']['initial_cash']=entry['initial_cash']
        rows.append({**entry,'cost_policy_version':VERSION,'model_dir':registry['model_dir'],'model_manifest':registry['model_manifest'],
            'calibration_path':registry['calibration_path'],'protocol_version':VERSION,'data_cutoff':registry['data_cutoff'],
            'research_path':str(path),'research_gate_passed':valid,'research_gate_failures':[] if valid else ['COST_POLICY_EVIDENCE_CHANGED'],
            'validation_hold':not valid,'research_gate_scope':'PAPER_COST_PROTECTION_NOT_PROFIT_PROOF',
            'latest_report':latest,'latest_portfolio':latest['portfolio'],'latest_status':latest['status'],'latest_trade_date':latest.get('trade_date'),
            'latest_execution':execution,
            'observation_days':sum(p.stem>=entry['activation_date'] for p in reports),'lifecycle':states.get(tag,{}),
            'archived':states.get(tag,{}).get('status') in ARCHIVED,'rank_ic':None,'rank_ic_ir':None,
            'cost_adjusted_top_k_excess_return':None,'promotion_eligible':False,
            'alpha_evidence_enabled':registry['alpha_evidence_enabled']})
    return rows


def prepare_cost_daily(*,root,store,broker,config,candidate,trade_date,input_root=None):
    from .mature_multifactor_runtime import infer
    from .research.cost_risk import cost_risk_inputs as current_risk_inputs
    from .research.cost_rebalance import cost_targets
    from .research.replay_actions import official_replay_actions,economic_snapshot
    from .research.style_v2_experiment import candidate_execution_config,execution_policy_hash
    from .strategy_universe import active_strategy_universe
    from .challenger_diagnostics import trading_dates,write_diagnostic
    from .qlib_challenger_daily import _finish,_record_account_history
    if trade_date<candidate['activation_date'] or candidate.get('validation_hold'):raise ValueError('COST_BRANCH_NOT_ACTIVE')
    account=candidate['account_id'];tag=candidate['candidate_tag'];config=candidate_execution_config(config,candidate)
    input_root=input_root or root
    scores,official,features=infer(input_root,candidate,trade_date)
    calibration=json.loads(Path(candidate['calibration_path']).read_bytes())
    if str(calibration['label_cutoff'])>trade_date:raise ValueError('CALIBRATION_NOT_YET_KNOWN')
    bars=pd.read_parquet(features/'bars-input.parquet');bars.trade_date=bars.trade_date.astype(str)
    marks=bars.loc[bars.trade_date.eq(trade_date)].set_index('ts_code').close.to_dict()
    broker.settle_corporate_actions(account,trade_date);broker.settle_available_positions(account,trade_date)
    held={p['symbol']:int(p['quantity']) for p in broker.account_snapshot({},account)['positions'] if p['quantity']>0}
    actions=account_actions(store,broker,account)
    for action in actions.loc[actions.record_date.eq(pd.Timestamp(trade_date))].itertuples():
        broker.register_corporate_entitlement(account_id=account,symbol=action.symbol,action_key=action.action_key,record_date=trade_date,
            cash_per_share=action.cash_per_share,stock_ratio=action.stock_ratio,cash_pay_date=None if pd.isna(action.cash_pay_date) else action.cash_pay_date.strftime('%Y%m%d'),
            stock_list_date=None if pd.isna(action.stock_list_date) else action.stock_list_date.strftime('%Y%m%d'))
    members=set(active_strategy_universe(store,trade_date).symbol);exits=set(held)-members
    proofs=[pd.read_parquet(p/'suspend_d.parquet') for p in (input_root/'state/mature-multifactor/runtime/official').glob('*') if (p/'results.json').exists()]
    proof=pd.concat(proofs,ignore_index=True) if proofs else pd.DataFrame()
    full=proof.loc[proof.trade_date.astype(str).eq(trade_date)&proof.suspend_type.eq('S')&(proof.suspend_timing.isna()|proof.suspend_timing.eq(''))] if not proof.empty else proof
    halted=set(full.ts_code) if not full.empty else set()
    for s in set(held)-set(marks):
        if s not in halted:raise ValueError('COST_HELD_CLOSE_UNOBSERVED')
        old=bars.loc[bars.ts_code.eq(s)&bars.trade_date.lt(trade_date)].sort_values('trade_date')
        if old.empty:raise ValueError('COST_HELD_LAST_MARK_UNKNOWN')
        marks[s]=float(old.iloc[-1].close)
        for action in actions.loc[actions.symbol.eq(s)&actions.ex_date.eq(pd.Timestamp(trade_date))].itertuples():marks[s]=(marks[s]-action.cash_per_share)/(1+action.stock_ratio)
    snapshot=economic_snapshot(broker,broker.account_snapshot(marks,account),marks,account,trade_date,actions)
    economic_held=set(held)|set(snapshot['unsettled_stock_rights'])
    risk_scores=scores.reindex(sorted((set(scores.index)&members)|economic_held))
    returns,size,raw,controls,benchmark,limits,audit=current_risk_inputs(store,bars,risk_scores,trade_date,suspension_proof=proof,candidate_members=members,basic_overlay=pd.read_parquet(official/'daily_basic.parquet'))
    if economic_held-set(controls.index):raise ValueError('COST_HELD_RISK_MISSING_NOT_EXIT')
    eligible=scores.reindex(controls.index.intersection(scores.index).intersection(pd.Index(sorted(members))))
    from .research.prediction_quality import prediction_quality
    if not prediction_quality(eligible.to_numpy())['passed']:raise ValueError('COST_PREDICTION_QUALITY_BLOCKED')
    frozen=(set(held)&members)-set(eligible.index)
    q=pd.Series(held,dtype=float).add(pd.Series(snapshot['unsettled_stock_rights'],dtype=float),fill_value=0)
    values=pd.Series({s:n*marks[s] for s,n in q.items() if n>0},dtype=float)
    raw_kind=candidate['portfolio_variant']=='mature_alpha158_raw'
    styles=pd.DataFrame({'execution_only':0.},index=controls.index) if raw_kind else controls
    limits={'execution_only':0.} if raw_kind else limits
    economics=cost_targets(eligible,values,snapshot['equity'],returns,styles,limits,benchmark,config,calibration,'raw' if raw_kind else 'balanced',exits=exits-halted,frozen=frozen|(set(held)&halted),size=size)
    if economics['status']!='PASS_COST_CONTINUOUS':raise ValueError('COST_CONTINUOUS_BLOCKED:'+economics['status'])
    targets=economics['weights'];calendar=trading_dates(store,trade_date);following=calendar[calendar.index(trade_date)+1]
    plan={'candidate_tag':tag,'account_id':account,'signal_date':trade_date,'created_at':datetime.now(UTC).isoformat(),
        'strategy_id':candidate['strategy_id'],'model_id':candidate['model_id'],'target_weights':targets,'styles':styles.to_dict('index'),
        'style_limits':limits,'benchmark_weights':None if raw_kind else benchmark.to_dict(),'style_controls_prestandardized':True,
        'bootstrap':not bool(held),'freeze_budget':False,'planned_trade_date':following,'cost_policy_version':VERSION,
        'execution_slippage_bps':config.slippage_bps,'execution_policy_sha256':execution_policy_hash(candidate),
        'real_orders_enabled':False,'promotion_eligible':False,'economics':economics,'risk_input_audit':audit,
        'frozen_forecast_symbols':sorted(frozen),'forced_exit_symbols':sorted(exits-halted),'locked_shares':snapshot['unsettled_stock_rights'],
        'receivable_cash':snapshot['receivable_cash'],'alpha_evidence_enabled':calibration['optional_alpha_enabled'],
        'style_comparison':{'chinext_weight':sum(w for s,w in targets.items() if s.startswith('30')),'star_weight':sum(w for s,w in targets.items() if s.startswith('68')),
            'effective_holdings':float(1/sum(w*w for w in targets.values())),'target_count_above_1bp':sum(w>.0001 for w in targets.values())}}
    write_diagnostic(root/'state/qlib-portfolio-plans'/tag/(trade_date+'.json'),plan)
    snapshot['initial_cash']=config.initial_cash
    _record_account_history(store,trade_date,snapshot,account,'qlib_candidate_account_daily')
    return _finish(trade_date,'EXPERIMENTAL_PAPER',candidate['model_id'],[s for s,w in targets.items() if w>1e-6],
        pd.DataFrame({'instrument':eligible.index,'score':eligible.to_numpy()}).to_dict('records'),[],[],snapshot,broker,
        root/'state/qlib-candidate-shadow'/tag,{**plan,'portfolio_variant':candidate['portfolio_variant'],'display_name':candidate['display_name'],
            'target_gross_exposure':sum(targets.values()),'observation_only':True},account_id=account)


def execute_cost_plan(plan,marks,held,available,cash,equity,config,can_buy,can_sell,*,fixed_quantities=None):
    from .research.cost_integer import plan_integer_execution
    from .research.cost_aware import risk_compliant
    from .research.cost_execution import holding_plan,finalize_cost_decision
    styles=pd.DataFrame.from_dict(plan['styles'],orient='index');limits=plan['style_limits'];benchmark=plan['benchmark_weights'];economics=plan['economics']
    locked=plan.get('locked_shares',{});q=pd.Series(held,dtype=float).add(pd.Series(locked,dtype=float),fill_value=0)
    current={s:n*marks[s]/equity for s,n in q.items() if n>0}
    if fixed_quantities is not None and {s:int(n) for s,n in fixed_quantities.items() if n>0}=={s:int(n) for s,n in held.items() if n>0} and not (set(plan.get('forced_exit_symbols',[]))&set(held)) and risk_compliant(current,styles,limits,benchmark):
        return holding_plan(marks,held,cash,equity,styles,limits,benchmark,locked_shares=locked)
    if fixed_quantities is None and economics.get('preserve_quantities') and risk_compliant(current,styles,limits,benchmark):
        return holding_plan(marks,held,cash,equity,styles,limits,benchmark,locked_shares=locked)
    result=plan_integer_execution(plan['target_weights'],marks,held,available,cash,equity,config,styles,limits,locked_shares=locked,
        can_buy=can_buy,can_sell=can_sell,benchmark_weights=benchmark,style_controls_prestandardized=True,
        time_limit=60 if fixed_quantities is None else 5,solver_threads=4,mip_relative_gap=.01,
        research_secondary_time_limit=10 if fixed_quantities is None else None,
        economic_tracking_weights=economics['economic_tracking_weights'],accept_validated_incumbent=True,
        research_objective_scale=10000 if benchmark is not None else None,
        stock_active_bound=None if benchmark is None else max(0.,.15-max(benchmark.values())),
        forced_sell_symbols=plan.get('forced_exit_symbols',[]),
        fixed_quantities=fixed_quantities if fixed_quantities is not None else {s:held[s] for s in plan.get('frozen_forecast_symbols',[]) if s in held})
    return finalize_cost_decision(result,marks,held,cash,equity,styles,limits,benchmark,economics,forced_exit=plan.get('forced_exit_symbols',[]),locked_shares=locked)
