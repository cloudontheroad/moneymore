"""Frozen native Alpha158 pair. Paper accounts only; no real-broker adapter."""
from __future__ import annotations

from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading

import numpy as np
import pandas as pd

from .challenger_diagnostics import trading_dates, write_diagnostic
from .research.mature_multifactor import current_risk_inputs, topk_dropout_targets
from .research.enhanced_indexing import EnhancedPolicy, optimize_snapshot

KINDS={'mature_alpha158_raw','mature_alpha158_balanced'}
_BUILD_LOCK=threading.RLock()


def is_mature(candidate):
    return candidate.get('portfolio_variant') in KINDS


def pair_view(root):
    """Small operating view, never exposes model files or training logs."""
    rows=mature_candidates(root)
    if not rows:return None
    registry=json.loads((root/'state/mature-multifactor/registry.json').read_text(encoding='utf-8'))
    backtest=json.loads(Path(registry['historical_replay']).read_bytes());accounts=[]
    for row in rows:
        latest=row['latest_report'];portfolio=latest['portfolio'];weights=latest.get('target_weights',{})
        accounts.append({'account_id':row['account_id'],'name':row['display_name'],'initial_cash':row['initial_cash'],
            'status':latest.get('status'),'signal_date':latest.get('signal_date'),'planned_trade_date':latest.get('planned_trade_date'),
            'actual_gross':portfolio['market_value']/portfolio['equity'],'target_gross':sum(weights.values()) if weights else None,
            'portfolio':portfolio,'style':latest.get('style_comparison',{}),
            'targets':[{'symbol':s,'weight':w} for s,w in sorted(weights.items(),key=lambda item:item[1],reverse=True) if w>1e-6]})
    return {'status':'INDEPENDENT_PAPER_OBSERVATION','model':registry.get('model_display_name','Qlib Alpha158 + LightGBM'),'accounts':accounts,
        'historical_replay':[{'name':'多因子原选股' if r['account']=='raw' else '多因子风格均衡','net_return':r['net_return'],
            'fills':r['fills'],'executed_sessions':r['passed_rebalances']} for r in backtest['cases']],
        'historical_replay_is_not_live_nav':True,'promotion_authorized':False,'real_orders_enabled':False}


def mature_candidates(root):
    path=root/'state/mature-multifactor/registry.json'
    if not path.exists():return []
    from .execution.paper import PaperBroker
    from .candidate_lifecycle import lifecycle_rows, ARCHIVED
    registry=json.loads(path.read_text(encoding='utf-8'));valid=True
    if registry.get('real_orders_enabled') is not False or registry.get('schema')!='MATURE_ALPHA158_PAIR_V1':raise ValueError('INVALID_MATURE_PAIR_SCOPE')
    for name,digest in registry['evidence_hashes'].items():
        file=Path(name).resolve()
        if not file.is_relative_to(root.resolve()) or not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:valid=False
    states=lifecycle_rows(root);rows=[];broker=PaperBroker(root/'state/paper_orders.sqlite3')
    for entry in registry['accounts']:
        tag=entry['candidate_tag'];folder=root/'state/qlib-candidate-shadow'/tag
        reports=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('*.json'))]
        latest=reports[-1] if reports else {'status':'AWAITING_FIRST_PLAN','portfolio':broker.account_snapshot({},entry['account_id'])}
        latest['portfolio']['initial_cash']=entry['initial_cash']
        rows.append({**entry,'research_gate_passed':valid,'research_gate_failures':[] if valid else ['MATURE_EVIDENCE_CHANGED'],
            'research_gate_scope':'PAPER_OBSERVATION_ENGINEERING_ONLY_NOT_PROFIT_OR_PROMOTION','validation_hold':not valid,
            'research_path':str(path),'model_dir':registry['model_dir'],'model_manifest':registry['model_manifest'],
            'data_cutoff':registry['data_cutoff'],'protocol_version':registry['version'],
            'observation_days':len(reports),'latest_report':latest,'latest_portfolio':latest['portfolio'],
            'latest_status':latest['status'],'latest_trade_date':latest.get('trade_date'),
            'lifecycle':states.get(tag,{}),'archived':states.get(tag,{}).get('status') in ARCHIVED,
            'rank_ic':registry['historical_rank_ic'],'rank_ic_ir':None,
            'cost_adjusted_top_k_excess_return':None,'promotion_eligible':False})
    return rows


def _run(root,script,arguments,log,timeout=1800):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as handle:
        result=subprocess.run([sys.executable,str(root/'scripts'/script),*map(str,arguments)],cwd=root,
            stdout=handle,stderr=subprocess.STDOUT,timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode:raise ValueError('MATURE_LOCAL_STEP_FAILED:'+script)


def runtime_inputs(root,day):
    """Reuse bound ready artifacts; at most three failed local attempts per date."""
    base=root/'state/mature-multifactor/runtime'
    with _BUILD_LOCK:
        def ready(kind):
            candidates=sorted((base/kind).glob(day+'-*'))
            return next((p for p in reversed(candidates) if (p/'results.json').exists()),None)
        official=ready('official')
        if official is None:
            for index in range(3):
                folder=base/'official'/f'{day}-{index}'
                if folder.exists():continue
                _run(root,'prepare_mature_close.py',['--date',day,'--output',folder],base/'logs'/f'{day}-official-{index}.log')
                official=folder;break
        if official is None:raise ValueError('MATURE_OFFICIAL_CLOSE_RETRY_BUDGET_EXHAUSTED')
        features=ready('features')
        if features is None:
            for index in range(3):
                folder=base/'features'/f'{day}-{index}'
                if folder.exists():continue
                _run(root,'build_mature_alpha158.py',['--prediction-only','--official-overlay',official,'--output',folder],base/'logs'/f'{day}-features-{index}.log')
                features=folder;break
        if features is None:raise ValueError('MATURE_FEATURE_RETRY_BUDGET_EXHAUSTED')
        manifest=json.loads((features/'results.json').read_bytes())
        if manifest['cutoff']!=day or not manifest.get('prediction_only'):raise ValueError('MATURE_STALE_CLOSE_NOT_CURRENT_SIGNAL')
        for name,digest in manifest['input_hashes'].items():
            if hashlib.sha256((features/name).read_bytes()).hexdigest()!=digest:raise ValueError('MATURE_RUNTIME_INPUT_CHANGED')
        return official,features


def infer(root,candidate,day):
    official,features=runtime_inputs(root,day)
    model_dir=Path(candidate['model_dir']);model_manifest=json.loads(Path(candidate['model_manifest']).read_bytes())
    for name in ['model.pkl','processor.pkl']:
        if hashlib.sha256((model_dir/name).read_bytes()).hexdigest()!=model_manifest['artifact_hashes'][name]:raise ValueError('MATURE_MODEL_CHANGED')
    panel=pd.read_parquet(features/'alpha158-panel.parquet');processor=pickle.loads((model_dir/'processor.pkl').read_bytes())
    names=processor['normalizer']['schema'];frame=panel[names].fillna(processor['medians']);frame.columns=pd.MultiIndex.from_product([['feature'],names]);frame[('label','TOTAL_RETURN')]=np.nan
    from .research.qlib_feature_preprocessing import transform_features
    from .qlib_challenger import QlibPanelDataset
    frame=transform_features(processor['normalizer'],frame)
    if model_manifest.get('model_class')=='qlib.contrib.model.linear.LinearModel':frame=frame.astype('float64')
    dataset=QlibPanelDataset(frame,{'latest':(day,day)})
    model=pickle.loads((model_dir/'model.pkl').read_bytes());score=model.predict(dataset,'latest').xs(pd.Timestamp(day),level='datetime')
    if not np.isfinite(score).all() or score.std()<=1e-12:raise ValueError('MATURE_LATEST_SCORE_INVALID')
    return score,official,features


def prepare_mature_daily(*,root,store,broker,config,candidate,trade_date):
    from .qlib_challenger_daily import _finish,_record_account_history
    from .research.style_v2_experiment import candidate_execution_config,execution_policy_hash
    from .research.replay_actions import official_replay_actions,economic_snapshot
    from .strategy_universe import active_strategy_universe
    tag=candidate['candidate_tag'];account=candidate['account_id'];config=candidate_execution_config(config,candidate)
    if trade_date<candidate['activation_date']:raise ValueError('MATURE_NOT_ACTIVE')
    broker.initialize_account(config.initial_cash,account)
    scores,official,features=infer(root,candidate,trade_date)
    bars=pd.read_parquet(features/'bars-input.parquet');marks=bars.loc[bars.trade_date.astype(str).eq(trade_date)].set_index('ts_code').close.to_dict()
    snapshot=broker.account_snapshot(marks,account);held={p['symbol']:int(p['quantity']) for p in snapshot['positions'] if p['quantity']>0}
    try:actions=official_replay_actions(store.read('dividend').loc[lambda x:x.ts_code.isin(held)])
    except FileNotFoundError:actions=official_replay_actions(pd.DataFrame())
    broker.settle_corporate_actions(account,trade_date);broker.settle_available_positions(account,trade_date)
    for action in actions.loc[actions.record_date.eq(pd.Timestamp(trade_date))].itertuples():
        broker.register_corporate_entitlement(account_id=account,symbol=action.symbol,action_key=action.action_key,record_date=trade_date,
            cash_per_share=action.cash_per_share,stock_ratio=action.stock_ratio,
            cash_pay_date=None if pd.isna(action.cash_pay_date) else action.cash_pay_date.strftime('%Y%m%d'),
            stock_list_date=None if pd.isna(action.stock_list_date) else action.stock_list_date.strftime('%Y%m%d'))
    members=set(active_strategy_universe(store,trade_date).symbol);exits=set(held)-members
    proof_frames=[pd.read_parquet(p/'suspend_d.parquet') for p in (root/'state/mature-multifactor/runtime/official').glob('*') if (p/'results.json').exists()]
    proof=pd.concat(proof_frames,ignore_index=True) if proof_frames else pd.DataFrame()
    full=proof.loc[proof.trade_date.astype(str).eq(trade_date)&proof.suspend_type.eq('S')&(proof.suspend_timing.isna()|proof.suspend_timing.eq(''))] if not proof.empty else proof
    halted=set(full.ts_code) if not full.empty else set()
    # A documented halt is not an unpriced fill. Carry the last known valuation.
    for symbol in set(held)-set(marks):
        if symbol not in halted:raise ValueError('MATURE_HELD_CLOSE_UNOBSERVED')
        old=bars.loc[bars.ts_code.eq(symbol)&bars.trade_date.astype(str).lt(trade_date)].sort_values('trade_date')
        if old.empty:raise ValueError('MATURE_HELD_LAST_MARK_UNKNOWN')
        marks[symbol]=float(old.iloc[-1].close)
        for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(trade_date))].itertuples():marks[symbol]=(marks[symbol]-action.cash_per_share)/(1+action.stock_ratio)
    snapshot=economic_snapshot(broker,broker.account_snapshot(marks,account),marks,account,trade_date,actions)
    if snapshot['unsettled_stock_rights']:raise ValueError('MATURE_PENDING_STOCK_RIGHTS_HOLD_AND_ALERT')
    held={p['symbol']:int(p['quantity']) for p in snapshot['positions'] if p['quantity']>0}
    risk_scores=scores.reindex(sorted((set(scores.index)&members)|set(held)))
    returns,size,raw,controls,benchmark,limits,audit=current_risk_inputs(store,bars,risk_scores,trade_date,
        suspension_proof=proof,candidate_members=members,basic_overlay=pd.read_parquet(official/'daily_basic.parquet'))
    if (set(held)&members)-set(scores.index) or set(held)-set(controls.index):raise ValueError('MATURE_HELD_INPUT_MISSING_NOT_EXIT')
    eligible=scores.reindex(controls.index.intersection(scores.index).intersection(pd.Index(sorted(members))))
    from .research.prediction_quality import prediction_quality
    if not prediction_quality(eligible.to_numpy())['passed']:raise ValueError('MATURE_UNRESOLVED_TOPK_BOUNDARY_TIE')
    values=pd.Series({s:q*marks[s] for s,q in held.items()},dtype=float)
    if candidate['portfolio_variant']=='mature_alpha158_raw':
        raw_targets=topk_dropout_targets(eligible,values.drop(index=list(exits),errors='ignore')/snapshot['equity']) if not (exits&halted) else None
        if raw_targets is None:
            weights=values/snapshot['equity'];locked=weights.reindex(sorted(exits&halted)).dropna();free=1-float(locked.sum())
            raw_targets={s:w*free for s,w in topk_dropout_targets(eligible,weights.drop(index=list(exits),errors='ignore')/free,topk=50-len(locked)).items()};raw_targets.update(locked.to_dict())
        targets=raw_targets;execution_styles=pd.DataFrame({'execution_only':0.},index=controls.index);execution_limits={'execution_only':0.};optimizer=None
    else:
        signal=risk_scores.reindex(controls.index);signal.loc[list(exits&set(signal.index))]=0.
        optimizer=optimize_snapshot(signal,returns,size,values/values.sum() if len(values) else values,EnhancedPolicy(objective_preconditioning=True),
            style_controls=controls,style_limits=limits,benchmark_weights=benchmark,controls_prestandardized=True,
            allow_cash_bootstrap=True,reject_turnover_relaxation=True,forced_sell_symbols=exits-halted,forced_hold_symbols=set(held)&halted,solver_max_iters=1000)
        if optimizer['status']!='PASS_OFFLINE_ONLY':raise ValueError('MATURE_NATIVE_OPTIMIZER_BLOCKED')
        targets=optimizer['weights'];execution_styles=controls;execution_limits=limits
    calendar=trading_dates(store,trade_date);following=calendar[calendar.index(trade_date)+1]
    plan={'candidate_tag':tag,'account_id':account,'signal_date':trade_date,'created_at':datetime.now(UTC).isoformat(),
        'strategy_id':candidate['strategy_id'],'model_id':candidate['model_id'],'target_weights':targets,
        'styles':execution_styles.to_dict('index'),'style_limits':execution_limits,'benchmark_weights':benchmark.to_dict(),
        'style_controls_prestandardized':True,'bootstrap':not bool(held),'freeze_budget':False,'planned_trade_date':following,
        'execution_slippage_bps':config.slippage_bps,'execution_policy_sha256':execution_policy_hash(candidate),
        'real_orders_enabled':False,'promotion_eligible':False,'optimizer':optimizer,'risk_input_audit':audit,
        'style_comparison':{'chinext_weight':sum(w for s,w in targets.items() if s.startswith('30')),
            'star_weight':sum(w for s,w in targets.items() if s.startswith('68')),
            'effective_holdings':float(1/sum(w*w for w in targets.values())),'target_count_above_1bp':sum(w>.0001 for w in targets.values())}}
    write_diagnostic(root/'state/qlib-portfolio-plans'/tag/(trade_date+'.json'),plan)
    snapshot['initial_cash']=config.initial_cash
    _record_account_history(store,trade_date,snapshot,account,'qlib_candidate_account_daily')
    return _finish(trade_date,'EXPERIMENTAL_PAPER',candidate['model_id'],[s for s,w in targets.items() if w>1e-6],
        pd.DataFrame({'instrument':eligible.index,'score':eligible.to_numpy()}).to_dict('records'),[],[],snapshot,broker,
        root/'state/qlib-candidate-shadow'/tag,{**plan,'portfolio_variant':candidate['portfolio_variant'],'display_name':candidate['display_name'],
            'target_gross_exposure':sum(targets.values()),'observation_only':True},account_id=account)


def reprice_reference_plan(plan,marks,held,cash,config,styles,limits,benchmark,buy,sell):
    """Recheck fixed quantities at freshly fetched quotes after a long MILP solve."""
    from .models import Side
    from .costs import execution_price,transaction_fee
    if plan.get('status')!='PASS_REFERENCE_QUOTES_ONLY':raise ValueError('ONLY_CERTIFIED_INTEGER_PLAN')
    required=set(plan['target_quantities'])|set(held)
    if required-set(marks) or not np.isfinite(cash) or any(not np.isfinite(marks[s]) or marks[s]<=0 for s in required):
        raise ValueError('INVALID_FRESH_REFERENCE_PRICE')
    updated={**plan,'orders':[dict(o) for o in plan['orders']]};remaining=float(cash);fees=0.
    for order in updated['orders']:
        symbol=order['symbol'];side=Side(order['side']);quantity=int(order['quantity'])
        if not (buy.get(symbol,False) if side==Side.BUY else sell.get(symbol,False)):raise ValueError('FRESH_QUOTE_NOT_TRADABLE')
        price=execution_price(marks[symbol],side,config.slippage_bps);fee=transaction_fee(price*quantity,side,config)
        remaining+=(-1 if side==Side.BUY else 1)*price*quantity-fee;fees+=fee
        order.update(reference_price=marks[symbol],estimated_execution_price=price,estimated_fee=fee)
    values=pd.Series({s:q*marks[s] for s,q in updated['target_quantities'].items() if q>0},dtype=float)
    old=pd.Series({s:q*marks[s] for s,q in held.items()},dtype=float)
    weights=(values/values.sum()).reindex(styles.index,fill_value=0.)
    active=(weights-pd.Series(benchmark).reindex(styles.index))@styles
    turnover=float(abs(weights-(old/old.sum()).reindex(styles.index,fill_value=0.)).sum()) if old.sum()>0 else 0.
    equity=float(cash+old.sum())
    if not np.isfinite([remaining,fees,turnover,equity,*active]).all() or remaining < -1e-6 or values.max()>config.max_position_weight*equity+1e-5 or (abs(active)>pd.Series(limits)+1e-5).any() or turnover>.20+1e-5:
        raise ValueError('FRESH_QUOTE_CASH_RISK_OR_TURNOVER_CHANGED')
    updated.update(estimated_cash=remaining,estimated_fees=fees,estimated_gross=float(values.sum()/equity),
        turnover_l1=turnover,active_style_exposures=active.to_dict(),fresh_quote_revalidated=True)
    return updated
