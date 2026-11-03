"""Six isolated fixed-episode ledgers; real PaperBroker, frozen OOS signals."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import yaml
from qlib.model.riskmodel import StructuredCovEstimator
from moneymore.config import BacktestConfig
from moneymore.models import Side
from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.execution.paper import PaperBroker,ExecutionBar
from moneymore.execution.risk import RiskResult,OrderIntent
from moneymore.research.style_price_features import named_price_risk_exposures
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.score_neutralization import neutralize_scores
from moneymore.research.style_replay_inputs import optimize_daily
from moneymore.research.quality_contract import quality_upper_only_bounds
from moneymore.research.integer_execution import plan_integer_execution,paper_intents_from_integer_plan
from moneymore.research.replay_actions import economic_snapshot
from moneymore.research.holding_eligibility import classify_holdings
from moneymore.research.quote_permissions import quote_permissions
from moneymore.research.order_chunks import liquidation_chunks


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fold',type=int,default=0,choices=range(4));p.add_argument('--held-evidence',type=Path);p.add_argument('--capture-from-study',type=Path);p.add_argument('--integer-direct-seconds',type=float);p.add_argument('--slippage-bps',type=float,choices=[2.,10.]);args=p.parse_args()
    if args.integer_direct_seconds is not None and not 0<args.integer_direct_seconds<=300:raise ValueError('INVALID_BOUNDED_INTEGER_BUDGET')
    if args.output.exists():raise FileExistsError('IMMUTABLE_COST_REPLAY')
    root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data')
    supplement=None
    if args.held_evidence:
        from moneymore.research.held_evidence import read_held_basic
        supplement=read_held_basic(args.held_evidence)
    inference=root/'state/style-balance/s3-feature-only-inference-20261103-a'
    evidence=root/f'state/style-balance/s3-cost-evidence-fold{args.fold}-20261103-a'
    checked=root/f'state/style-balance/s3-cost-evidence-audit-fold{args.fold}-20261103-a'
    validation=json.loads((inference/'independent-audit.json').read_text());manifest=json.loads((evidence/'manifest.json').read_text())
    checks=json.loads((checked/'results.json').read_text())
    if checks['status']!='GATES_ACTIONS_AND_QUOTES_AUDITED':raise ValueError('COST_EVIDENCE_NOT_READY')
    for name,digest in validation['artifact_hashes'].items():
        if hashlib.sha256((inference/name).read_bytes()).hexdigest()!=digest:raise ValueError('INFERENCE_CHANGED')
    for item in manifest['requests']:
        if hashlib.sha256((evidence/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('EXECUTION_EVIDENCE_CHANGED')
    protocol_path=root/'configs/style_s3_cost_replay_v1.yaml';protocol=yaml.safe_load(protocol_path.read_text())
    if args.slippage_bps is not None:protocol['slippage_bps_each_side']=args.slippage_bps
    slip=protocol['slippage_bps_each_side']/10000
    sessions=manifest['episode']['sessions'];calendar=trading_dates(store,sessions[-1]);first=calendar.index(sessions[0]);signal0=calendar[first-1]
    score_frames={m:pd.read_parquet(inference/f'fold{args.fold}-price_only-{m}-scores.parquet').score for m in protocol['models']}
    available_features=pd.read_parquet(inference/'available-features.parquet')
    foundation=json.loads((root/'state/style-balance/s1-feature-paths-20261102-a/audit.json').read_text())
    pool_path=Path(foundation['historical_pool']['path'])
    if hashlib.sha256(pool_path.read_bytes()).hexdigest()!=foundation['historical_pool']['sha256']:raise ValueError('POOL_CHANGED')
    pool=pd.read_parquet(pool_path);pool['date']=pd.to_datetime(pool.signal_date).dt.strftime('%Y%m%d')
    symbols=sorted(set(available_features.loc[available_features.index.get_level_values(0).isin(pd.to_datetime([signal0]+sessions))].index.get_level_values(1)))
    start=calendar[first-121];filters=[('ts_code','in',symbols),('trade_date','>=',start),('trade_date','<=',sessions[-1])]
    bars=store.read('daily',columns=['ts_code','trade_date','open','close','amount'],filters=filters)
    bars=bars.merge(store.read('adj_factor',columns=['ts_code','trade_date','adj_factor'],filters=filters),on=['ts_code','trade_date'],how='left',validate='one_to_one')
    bars['total_price']=bars.close*bars.adj_factor
    prices=bars.pivot(index='trade_date',columns='ts_code',values='total_price');quotes=bars.set_index(['trade_date','ts_code'])
    membership=store.read('baseline_v2_industry_membership');actions=pd.read_parquet(checked/'official-actions.parquet')
    cfg=BacktestConfig(initial_cash=protocol['capital_each_episode'],commission_rate=protocol['commission_rate'],minimum_commission=protocol['minimum_commission'],
        stamp_duty_rate=protocol['stamp_duty_rate'],transfer_fee_rate=protocol['transfer_fee_rate'],slippage_bps=protocol['slippage_bps_each_side'],lot_size=100,max_position_weight=.15,max_gross_exposure=1.,max_drawdown=.2)
    args.output.mkdir(parents=True);(args.output/'protocol.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
    (args.output/'episode-binding.json').write_text(json.dumps({'fold':args.fold,'evidence':str(evidence.resolve()),'audit':str(checked.resolve()),
        'manifest_sha256':hashlib.sha256((evidence/'manifest.json').read_bytes()).hexdigest(),
        'audit_sha256':hashlib.sha256((checked/'results.json').read_bytes()).hexdigest()},indent=2))
    (args.output/'solver-policy.json').write_text(json.dumps({'mode':'direct' if args.integer_direct_seconds else 'staged_seed',
        'time_limit':args.integer_direct_seconds or 10,'target_time_limit':None if args.integer_direct_seconds else 20,
        'secondary_time_limit':min(30,args.integer_direct_seconds) if args.integer_direct_seconds else 10,
        'mip_relative_gap':.01,'threads':2,'economic_constraints_changed':False},indent=2))
    if args.held_evidence:
        (args.output/'held-evidence-binding.json').write_text(json.dumps({'path':str(args.held_evidence.resolve()),
            'manifest_sha256':hashlib.sha256((args.held_evidence/'manifest.json').read_bytes()).hexdigest()},indent=2))
    bars.to_parquet(args.output/'market-history.parquet',index=False);contexts={};projections={};results=[]
    def context(day,score,held):
        # Exposure/risk inputs depend on the union of names, not quantities or
        # which already-scored names happen to be held by this account.
        key=(day,tuple(sorted(set(score.index)|set(held))))
        if key in contexts:return contexts[key]
        current=pool.loc[pool.date.eq(day)].set_index('ts_code')
        ix=pd.Index(sorted(set(score.index)|set(held)))
        # Risk-only held exits require actual same-day basic data, not a stale
        # pool row or an invented size. Missing mandatory data remains blocked.
        attrs=current.reindex(ix).copy()
        exits=ix.difference(current.index)
        if len(exits):
            basic=store.read('daily_basic',columns=['ts_code','trade_date','total_mv','pb','pe_ttm','dv_ttm'],
                filters=[('ts_code','in',exits.tolist()),('trade_date','==',day)]).set_index('ts_code')
            if supplement is not None:
                extra=supplement.loc[supplement.trade_date.astype(str).eq(day)&supplement.ts_code.isin(exits)].set_index('ts_code')
                # Never silently override an existing local observation.
                for symbol in basic.index.intersection(extra.index):
                    for field in ['total_mv','pb','pe_ttm','dv_ttm']:
                        left,right=basic.loc[symbol,field],extra.loc[symbol,field]
                        if not (pd.isna(left) and pd.isna(right)) and not np.isclose(left,right,equal_nan=True):raise ValueError('HELD_BASIC_SOURCE_CONFLICT')
                basic=pd.concat([basic,extra.loc[extra.index.difference(basic.index)]])
            for field in ['total_mv','pb','pe_ttm','dv_ttm']:attrs.loc[exits,field]=basic[field].reindex(exits)
        days=[d for d in calendar if d<=day][-121:]
        hist=prices.reindex(index=days,columns=ix)
        raw,_=named_price_risk_exposures(hist,attrs.total_mv*10000)
        raw['book_to_price']=1/attrs.pb.where(lambda x:x>0)
        raw['earnings_yield']=1/attrs.pe_ttm.replace(0,np.nan)
        raw['dividend_yield']=attrs.dv_ttm/100
        raw['industry']=ix.map(industry_as_of(membership,day).l1_code).fillna('UNKNOWN')
        raw['listing_board']=['STAR' if s.startswith('68') else 'CHINEXT' if s.startswith('30') else 'MAIN' for s in ix]
        raw=raw.replace([np.inf,-np.inf],np.nan)
        c,eligible,_=uncertain_controls(raw,['log_size','momentum60','volatility20','market_beta120','residual_volatility120']);raw=raw.loc[eligible]
        coverage=classify_holdings(held,current.index,score.index,raw.index)
        if coverage['missing_risk']:raise ValueError('HELD_RISK_DATA_MISSING_NOT_A_POOL_EXIT:'+','.join(coverage['missing_risk']))
        for field in raw.select_dtypes(include='number').columns:
            if field not in ['log_size','momentum60','volatility20','market_beta120','residual_volatility120']:c['unknown_'+field]=raw[field].isna().astype(float)
        c['unknown_industry']=raw.industry.eq('UNKNOWN').astype(float)
        wb=current.total_mv.reindex(raw.index,fill_value=0.);wb/=wb.sum()
        returns=hist.reindex(columns=raw.index).pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(returns).all().all():raise ValueError('RISK_HISTORY_NOT_COMPLETE')
        np.random.seed(17);F,cov,specific=StructuredCovEstimator(num_factors=10,factor_model='pca',scale_return=False).predict(returns,is_price=False,return_decomposed_components=True)
        limits=pd.Series({f:.01 if f.startswith('unknown_') else .03 if f.startswith(('industry_','board_')) else .1 for f in c.columns})
        contexts[key]=(raw,c,F,cov,specific,wb,limits);return contexts[key]
    def signal(day,model,variant):
        key=(day,model)
        if key not in projections:
            score=score_frames[model].xs(pd.Timestamp(day),level='datetime');attrs=pool.loc[pool.date.eq(day)].set_index('ts_code')
            features=available_features.xs(pd.Timestamp(day),level='datetime').reindex(score.index)
            z=pd.DataFrame({'log_size':np.log(attrs.total_mv.reindex(score.index)*10000),'book_to_price':1/attrs.pb.reindex(score.index).where(lambda x:x>0),
                'dividend_yield':attrs.dv_ttm.reindex(score.index)/100,'return_20':features.return_20,'volatility_20':features.volatility_20}).replace([np.inf,-np.inf],np.nan)
            for field in list(z.columns):
                z['unknown_'+field]=z[field].isna().astype(float)
                if not z[field].notna().any():raise ValueError('NO_STYLE_REFERENCE')
                z[field]=z[field].fillna(z[field].median())
            industry=pd.Series(score.index.map(industry_as_of(membership,day).l1_code),index=score.index).fillna('UNKNOWN')
            z=pd.concat([z,pd.get_dummies(industry,prefix='industry',dtype=float)],axis=1)
            residual,_=neutralize_scores(score,z)
            projections[key]={'raw':score,'contemporaneous_style_component':score-residual,'contemporaneous_score_residual':residual}
            pd.DataFrame(projections[key]).to_parquet(args.output/f'signals-{day}-{model}.parquet')
        return projections[key][variant]
    for model in protocol['models']:
      for variant in protocol['signal_variants']:
        account=f'{model}-{variant}';folder=args.output/account;folder.mkdir();broker=PaperBroker(folder/'paper.sqlite3');broker.initialize_account(cfg.initial_cash,'isolated')
        last=bars.loc[bars.trade_date.le(signal0)].sort_values('trade_date').groupby('ts_code').tail(1).set_index('ts_code').close.to_dict()
        events=[];equity=[]
        for offset,day in enumerate(sessions,1):
            broker.settle_corporate_actions('isolated',day);broker.settle_available_positions('isolated',day)
            qday=quotes.loc[day].copy();susp=pd.read_parquet(evidence/f'suspend_d-{day}.parquet')
            halted=set(susp.loc[susp.suspend_type.eq('S') & (susp.suspend_timing.isna()|susp.suspend_timing.eq('')),'ts_code']) if not susp.empty else set()
            opening_marks=qday.open.dropna().to_dict();closing_marks=qday.close.dropna().to_dict()
            for symbol in halted-set(opening_marks):
                if symbol not in last:continue
                mark=last[symbol]
                for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():mark=(mark-action.cash_per_share)/(1+action.stock_ratio)
                opening_marks[symbol]=mark;closing_marks[symbol]=mark
            opened=economic_snapshot(broker,broker.account_snapshot(opening_marks,'isolated'),opening_marks,'isolated',day,actions)
            held={p['symbol']:int(p['quantity']) for p in opened['positions'] if p['quantity']>0};available={p['symbol']:int(p['available_quantity']) for p in opened['positions']}
            if set(held)-set(opening_marks):raise ValueError('HELD_MARK_UNOBSERVABLE')
            if offset in protocol['rebalance_execution_offsets'] or offset==protocol['liquidate_execution_offset']:
                gate=pd.read_parquet(evidence/f'stk_limit-{day}.parquet').set_index('ts_code')
                buy,sell=quote_permissions(opening_marks,halted,gate,protocol['slippage_bps_each_side'])
                broker.cancel_pending('isolated','FIXED_EPISODE_REPLAN',signal_date=None)
                if offset==protocol['liquidate_execution_offset']:
                    for s,quantity in held.items():
                        count=available.get(s,0)
                        if count and sell.get(s,False):
                            prior_signal=calendar[calendar.index(day)-1]
                            for part,chunk in enumerate(liquidation_chunks(s,count)):
                                intent=OrderIntent(f'{account}:terminal:{day}:{s}:{part}',account,s,Side.SELL,chunk,prior_signal,'PREDECLARED_TERMINAL_EXIT')
                                broker.submit(RiskResult(True,intent,None),'isolated')
                                broker.execute_pending(ExecutionBar(s,day,opening_marks[s],opening_marks[s],False,True),cfg,'isolated')
                    events.append({'date':day,'status':'TERMINAL_EXIT_ATTEMPT','held_before':len(held)})
                else:
                    prior=calendar[calendar.index(day)-1];scores=signal(prior,model,variant)
                    try:
                        candidates=pool.loc[pool.date.eq(prior),'ts_code']
                        classification=classify_holdings(held,candidates,scores.index,held)
                        classification.pop('missing_risk')
                        classification['risk_coverage']='NOT_YET_CHECKED'
                        (folder/f'holding-status-{day}.json').write_text(json.dumps(classification,indent=2))
                        if classification['in_pool_missing_score']:raise ValueError('IN_POOL_SCORE_MISSING_NOT_EXIT:'+','.join(classification['in_pool_missing_score']))
                        raw,c,F,cov,specific,wb,limits=context(prior,scores,held)
                        if opened['unsettled_stock_rights']:raise ValueError('UNSETTLED_STOCK_RIGHTS_REQUIRE_SEPARATE_PLANNING')
                        zero={f:{'upper':0.,'lower':0.} for f in c.columns}
                        old_close=quotes.loc[prior].close;held_values=pd.Series(held,dtype=float)*old_close.reindex(held)
                        if args.capture_from_study:
                            continuous_case={'scores':scores.to_dict(),'controls':c.to_dict(orient='split'),'F':F.tolist(),'covariance':cov.tolist(),
                                'specific':specific.tolist(),'wb':wb.to_dict(),'limits':limits.to_dict(),'held_values':held_values.to_dict(),
                                'candidates':candidates.tolist(),'quality_envelope':zero}
                            (folder/f'continuous-case-{day}.json').write_text(json.dumps(continuous_case,allow_nan=False))
                        target=optimize_daily(scores,raw,c,F,cov,specific,wb,limits,held_values,research_quality_envelope=zero,candidate_symbols=candidates)
                        if args.capture_from_study:
                            source=args.capture_from_study/account/f'plan-{day}.json'
                            plan=json.loads(source.read_text())
                            if (plan.get('reason') or '').startswith(('INTEGER_FEASIBILITY_USER_LIMIT','WARM_START_SEED_USER_LIMIT','INTEGER_FEASIBILITY_INFEASIBLE')):
                                from moneymore.research.integer_case import encode_case
                                case=encode_case(target,pd.Series(opening_marks),held,available,opened['cash'],opened['equity'],cfg,c,limits,
                                    can_buy=buy,can_sell=sell,benchmark_weights=wb,style_controls_prestandardized=True,research_quality_envelope=zero)
                                case['source_plan_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
                                (folder/f'integer-case-{day}.json').write_text(json.dumps(case,indent=2,allow_nan=False))
                        else:
                            solver_options={'time_limit':args.integer_direct_seconds,'research_secondary_time_limit':min(30,args.integer_direct_seconds)} if args.integer_direct_seconds else {
                                'time_limit':10,'research_staged_warm_start':True,'research_target_time_limit':20}
                            plan=plan_integer_execution(target,pd.Series(opening_marks),held,available,opened['cash'],opened['equity'],cfg,c,limits,
                                can_buy=buy,can_sell=sell,benchmark_weights=wb,style_controls_prestandardized=True,research_quality_envelope=zero,
                                **solver_options,solver_threads=2,mip_relative_gap=.01)
                    except ValueError as error:plan={'status':'BLOCKED','reason':str(error),'orders':[],'orders_enabled':False}
                    if plan['status']=='PASS_REFERENCE_QUOTES_ONLY':
                        for intent in paper_intents_from_integer_plan(plan,account,prior):broker.submit(intent,'isolated')
                        for order in plan['orders']:
                            s=order['symbol'];broker.execute_pending(ExecutionBar(s,day,opening_marks[s],opening_marks[s],buy[s],sell[s]),cfg,'isolated')
                        actual=broker.account_snapshot(opening_marks,'isolated');qty={p['symbol']:int(p['quantity']) for p in actual['positions'] if p['quantity']>0}
                        if qty!={s:q for s,q in plan['target_quantities'].items() if q>0} or abs(actual['cash']-plan['estimated_cash'])>.01:raise ValueError('ACTUAL_EXECUTION_MISMATCH')
                        values=pd.Series(qty)*pd.Series(opening_marks).reindex(qty);active=values/values.sum() @ (c-(wb @ c)).reindex(qty)
                        lo,hi=quality_upper_only_bounds(c,wb,limits,zero)
                        if not np.isfinite(active).all() or max((active-hi).max(),(lo-active).max())>1e-5:raise ValueError('ACTUAL_STYLE_BREACH')
                    events.append({'date':day,'status':plan['status'],'reason':plan.get('reason'),'orders':len(plan['orders'])})
                    (folder/f'plan-{day}.json').write_text(json.dumps(plan,indent=2,allow_nan=False))
                print(json.dumps({'account':account,'date':day,'event':events[-1]}),flush=True)
            for action in actions.loc[actions.record_date.eq(pd.Timestamp(day))].to_dict('records'):
                broker.register_corporate_entitlement('isolated',action['symbol'],action['action_key'],day,action['cash_per_share'],action['stock_ratio'],
                    action['cash_pay_date'].strftime('%Y%m%d') if pd.notna(action['cash_pay_date']) else None,action['stock_list_date'].strftime('%Y%m%d') if pd.notna(action['stock_list_date']) else None)
            closed=economic_snapshot(broker,broker.account_snapshot(closing_marks,'isolated'),closing_marks,'isolated',day,actions)
            equity.append({'date':day,'equity':closed['equity'],'cash':closed['cash'],'market_value':closed['market_value'],'receivable_cash':closed['receivable_cash'],'unsettled_stock_value':closed['unsettled_stock_value']});last.update(closing_marks)
        pd.DataFrame(equity).to_parquet(folder/'equity.parquet',index=False)
        with sqlite3.connect(folder/'paper.sqlite3') as con:
            fills=pd.read_sql_query('SELECT * FROM fills',con);ledger=pd.read_sql_query('SELECT * FROM corporate_action_ledger',con)
        from moneymore.research.replay_attribution import reconstruct_cash,reconstruct_positions
        dates=[e['date'] for e in equity];cash=reconstruct_cash(fills,ledger,dates,cfg.initial_cash);positions=reconstruct_positions(fills,ledger,dates)
        if any(abs(cash[e['date']]-e['cash'])>.01 for e in equity):raise ValueError('CASH_LEDGER_RECONSTRUCTION_FAILED')
        end=broker.account_snapshot(last,'isolated');actual_q={p['symbol']:int(p['quantity']) for p in end['positions'] if p['quantity']>0}
        if positions[dates[-1]]!=actual_q:raise ValueError('POSITION_LEDGER_RECONSTRUCTION_FAILED')
        paths=np.r_[cfg.initial_cash,[e['equity'] for e in equity]];dd=paths/np.maximum.accumulate(paths)-1
        row={'account':account,'model':model,'variant':variant,'fills':len(fills),'fees':float(fills.fee.sum()) if len(fills) else 0.,
            'net_return':float(paths[-1]/cfg.initial_cash-1),'max_drawdown':float(dd.min()),'end_cash':equity[-1]['cash'],'end_equity':equity[-1]['equity'],
            'end_positions':actual_q,'receivable_cash':equity[-1]['receivable_cash'],'unsettled_stock_value':equity[-1]['unsettled_stock_value'],
            'planned_rebalances':4,'passed_rebalances':sum(e['status']=='PASS_REFERENCE_QUOTES_ONLY' for e in events),'events':events,'ledger_reconciled':True}
        results.append(row);(args.output/'progress.json').write_text(json.dumps(results,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in row.items() if k!='events'}),flush=True)
    (args.output/'results.json').write_text(json.dumps({'status':'ISOLATED_COST_EPISODE_COMPLETE_NOT_RELEASED','cases':results,'orders_enabled':False,'adopted':False,
        'limitations':['one_episode_not_full_year_or_forward','blocked_rebalances_not_comparable_optimal_execution','gross_dividends_not_investor_after_tax',
            'risk_controls_use_limited_reference_and_available_named_factors_not_complete_Barra']},indent=2,allow_nan=False))


if __name__=='__main__':main()
