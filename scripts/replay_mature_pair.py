"""One fixed 20-session native-model portfolio comparison in isolated ledgers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker, ExecutionBar
from moneymore.research.enhanced_indexing import EnhancedPolicy, optimize_snapshot
from moneymore.research.mature_multifactor import current_risk_inputs, topk_dropout_targets
from moneymore.research.integer_execution import plan_integer_execution, paper_intents_from_integer_plan
from moneymore.research.quote_permissions import quote_permissions
from moneymore.research.replay_actions import economic_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_MATURE_PAIR_REPLAY')
    evidence=json.loads((args.evidence/'results.json').read_bytes())
    if evidence['status']!='READY' or not (args.evidence/'audit/results.json').exists():raise ValueError('OFFICIAL_EXECUTION_EVIDENCE_REQUIRED')
    for item in evidence['requests']:
        if hashlib.sha256((args.evidence/item['file']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('OFFICIAL_INPUT_CHANGED')
    args.output.mkdir(parents=True);root=Path(__file__).resolve().parents[1];store=ParquetStore(root/'data')
    model=Path(evidence['model']);manifest=json.loads((model/'results.json').read_bytes());inputs=Path(manifest['inputs'])
    scores=pd.read_parquet(model/'test-predictions.parquet').score
    bars=pd.read_parquet(inputs/'bars-input.parquet');quotes=bars.set_index(['trade_date','ts_code'])
    actions=pd.read_parquet(args.evidence/'audit/official-actions.parquet')
    proofs=pd.concat([pd.read_parquet(args.evidence/f'suspend_d-{day}.parquet') for day in evidence['sessions']],ignore_index=True)
    membership=pd.read_parquet(inputs/'membership.parquet')
    config=BacktestConfig.model_validate({**BacktestConfig.from_yaml(root/'configs/default.yaml').model_dump(),'initial_cash':10_000_000,'slippage_bps':10})
    protocol={'capital_each_episode':config.initial_cash,'slippage_bps_each_side':10,'commission_rate':config.commission_rate,
        'minimum_commission':config.minimum_commission,'stamp_duty_rate':config.stamp_duty_rate,'transfer_fee_rate':config.transfer_fee_rate}
    (args.output/'protocol.yaml').write_text(yaml.safe_dump(protocol))
    (args.output/'episode-binding.json').write_text(json.dumps({'evidence':str(args.evidence.resolve()),'audit':str((args.evidence/'audit').resolve()),
        'manifest_sha256':hashlib.sha256((args.evidence/'manifest.json').read_bytes()).hexdigest(),
        'audit_sha256':hashlib.sha256((args.evidence/'audit/results.json').read_bytes()).hexdigest()}))
    bars.to_parquet(args.output/'market-history.parquet',index=False)
    books={};equities={};events={}
    for name in ['raw','balanced']:
        directory=args.output/name;directory.mkdir();books[name]=PaperBroker(directory/'paper.sqlite3');books[name].initialize_account(config.initial_cash,'isolated');equities[name]=[];events[name]=[]
    last_marks=quotes.loc[evidence['signals'][0]].close.dropna().to_dict()
    for signal,day in zip(evidence['signals'],evidence['sessions']):
        signal_scores=scores.xs(pd.Timestamp(signal),level='datetime')
        members=set(membership.loc[membership.datetime.eq(pd.Timestamp(signal)),'instrument'])
        prices=quotes.loc[day].open.dropna().to_dict();close=quotes.loc[day].close.dropna().to_dict();prior=quotes.loc[signal].close.dropna().to_dict()
        suspension=pd.read_parquet(args.evidence/f'suspend_d-{day}.parquet')
        halted=set(suspension.loc[suspension.suspend_type.eq('S')&(suspension.suspend_timing.isna()|suspension.suspend_timing.eq('')),'ts_code']) if not suspension.empty else set()
        for symbol in halted-set(prices):
            if symbol not in last_marks:continue
            value=last_marks[symbol]
            for action in actions.loc[actions.symbol.eq(symbol)&actions.ex_date.eq(pd.Timestamp(day))].itertuples():
                value=(value-action.cash_per_share)/(1+action.stock_ratio)
            prices[symbol]=value;close[symbol]=value
        gate=pd.read_parquet(args.evidence/f'stk_limit-{day}.parquet').set_index('ts_code')
        buy,sell=quote_permissions(prices,halted,gate,10)
        for name,broker in books.items():
            broker.settle_corporate_actions('isolated',day);broker.settle_available_positions('isolated',day)
            snap=economic_snapshot(broker,broker.account_snapshot(prices,'isolated'),prices,'isolated',day,actions);held={p['symbol']:int(p['quantity']) for p in snap['positions'] if p['quantity']>0};available={p['symbol']:int(p['available_quantity']) for p in snap['positions']}
            exits=set(held)-members
            risk_scores=signal_scores.reindex(sorted(set(signal_scores.index)|set(held)))
            returns,size,raw,controls,benchmark,limits,risk_audit=current_risk_inputs(store,bars,risk_scores,signal,
                suspension_proof=proofs,candidate_members=members)
            controls.to_parquet(args.output/name/f'{day}-controls.parquet')
            benchmark.to_frame('weight').to_parquet(args.output/name/f'{day}-benchmark.parquet')
            eligible=signal_scores.reindex(controls.index.intersection(signal_scores.index))
            event={'date':day,'signal_date':signal,'universe_exits':sorted(exits)}
            if set(held)-set(prices) or set(held)-set(controls.index) or (set(held)&members)-set(eligible.index) or snap['unsettled_stock_rights']:
                event.update(status='BLOCKED',reason='HELD_PRICE_OR_RISK_MISSING_NOT_AN_EXIT')
            else:
                old_values=pd.Series({s:q*prices[s] for s,q in held.items()},dtype=float)
                if name=='raw':
                    weights=old_values/snap['equity'];locked_exits=weights.reindex(sorted(exits&halted)).dropna();free=1-float(locked_exits.sum())
                    targets={s:w*free for s,w in topk_dropout_targets(eligible,weights.drop(index=list(exits),errors='ignore')/free,
                        topk=50-len(locked_exits)).items()};targets.update(locked_exits.to_dict())
                    integer_styles=pd.DataFrame({'execution_only':0.},index=controls.index);integer_limits={'execution_only':0.}
                else:
                    model_signal=risk_scores.reindex(controls.index)
                    # Missing scores on explicit exits are not a predicted zero:
                    # the native optimizer fixes tradable exits to zero weight.
                    model_signal.loc[list(exits&set(model_signal.index))]=0.
                    solved=optimize_snapshot(model_signal,returns,size,old_values/old_values.sum() if len(old_values) else old_values,
                        EnhancedPolicy(objective_preconditioning=True),style_controls=controls,style_limits=limits,
                        benchmark_weights=benchmark,controls_prestandardized=True,allow_cash_bootstrap=True,reject_turnover_relaxation=True,
                        forced_sell_symbols=exits-halted,forced_hold_symbols=set(held)&halted,solver_max_iters=1000)
                    (args.output/name/f'{day}-continuous.json').write_text(json.dumps(solved,indent=2,allow_nan=False))
                    targets=solved['weights'];integer_styles=controls;integer_limits=limits
                if not targets:
                    event.update(status='BLOCKED',reason='CONTINUOUS_TARGET_NOT_AVAILABLE')
                else:
                    integer=plan_integer_execution(targets,prices,held,available,snap['cash'],snap['equity'],config,integer_styles,integer_limits,
                        can_buy=buy,can_sell=sell,benchmark_weights=benchmark,style_controls_prestandardized=True,
                        time_limit=60,solver_threads=4,mip_relative_gap=.01,research_secondary_time_limit=10)
                    (args.output/name/f'{day}-integer.json').write_text(json.dumps(integer,indent=2,allow_nan=False))
                    event.update(status=integer['status'],reason=integer.get('reason'))
                    if integer['status']=='PASS_REFERENCE_QUOTES_ONLY':
                        for intent in paper_intents_from_integer_plan(integer,'mature_'+name,signal):
                            if broker.submit(intent,'isolated')!='PENDING':raise ValueError('PAPER_SUBMIT_MISMATCH')
                        for order in integer['orders']:
                            symbol=order['symbol'];broker.execute_pending(ExecutionBar(symbol,day,prices[symbol],prices[symbol],buy.get(symbol,False),sell.get(symbol,False)),config,'isolated')
                        actual=broker.account_snapshot(prices,'isolated');actual_quantities={p['symbol']:int(p['quantity']) for p in actual['positions'] if p['quantity']>0}
                        expected={s:q for s,q in integer['target_quantities'].items() if q>0}
                        if actual_quantities!=expected or abs(actual['cash']-integer['estimated_cash'])>.01:raise ValueError('ACTUAL_LEDGER_NOT_INTEGER_PLAN')
                        event['orders']=len(integer['orders'])
            at_close=broker.account_snapshot(close,'isolated');owned={p['symbol'] for p in at_close['positions'] if p['quantity']>0}
            for action in actions.loc[actions.symbol.isin(owned)&actions.record_date.eq(pd.Timestamp(day))].itertuples():
                broker.register_corporate_entitlement(account_id='isolated',symbol=action.symbol,action_key=action.action_key,
                    record_date=day,cash_per_share=action.cash_per_share,stock_ratio=action.stock_ratio,
                    cash_pay_date=None if pd.isna(action.cash_pay_date) else action.cash_pay_date.strftime('%Y%m%d'),
                    stock_list_date=None if pd.isna(action.stock_list_date) else action.stock_list_date.strftime('%Y%m%d'))
            at_close=economic_snapshot(broker,at_close,close,'isolated',day,actions)
            if not broker.reconcile('isolated').matched:raise ValueError('PAPER_RECONCILIATION_FAILED')
            equities[name].append({'date':day,**{k:at_close[k] for k in ['cash','equity','receivable_cash','unsettled_stock_value']}})
            events[name].append(event);pd.DataFrame(equities[name]).to_parquet(args.output/name/'equity.parquet',index=False)
        last_marks.update(close)
        (args.output/'progress.json').write_text(json.dumps({'completed_sessions':len(equities['raw']),'total_sessions':20,'events':events},indent=2))
        print(json.dumps({'date':day,'raw':events['raw'][-1]['status'],'balanced':events['balanced'][-1]['status']}),flush=True)
    cases=[]
    for name,broker in books.items():
        equity=pd.DataFrame(equities[name]);last=equity.iloc[-1]
        cases.append({'account':name,'net_return':float(last.equity/config.initial_cash-1),'max_drawdown':float((equity.equity/equity.equity.cummax().clip(lower=config.initial_cash)-1).min()),
            'fills':len(broker.fills('isolated')),'planned_rebalances':20,'passed_rebalances':sum(e['status']=='PASS_REFERENCE_QUOTES_ONLY' for e in events[name]),
            'unsettled_stock_value':float(last.unsettled_stock_value),'events':events[name]})
    (args.output/'results.json').write_text(json.dumps({'status':'P3_FIXED_REPLAY_COMPLETE_REQUIRES_INDEPENDENT_AUDIT','cases':cases,
        'terminal_policy':'mark_to_market_no_forced_liquidation','real_orders_enabled':False,'same_forecast_and_cost':True},indent=2,allow_nan=False))
    print(json.dumps([{k:v for k,v in c.items() if k!='events'} for c in cases]))


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):main()
