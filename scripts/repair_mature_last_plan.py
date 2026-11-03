"""Copy a completed isolated study; certify only its final blocked integer plan."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import yaml
from moneymore.config import BacktestConfig
from moneymore.execution.paper import PaperBroker, ExecutionBar
from moneymore.research.integer_execution import plan_integer_execution, paper_intents_from_integer_plan
from moneymore.research.quote_permissions import quote_permissions
from moneymore.research.replay_actions import economic_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];allowed=(root/'state/mature-multifactor').resolve()
    if not args.study.resolve().is_relative_to(allowed) or not args.output.resolve().is_relative_to(allowed):raise ValueError('ISOLATED_SCOPE_REQUIRED')
    if args.output.exists():raise FileExistsError('IMMUTABLE_REPAIR')
    original=json.loads((args.study/'results.json').read_bytes());case=next(c for c in original['cases'] if c['account']=='balanced')
    day=case['events'][-1]['date'];signal=case['events'][-1]['signal_date']
    old=json.loads((args.study/'balanced'/f'{day}-integer.json').read_bytes())
    if old['reason']!='INTEGER_FEASIBILITY_USER_LIMIT' or any(e['status']!='PASS_REFERENCE_QUOTES_ONLY' for e in case['events'][:-1]):raise ValueError('ONLY_FINAL_TIMEOUT_REPAIR_ALLOWED')
    shutil.copytree(args.study,args.output,ignore=shutil.ignore_patterns('independent-audit.json'))
    folder=args.output/'balanced';broker=PaperBroker(folder/'paper.sqlite3')
    import sqlite3
    with sqlite3.connect(broker.database) as connection:
        if connection.execute('SELECT COUNT(*) FROM corporate_entitlements WHERE record_date=?',(day,)).fetchone()[0]:raise ValueError('FINAL_RECORD_DATE_REQUIRES_FULL_REPLAY')
    protocol=yaml.safe_load((args.output/'protocol.yaml').read_text());config=BacktestConfig.model_validate({**BacktestConfig.from_yaml(root/'configs/default.yaml').model_dump(),
        'initial_cash':protocol['capital_each_episode'],'slippage_bps':protocol['slippage_bps_each_side']})
    binding=json.loads((args.output/'episode-binding.json').read_bytes());evidence=Path(binding['evidence'])
    market=pd.read_parquet(args.output/'market-history.parquet');quotes=market.loc[market.trade_date.astype(str).eq(day)].set_index('ts_code')
    marks=quotes.open.dropna().to_dict();closing=quotes.close.dropna().to_dict();snap=broker.account_snapshot(marks,'isolated')
    held={p['symbol']:int(p['quantity']) for p in snap['positions'] if p['quantity']>0};available={p['symbol']:int(p['available_quantity']) for p in snap['positions']}
    if set(held)-set(marks):raise ValueError('REPAIR_MARKS_NOT_IDENTICAL')
    controls=pd.read_parquet(folder/f'{day}-controls.parquet');benchmark=pd.read_parquet(folder/f'{day}-benchmark.parquet').weight
    continuous=json.loads((folder/f'{day}-continuous.json').read_bytes())
    suspension=pd.read_parquet(evidence/f'suspend_d-{day}.parquet');halted=set(suspension.loc[suspension.suspend_type.eq('S')&(suspension.suspend_timing.isna()|suspension.suspend_timing.eq('')),'ts_code']) if not suspension.empty else set()
    buy,sell=quote_permissions(marks,halted,pd.read_parquet(evidence/f'stk_limit-{day}.parquet').set_index('ts_code'),config.slippage_bps)
    plan=plan_integer_execution(continuous['weights'],marks,held,available,snap['cash'],snap['equity'],config,controls,continuous['named_style_limits'],
        can_buy=buy,can_sell=sell,benchmark_weights=benchmark,style_controls_prestandardized=True,
        time_limit=300,solver_threads=4,mip_relative_gap=.01,research_secondary_time_limit=10)
    (folder/f'{day}-integer-before-repair.json').write_text(json.dumps(old,indent=2))
    (folder/f'{day}-integer.json').write_text(json.dumps(plan,indent=2,allow_nan=False))
    if plan['status']!='PASS_REFERENCE_QUOTES_ONLY':raise ValueError('FINAL_REPAIR_NOT_CERTIFIED:'+str(plan.get('reason')))
    for intent in paper_intents_from_integer_plan(plan,'mature_balanced',signal):
        if broker.submit(intent,'isolated')!='PENDING':raise ValueError('REPAIR_DUPLICATE_ORDER')
    for order in plan['orders']:
        s=order['symbol'];broker.execute_pending(ExecutionBar(s,day,marks[s],marks[s],buy.get(s,False),sell.get(s,False)),config,'isolated')
    actual=broker.account_snapshot(marks,'isolated');q={p['symbol']:int(p['quantity']) for p in actual['positions'] if p['quantity']>0}
    if q!={s:n for s,n in plan['target_quantities'].items() if n>0} or abs(actual['cash']-plan['estimated_cash'])>.01:raise ValueError('REPAIR_EXECUTION_MISMATCH')
    actions=pd.read_parquet(Path(binding['audit'])/'official-actions.parquet')
    for action in actions.loc[actions.symbol.isin(q)&actions.record_date.eq(pd.Timestamp(day))].itertuples():
        broker.register_corporate_entitlement(account_id='isolated',symbol=action.symbol,action_key=action.action_key,
            record_date=day,cash_per_share=action.cash_per_share,stock_ratio=action.stock_ratio,
            cash_pay_date=None if pd.isna(action.cash_pay_date) else action.cash_pay_date.strftime('%Y%m%d'),
            stock_list_date=None if pd.isna(action.stock_list_date) else action.stock_list_date.strftime('%Y%m%d'))
    closed=economic_snapshot(broker,broker.account_snapshot(closing,'isolated'),closing,'isolated',day,actions)
    equity=pd.read_parquet(folder/'equity.parquet')
    for key in ['cash','equity','receivable_cash','unsettled_stock_value']:equity.loc[equity.date.eq(day),key]=closed[key]
    equity.to_parquet(folder/'equity.parquet',index=False)
    case['events'][-1]={'date':day,'signal_date':signal,'status':'PASS_REFERENCE_QUOTES_ONLY','reason':None,'orders':len(plan['orders']),
        'prior_attempt':case['events'][-1],'repair':'SAME_TARGETS_RISK_REFERENCE_QUOTES_AND_STATE_LONGER_COMPUTE_ONLY'}
    case.update(net_return=float(closed['equity']/config.initial_cash-1),max_drawdown=float((equity.equity/equity.equity.cummax().clip(lower=config.initial_cash)-1).min()),
        fills=len(broker.fills('isolated')),passed_rebalances=20,unsettled_stock_value=closed['unsettled_stock_value'])
    original['repair_parent_result_sha256']=hashlib.sha256((args.study/'results.json').read_bytes()).hexdigest()
    (args.output/'results.json').write_text(json.dumps(original,indent=2,allow_nan=False))
    print(json.dumps({'status':'FINAL_PLAN_EXECUTED_IN_COPIED_RESEARCH_LEDGER','orders':len(plan['orders']),'all_sessions':20,'real_orders_enabled':False}))


if __name__=='__main__':main()
