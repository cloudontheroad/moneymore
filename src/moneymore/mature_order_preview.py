"""Reference-close order previews, separate from submitted paper orders."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_preview(root, candidate):
    if not str(candidate.get('portfolio_variant','')).startswith('mature_alpha158_'):return None
    report=candidate.get('latest_report',{});day=report.get('signal_date')
    if not day:return None
    path=root/'state/qlib-order-previews'/candidate['candidate_tag']/(day+'.json')
    plan=root/'state/qlib-portfolio-plans'/candidate['candidate_tag']/(day+'.json')
    if not path.exists() or not plan.exists():return None
    preview=json.loads(path.read_bytes())
    if preview.get('plan_sha256')!=hashlib.sha256(plan.read_bytes()).hexdigest():return None
    if (candidate.get('lifecycle') or {}).get('model_quality_hold'):
        preview.update(status='MODEL_QUALITY_HOLD',model_quality_hold=True,
            notice='模型预测区分度未通过，旧计划已暂停，不可执行；不代表明日将按此买入。')
        preview['orders']=[{**row,'status':'模型暂停，旧计划不可执行'} for row in preview.get('orders',[])]
    return preview


def build_preview(root, candidate, day):
    from .config import BacktestConfig
    from .data.store import ParquetStore
    from .execution.paper import PaperBroker
    from .research.integer_execution import plan_integer_execution
    from .research.style_v2_experiment import candidate_execution_config
    from .challenger_diagnostics import write_diagnostic
    plan_path=root/'state/qlib-portfolio-plans'/candidate['candidate_tag']/(day+'.json')
    plan=json.loads(plan_path.read_bytes());digest=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    output=root/'state/qlib-order-previews'/candidate['candidate_tag']/(day+'.json')
    if output.exists():
        old=json.loads(output.read_bytes())
        if old.get('plan_sha256')==digest:return old
    store=ParquetStore(root/'data');broker=PaperBroker(root/'state/paper_orders.sqlite3')
    bundle=sorted((root/'state/mature-multifactor/runtime/official').glob(day+'-*'))[-1]
    manifest=json.loads((bundle/'results.json').read_bytes());source=bundle/'daily.parquet'
    if manifest['status']!='OFFICIAL_CLOSE_READY' or hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['hashes']['daily.parquet']:raise ValueError('INVALID_REFERENCE_CLOSE')
    marks=pd.read_parquet(source).set_index('ts_code').close.to_dict()
    snapshot=broker.account_snapshot(marks,candidate['account_id']);config=candidate_execution_config(BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate)
    held={p['symbol']:p['quantity'] for p in snapshot['positions']};available={p['symbol']:p['available_quantity'] for p in snapshot['positions']}
    styles=pd.DataFrame.from_dict(plan['styles'],orient='index')
    symbols=set(plan['target_weights'])|set(held)
    # Next-session limits, suspensions and quotes are unknown: do NOT pretend these are executable quotes.
    result=plan_integer_execution(plan['target_weights'],marks,held,available,snapshot['cash'],snapshot['equity'],config,styles,plan['style_limits'],
        can_buy={s:s in marks for s in symbols},can_sell={s:s in marks for s in symbols},
        benchmark_weights=plan['benchmark_weights'],style_controls_prestandardized=True,time_limit=300,solver_threads=4,mip_relative_gap=.01,research_secondary_time_limit=10)
    instruments=store.read('instruments').drop_duplicates('ts_code',keep='last').set_index('ts_code').to_dict('index')
    orders=[];industries={};boards={}
    for order in result.get('orders',[]):
        symbol=order['symbol'];meta=instruments.get(symbol,{})
        board='创业板' if symbol.startswith('30') else '科创板' if symbol.startswith('68') else '沪市主板' if symbol.endswith('.SH') else '深市主板'
        value=order['quantity']*order['reference_price'];weight=value/snapshot['equity'];industry=str(meta.get('industry') or '未分类')
        orders.append({**order,'name':meta.get('name',symbol),'industry':industry,'board':board,'estimated_value':value,
            'target_weight':plan['target_weights'].get(symbol,0),'estimated_weight':weight,'signal_date':day,
            'planned_trade_date':plan['planned_trade_date'],'status':'待开盘核验（未提交）'})
    for symbol,quantity in result.get('target_quantities',{}).items():
        if quantity<=0:continue
        meta=instruments.get(symbol,{});industry=str(meta.get('industry') or '未分类');board='创业板' if symbol.startswith('30') else '科创板' if symbol.startswith('68') else '主板'
        weight=quantity*marks[symbol]/snapshot['equity'];industries[industry]=industries.get(industry,0)+weight;boards[board]=boards.get(board,0)+weight
    preview={'status':'REFERENCE_CLOSE_PREVIEW' if result['status']=='PASS_REFERENCE_QUOTES_ONLY' else 'PREVIEW_BLOCKED',
        'reason':result.get('reason'),'account_id':candidate['account_id'],'plan_sha256':digest,'signal_date':day,'planned_trade_date':plan['planned_trade_date'],
        'reference_date':day,'orders_submitted':False,'real_orders_enabled':False,'orders':sorted(orders,key=lambda r:-r['estimated_value']),
        'industry_weights':industries,'board_weights':boards,'estimated_cash':result.get('estimated_cash'),'integer_plan':result,
        'notice':'按前日收盘价及10bps滑点测算；次日行情、停牌和涨跌停未知，开盘重新核验和计算，数量与价格可能改变。'}
    write_diagnostic(output,preview)
    return preview


if __name__=='__main__':
    import argparse
    from .mature_multifactor_runtime import mature_candidates
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser();parser.add_argument('--date',required=True);args=parser.parse_args();root=Path.cwd()
    with threadpool_limits(limits=4):
        for candidate in mature_candidates(root):
            result=build_preview(root,candidate,args.date)
            print(json.dumps({'account':candidate['account_id'],'status':result['status'],'orders':len(result['orders']),'boards':result['board_weights'],'industries':result['industry_weights']},ensure_ascii=True),flush=True)
