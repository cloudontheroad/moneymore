"""Reference-only cost-aware orders: same decision as the next-open executor."""
from pathlib import Path
import hashlib,json
import pandas as pd

def build_preview(root,candidate,day):
    from .config import BacktestConfig
    from .data.store import ParquetStore
    from .execution.paper import PaperBroker
    from .research.style_v2_experiment import candidate_execution_config
    from .research.replay_actions import official_replay_actions,economic_snapshot
    from .cost_ab_runtime import execute_cost_plan,account_actions
    from .challenger_diagnostics import write_diagnostic
    path=root/'state/qlib-portfolio-plans'/candidate['candidate_tag']/(day+'.json');plan=json.loads(path.read_bytes())
    store=ParquetStore(root/'data');broker=PaperBroker(root/'state/paper_orders.sqlite3')
    bundle=next(p for p in reversed(sorted((root/'state/mature-multifactor/runtime/official').glob(day+'-*'))) if (p/'results.json').exists())
    manifest=json.loads((bundle/'results.json').read_bytes());source=bundle/'daily.parquet'
    if manifest['status']!='OFFICIAL_CLOSE_READY' or hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['hashes']['daily.parquet']:raise ValueError('INVALID_REFERENCE_CLOSE')
    marks=pd.read_parquet(source).set_index('ts_code').close.to_dict()
    config=candidate_execution_config(BacktestConfig.from_yaml(root/'configs/default.yaml'),candidate)
    snapshot=economic_snapshot(broker,broker.account_snapshot(marks,candidate['account_id']),marks,candidate['account_id'],day,account_actions(store,broker,candidate['account_id']))
    held={p['symbol']:p['quantity'] for p in snapshot['positions'] if p['quantity']>0};available={p['symbol']:p['available_quantity'] for p in snapshot['positions']}
    symbols=set(plan['target_weights'])|set(held)
    if {s for s in symbols if plan['target_weights'].get(s,0)>1e-6 or s in held}-set(marks):
        result={'status':'BLOCKED','reason':'REFERENCE_CLOSE_MISSING','orders':[]}
    else:
        result=execute_cost_plan(plan,marks,held,available,snapshot['cash'],snapshot['equity'],config,{s:s in marks for s in symbols},{s:s in marks for s in symbols})
    instruments=store.read('instruments').drop_duplicates('ts_code',keep='last').set_index('ts_code').to_dict('index')
    orders=[];industries={};boards={}
    for o in result.get('orders',[]):
        s=o['symbol'];meta=instruments.get(s,{});board='创业板' if s.startswith('30') else '科创板' if s.startswith('68') else '主板'
        orders.append({**o,'name':meta.get('name',s),'industry':meta.get('industry','未分类'),'board':board,
            'estimated_value':o['quantity']*o['reference_price'],'target_weight':plan['target_weights'].get(s,0),
            'estimated_weight':o['quantity']*o['reference_price']/snapshot['equity'],'signal_date':day,'planned_trade_date':plan['planned_trade_date'],'status':'待开盘核验（未提交）'})
    for s,q in result.get('target_quantities',{}).items():
        if q<=0:continue
        meta=instruments.get(s,{});industry=meta.get('industry','未分类');board='创业板' if s.startswith('30') else '科创板' if s.startswith('68') else '主板'
        weight=q*marks[s]/snapshot['equity'];industries[industry]=industries.get(industry,0)+weight;boards[board]=boards.get(board,0)+weight
    preview={'status':'REFERENCE_CLOSE_PREVIEW' if result['status']=='PASS_REFERENCE_QUOTES_ONLY' else 'PREVIEW_BLOCKED','reason':result.get('reason'),
        'account_id':candidate['account_id'],'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'signal_date':day,'planned_trade_date':plan['planned_trade_date'],
        'reference_date':day,'orders_submitted':False,'real_orders_enabled':False,'orders':orders,'industry_weights':industries,'board_weights':boards,
        'estimated_cash':result.get('estimated_cash'),'integer_plan':result,'cost_decision':result.get('cost_decision'),
        'notice':'成本保护模拟分支；分数仍连续，但校准未证明可交易优势时不主动轮换。必要风险和退出单独处理；参考价不代表次日成交。'}
    write_diagnostic(root/'state/qlib-order-previews'/candidate['candidate_tag']/(day+'.json'),preview)
    return preview
