"""One bounded A/B paper release, gated by the completed fixed study and QA."""
from datetime import datetime,UTC
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib,json,sqlite3,argparse
import pandas as pd
from threadpoolctl import threadpool_limits
from filelock import FileLock
from prepare_four_strategy_2026 import ROOT,save
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.execution.paper_branch import fork_paper_account
from moneymore.cost_ab_runtime import prepare_cost_daily,VERSION
from moneymore.portfolio_candidate import execute_portfolio_candidate_open
from moneymore.challenger_diagnostics import trading_dates
OUT=ROOT/'state/cost-aware-ab-v2'

class IsolatedHistory:
    def __init__(self,base,output):self.base=base;self.output=output;self.curated=base.curated
    def read(self,*a,**kw):return self.base.read(*a,**kw)
    def merge_curated(self,*a,**kw):return self.output.merge_curated(*a,**kw)

def entries(day,next_day):
    original=json.loads((ROOT/'state/mature-multifactor/registry.json').read_bytes());accounts=[]
    for source,suffix,name in zip(original['accounts'],['CA','CB'],['A·成本保护','B·风格成本保护']):
        tag=day+suffix
        accounts.append({**source,'candidate_tag':tag,'account_id':'qlib_candidate_'+tag,'display_name':name,
            'cost_policy_version':VERSION,'activation_date':day,'first_execution_date':next_day,'created_at':datetime.now(UTC).isoformat(),
            'strategy_id':'cost_aware_v2_'+suffix,'source_account_id':source['account_id'],'comparison_reference_account':source['account_id'],
            'execution_policy':{'version':VERSION,'slippage_bps':10,'real_orders_enabled':False},
            'model_dir':original['model_dir'],'model_manifest':original['model_manifest'],'calibration_path':str(OUT/'t2/deployment.json')})
    return original,accounts

def preflight(day):
    base=ParquetStore(ROOT/'data');calendar=trading_dates(base,day);next_day=calendar[calendar.index(day)+1]
    original,accounts=entries(day,next_day)
    folder=next((OUT/'release-preflight'/f'{day}-{i}' for i in range(3) if not (OUT/'release-preflight'/f'{day}-{i}').exists()),None)
    if folder is None:raise ValueError('PREFLIGHT_RETRY_BUDGET_EXHAUSTED')
    folder.mkdir(parents=True);db=folder/'state/paper_orders.sqlite3';db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(f'file:{(ROOT/"state/paper_orders.sqlite3").as_posix()}?mode=ro',uri=True) as src,sqlite3.connect(db) as target:src.backup(target)
    broker=PaperBroker(db);store=IsolatedHistory(base,ParquetStore(folder/'data'));config=BacktestConfig.from_yaml(ROOT/'configs/default.yaml');results=[]
    for candidate in accounts:
        branch=fork_paper_account(broker,candidate['source_account_id'],candidate['account_id'],day)
        prepare_cost_daily(root=folder,input_root=ROOT,store=store,broker=broker,config=config,candidate=candidate,trade_date=day)
        plan=json.loads((folder/'state/qlib-portfolio-plans'/candidate['candidate_tag']/(day+'.json')).read_bytes())
        # Deliberately isolated fixture at known close, NOT evidence of next-day execution.
        bundles=[p for p in (ROOT/'state/mature-multifactor/runtime/features').glob(day+'-*') if (p/'results.json').exists()]
        marks=pd.read_parquet(sorted(bundles)[-1]/'bars-input.parquet').loc[lambda x:x.trade_date.astype(str).eq(day)].set_index('ts_code').close.to_dict()
        mock_time=datetime.strptime(next_day+'093100','%Y%m%d%H%M%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
        class FixtureQuotes:
            name='ISOLATED_CLOSE_PRICE_FIXTURE_NOT_LIVE'
            def intraday_quotes(self,symbols):return pd.DataFrame([{'symbol':s,'last':marks[s],'timestamp':mock_time.timestamp()*1000,
                'up_limit':marks[s]*2,'down_limit':marks[s]*.5,'execution_ready':True} for s in symbols])
        kwargs=dict(root=folder,store=store,broker=broker,config=config,candidate=candidate,provider=FixtureQuotes(),trade_date=next_day,now=mock_time)
        execution=execute_portfolio_candidate_open(**kwargs)
        if execution['status']!='COMPLETED' or not execution['reconciliation']['matched']:raise ValueError('COST_PREFLIGHT_EXECUTION_FAILED:'+candidate['candidate_tag']+':'+str(execution.get('plan',execution).get('reason',execution['status'])))
        before=len(broker.fills(candidate['account_id']));duplicate=execute_portfolio_candidate_open(**kwargs)
        assert duplicate['status']=='COMPLETED' and len(broker.fills(candidate['account_id']))==before
        results.append({'candidate_tag':candidate['candidate_tag'],'branch':branch,'signal_date':day,'fixture_trade_date':next_day,
            'fixture_only':True,'orders':len(execution['plan']['orders']),'cost_decision':execution['plan'].get('cost_decision'),'reconciled':True,'restart_idempotent':True})
    save(folder/'results.json',{'status':'PAPER_PREFLIGHT_PASS','accounts':results,'real_orders_enabled':False})
    return folder,original,accounts

def publish(day):
    path=OUT/'registry.json'
    if path.exists():raise ValueError('COST_PAIR_ALREADY_PUBLISHED_NO_DUPLICATE_RELEASE')
    audit=json.loads((OUT/'t5-audit/results.json').read_bytes());t4=json.loads((OUT/'t4/results.json').read_bytes())
    assert audit['status']=='SIX_COMPONENT_LEDGER_AUDIT_PASS' and len(audit['cases'])==6
    assert t4['status']=='ORIGINAL_44_DATES_DIAGNOSED' and len(t4['rows'])==44
    if any(r['resolution']=='REQUIRES_ADDITIONAL_INPUT_DIAGNOSIS' for r in t4['rows']):raise ValueError('UNRESOLVED_T4_CAUSE')
    qa=json.loads((OUT/'release-qa.json').read_bytes());assert qa['backend_passed'] and qa['frontend_typecheck_passed']
    folder,original,accounts=preflight(day)
    database=ROOT/'state/paper_orders.sqlite3';backup=OUT/('paper-before-release-'+datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.sqlite3')
    with sqlite3.connect(database) as src,sqlite3.connect(backup) as dst:src.backup(dst)
    broker=PaperBroker(database);receipts=[]
    for entry in accounts:
        receipts.append(fork_paper_account(broker,entry['source_account_id'],entry['account_id'],day))
        for name in ['model_dir','model_manifest','calibration_path']:entry.pop(name,None)
    hashes={**original['evidence_hashes']}
    files=[*sorted((ROOT/'src/moneymore/research').glob('cost_*.py')),
        *[ROOT/'src/moneymore'/p for p in ['cost_ab_runtime.py','cost_order_preview.py','execution/paper_branch.py','portfolio_candidate.py']],
        ROOT/'configs/default.yaml',OUT/'t2/deployment.json',OUT/'t5-audit/results.json',OUT/'t4/results.json',folder/'results.json',OUT/'release-qa.json']
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    registry={'schema':'COST_AWARE_PAPER_BRANCH_V1','version':VERSION,'accounts':accounts,'model_dir':original['model_dir'],
        'model_manifest':original['model_manifest'],'data_cutoff':original['data_cutoff'],'calibration_path':str(OUT/'t2/deployment.json'),
        'alpha_evidence_enabled':json.loads((OUT/'t2/deployment.json').read_bytes())['optional_alpha_enabled'],'evidence_hashes':hashes,
        'admission_scope':'PAPER_COST_AND_EXECUTION_PROTECTION_NOT_PROFIT_OR_MODEL_SUPERIORITY','promotion_authorized':False,'real_orders_enabled':False}
    # Reports contain actual branch state, never the fixture's future fills/NAV.
    store=ParquetStore(ROOT/'data');config=BacktestConfig.from_yaml(ROOT/'configs/default.yaml')
    for entry in accounts:
        private={**entry,'model_dir':original['model_dir'],'model_manifest':original['model_manifest'],'calibration_path':registry['calibration_path']}
        prepare_cost_daily(root=ROOT,store=store,broker=broker,config=config,candidate=private,trade_date=day)
        from moneymore.cost_order_preview import build_preview
        build_preview(ROOT,private,day)
    save(path,registry)
    save(OUT/'release-results.json',{'status':'PUBLISHED_INDEPENDENT_PAPER_BRANCHES','accounts':receipts,'signal_date':day,
        'preflight':str(folder/'results.json'),'real_trades_sent':0,'historical_returns_rewritten':False,'real_orders_enabled':False})
    print('PUBLISHED_INDEPENDENT_PAPER_BRANCHES',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--date',required=True);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    with FileLock(str(OUT/'release.lock'),timeout=1),threadpool_limits(limits=4):
        if args.publish:publish(args.date)
        else:preflight(args.date)
