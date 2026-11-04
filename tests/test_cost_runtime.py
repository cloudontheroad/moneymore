from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker,ExecutionBar
from moneymore.execution.risk import OrderIntent,RiskResult
from moneymore.models import Side
from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.style_v2_experiment import execution_policy_hash
from moneymore.portfolio_candidate import execute_portfolio_candidate_open
from moneymore.cost_ab_runtime import cost_candidates

def test_compensation_budget_survives_restart_and_is_bounded(tmp_path):
    from moneymore.cost_ab_runtime import reserve_execution_attempt
    assert reserve_execution_attempt(tmp_path,'tag','20261104',1000) is None
    assert reserve_execution_attempt(tmp_path,'tag','20261104',1100)=='COST_EXECUTION_COMPENSATION_COOLDOWN'
    assert reserve_execution_attempt(tmp_path,'tag','20261104',1300) is None
    assert reserve_execution_attempt(tmp_path,'tag','20261104',1600) is None
    assert reserve_execution_attempt(tmp_path,'tag','20261104',1900)=='COST_EXECUTION_BUDGET_EXHAUSTED'
    assert reserve_execution_attempt(tmp_path,'tag','20261105',1900) is None

def test_absent_cost_registry_is_inert(tmp_path):
    assert cost_candidates(tmp_path)==[]
    assert not (tmp_path/'state').exists()

def test_cost_candidate_exposes_actual_block_not_only_reference_preview(tmp_path):
    account='qlib_candidate_20261104CA';broker=PaperBroker(tmp_path/'state/paper_orders.sqlite3');broker.initialize_account(10000,account)
    write_diagnostic(tmp_path/'state/cost-aware-ab-v2/registry.json',{'schema':'COST_AWARE_PAPER_BRANCH_V1','real_orders_enabled':False,
        'evidence_hashes':{},'accounts':[{'candidate_tag':'20261104CA','account_id':account,'initial_cash':10000,'activation_date':'20261104'}],
        'model_dir':'fixture','model_manifest':'fixture','calibration_path':'fixture','data_cutoff':'20261104','alpha_evidence_enabled':False})
    write_diagnostic(tmp_path/'state/qlib-portfolio-execution/20261104CA/20261105.json',{'status':'QUOTE_QUALITY_BLOCKED','trade_date':'20261105'})
    rows=cost_candidates(tmp_path)
    assert rows[0]['latest_execution']['status']=='QUOTE_QUALITY_BLOCKED'
    assert broker.fills(account)==[]

def test_full_cost_hold_quote_refresh_reconcile_and_restart(tmp_path,monkeypatch):
    monkeypatch.setattr('moneymore.portfolio_candidate.trading_dates',lambda *a:['20261103','20261104'])
    symbols=[f'{600000+i}.SH' for i in range(10)];account='qlib_candidate_20261104CA';tag='20261104CA'
    config=BacktestConfig(initial_cash=100000,commission_rate=.00015,minimum_commission=5,transfer_fee_rate=.00001,
        stamp_duty_rate=.0005,slippage_bps=10,lot_size=100,max_position_weight=.15,max_gross_exposure=1.,max_drawdown=.15)
    broker=PaperBroker(tmp_path/'state/paper_orders.sqlite3');broker.initialize_account(100000,account)
    for s in symbols:
        order=OrderIntent(idempotency_key=s,strategy_id='parent',symbol=s,side=Side.BUY,quantity=900,signal_date='20261102',reason_code='TEST_PARENT')
        broker.submit(RiskResult(True,order,None),account)
        broker.execute_pending(ExecutionBar(s,'20261103',10,10,True,True),config,account)
    candidate={'candidate_tag':tag,'account_id':account,'cost_policy_version':'cost_aware_ab_v2','portfolio_variant':'mature_alpha158_raw',
        'initial_cash':100000,'execution_policy':{'version':'cost-v2','slippage_bps':10,'real_orders_enabled':False}}
    plan={'signal_date':'20261103','strategy_id':'cost-v2','bootstrap':False,'freeze_budget':False,'target_weights':dict.fromkeys(symbols,.09),
        'styles':{s:{'none':0.} for s in symbols},'style_limits':{'none':0.},'benchmark_weights':None,'locked_shares':{},
        'execution_policy_sha256':execution_policy_hash(candidate),'execution_slippage_bps':10,
        'economics':{'mu':dict.fromkeys(symbols,0.),'risk_model':None,'economic_tracking_weights':dict.fromkeys(symbols,0.),'preserve_quantities':True,'alpha_enabled':False}}
    write_diagnostic(tmp_path/'state/qlib-portfolio-plans'/tag/'20261103.json',plan)
    store=ParquetStore(tmp_path/'data');store.merge_curated('dividend',[pd.DataFrame([{'ts_code':'unused','div_proc':'预案'}])],['ts_code'])
    now=datetime(2026,11,4,9,31,tzinfo=ZoneInfo('Asia/Shanghai'))
    class Provider:
        name='fixture';calls=0
        def intraday_quotes(self,names):
            self.calls+=1
            return pd.DataFrame([{'symbol':s,'last':10.,'timestamp':now.timestamp()*1000,'execution_ready':True,'up_limit':11.,'down_limit':9.} for s in names])
    provider=Provider();before=broker.account_snapshot(dict.fromkeys(symbols,10.),account);fills=len(broker.fills(account))
    args=dict(root=tmp_path,store=store,broker=broker,config=config,candidate=candidate,provider=provider,trade_date='20261104',now=now)
    result=execute_portfolio_candidate_open(**args)
    assert result['status']=='COMPLETED',result
    assert result['plan']['cost_decision']=='COST_NO_TRADE'
    assert result['plan']['fresh_quote_revalidated'] and provider.calls==2
    assert result['reconciliation']['matched'] and len(broker.fills(account))==fills
    assert result['portfolio']['cash']==before['cash']
    assert [(p['symbol'],p['quantity'],p['avg_cost']) for p in result['portfolio']['positions']]==[(p['symbol'],p['quantity'],p['avg_cost']) for p in before['positions']]
    assert all(p['available_quantity']==900 for p in result['portfolio']['positions'])
    assert execute_portfolio_candidate_open(**args)['status']=='COMPLETED' and provider.calls==2
