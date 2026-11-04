import sqlite3
from pathlib import Path
import pytest
from moneymore.execution.paper import PaperBroker,ExecutionBar
from moneymore.execution.risk import OrderIntent,RiskResult
from moneymore.execution.paper_branch import fork_paper_account
from moneymore.models import Side
from moneymore.config import BacktestConfig

def test_fork_preserves_cash_shares_rights_and_parent_and_is_idempotent(tmp_path):
    broker=PaperBroker(tmp_path/'paper.sqlite3');broker.initialize_account(100000,'source')
    intent=OrderIntent(idempotency_key='source-buy',strategy_id='original',symbol='600000.SH',side=Side.BUY,quantity=100,signal_date='20261102',reason_code='TEST')
    broker.submit(RiskResult(True,intent,None),'source')
    broker.execute_pending(ExecutionBar('600000.SH','20261103',10,10,True,True),BacktestConfig.from_yaml(Path('configs/default.yaml')),'source')
    before=broker.account_snapshot({'600000.SH':11},'source')
    broker.register_corporate_entitlement(account_id='source',symbol='600000.SH',action_key='div',record_date='20261103',cash_per_share=.1,stock_ratio=.1,cash_pay_date='20261105',stock_list_date='20261106')
    receipt=fork_paper_account(broker,'source','qlib_candidate_new','20261104')
    assert fork_paper_account(broker,'source','qlib_candidate_new','20261104')==receipt
    after=broker.account_snapshot({'600000.SH':11},'qlib_candidate_new')
    assert (before['cash'],before['equity'],before['positions'])==(after['cash'],after['equity'],after['positions'])
    assert broker.reconcile('qlib_candidate_new').matched
    with sqlite3.connect(broker.database) as db:
        assert db.execute('SELECT COUNT(*) FROM paper_inherited_fills').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM corporate_entitlements WHERE account_id=?',('qlib_candidate_new',)).fetchone()[0]==1
    assert broker.account_snapshot({'600000.SH':11},'source')==before
    broker.settle_corporate_actions('qlib_candidate_new','20261106')
    assert broker.reconcile('qlib_candidate_new').matched
    assert broker.account_snapshot({'600000.SH':11},'source')==before

def test_unresolved_parent_order_blocks_fork(tmp_path):
    broker=PaperBroker(tmp_path/'paper.sqlite3');broker.initialize_account(100000,'source')
    intent=OrderIntent(idempotency_key='pending',strategy_id='original',symbol='600000.SH',side=Side.BUY,quantity=100,signal_date='20261103',reason_code='TEST')
    broker.submit(RiskResult(True,intent,None),'source')
    with pytest.raises(ValueError,match='PENDING'):fork_paper_account(broker,'source','qlib_candidate_new','20261104')
    with sqlite3.connect(broker.database) as db:assert not db.execute('SELECT 1 FROM accounts WHERE account_id=?',('qlib_candidate_new',)).fetchone()
