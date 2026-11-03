import pytest
from moneymore.config import BacktestConfig
from moneymore.models import Side
from moneymore.execution.paper import PaperBroker,ExecutionBar
from moneymore.execution.risk import OrderIntent,RiskResult


@pytest.mark.parametrize('previous_session',[True,False])
def test_predeclared_exit_uses_prior_signal_and_respects_t1(tmp_path,previous_session):
    cfg=BacktestConfig(initial_cash=100_000,commission_rate=.00015,minimum_commission=5,stamp_duty_rate=.0005,
        transfer_fee_rate=.00001,slippage_bps=10,lot_size=100,max_position_weight=1.,max_gross_exposure=1.,max_drawdown=.2)
    broker=PaperBroker(tmp_path/'exit.sqlite3');broker.initialize_account(cfg.initial_cash,'isolated')
    buy=OrderIntent('initial','test','AAA',Side.BUY,100,'20240628','ENTRY')
    broker.submit(RiskResult(True,buy,None),'isolated')
    assert len(broker.execute_pending(ExecutionBar('AAA','20240701',10.,10.),cfg,'isolated'))==1
    broker.settle_available_positions('isolated','20240702')
    signal='20240701' if previous_session else '20240702'
    sell=OrderIntent('terminal','test','AAA',Side.SELL,100,signal,'PREDECLARED_TERMINAL_EXIT')
    broker.submit(RiskResult(True,sell,None),'isolated')
    fills=broker.execute_pending(ExecutionBar('AAA','20240702',10.,10.),cfg,'isolated')
    assert len(fills)==(1 if previous_session else 0)
