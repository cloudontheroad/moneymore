from types import SimpleNamespace
import pandas as pd
import pytest
from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.portfolio_candidate import prepare_portfolio_candidate


@pytest.mark.parametrize('still_member',[False,True])
def test_exited_stock_is_not_confused_with_missing_forecast(tmp_path,monkeypatch,still_member):
    held='EXIT';candidate=dict(candidate_tag='x',account_id='a')
    broker=SimpleNamespace(account_snapshot=lambda *a:dict(positions=[dict(symbol=held,quantity=100)],equity=2000,cash=1000))
    monkeypatch.setattr('moneymore.strategy_universe.active_strategy_universe',lambda *a:pd.DataFrame({'symbol':['A']+([held] if still_member else [])}))
    captured={}
    def risk(store,scores,day):
        captured['scores']=scores
        return pd.DataFrame({'A':[.01],held:[.02]}),pd.Series({'A':1.,held:1.}),pd.DataFrame({'style':[0.,1.]},index=['A',held])
    monkeypatch.setattr('moneymore.portfolio_candidate.risk_snapshot',risk)
    def optimize(*a,**kw):
        captured['forced']=kw['forced_sell_symbols']
        return dict(status='BLOCKED',failures=['TEST_STOP_BEFORE_WRITES'])
    monkeypatch.setattr('moneymore.portfolio_candidate.optimize_snapshot',optimize)
    write_diagnostic(tmp_path/'state/qlib-portfolio-budgets/x.json',dict(limits={'style':1.}))
    with pytest.raises(ValueError,match='HELD_SCORE_MISSING_NOT_EXIT' if still_member else 'TEST_STOP_BEFORE_WRITES'):
        prepare_portfolio_candidate(root=tmp_path,store=None,broker=broker,config=None,candidate=candidate,
            trade_date='20261110',score_frame=pd.DataFrame({'instrument':['A'],'score':[1.]}),
            bootstrap_weights={},marks={held:10.,'A':10.},strategy_id='s',model_id='m')
    if not still_member:
        assert captured['scores'][held]==0. and captured['forced']=={held}
    else:
        assert not captured
