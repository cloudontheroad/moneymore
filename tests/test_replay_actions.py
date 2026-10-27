import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.execution_replay import run_execution_replay
from moneymore.research.replay_actions import official_replay_actions


def event(**changes):
    return dict(ts_code="AAA", end_date="20251231", div_proc="实施", cash_div_tax=1.,
                stk_div=0., stk_bo_rate=None, stk_co_rate=None, record_date="20260106",
                ex_date="20260107", pay_date="20260109", div_listdate=None,
                imp_ann_date="20260101", **changes)


def config():
    return BacktestConfig(initial_cash=10000., commission_rate=0., minimum_commission=0.,
        stamp_duty_rate=0., transfer_fee_rate=0., slippage_bps=0., lot_size=100,
        max_position_weight=1., max_gross_exposure=1., max_drawdown=.5)


def replay(tmp_path, action, prices, targets=None, provider=None):
    bars = pd.DataFrame([dict(date=d, symbol="AAA", open=p, close=p) for d, p in prices])
    if targets is None:
        targets = pd.DataFrame([dict(date="20260105", symbol="AAA", target_weight=1.)])
    return run_execution_replay(bars=bars, targets=targets, config=config(),
        database=tmp_path / "research.sqlite3", strategy_id="test", account_id="test",
        corporate_actions=official_replay_actions(pd.DataFrame([action])),
        accrue_action_receivables=True, closing_target_provider=provider)


def test_cash_accrues_only_from_ex_date_and_settles_without_double_count(tmp_path):
    result = replay(tmp_path, event(), [("20260105", 10), ("20260106", 10),
        ("20260107", 9), ("20260108", 9), ("20260109", 9)])
    equity = result.equity.set_index("trade_date")
    assert equity.loc["20260106", "receivable_cash"] == 0
    assert equity.loc["20260107", "receivable_cash"] == 1000
    assert equity.loc["20260108", "cash"] == 0
    assert equity.loc["20260109", "receivable_cash"] == 0
    assert equity.loc["20260109", "cash"] == 1000
    assert equity.equity.tolist() == [10000.] * 5
    assert result.reconciliation["matched"]


def test_stock_rights_valued_before_listing_not_spendable(tmp_path):
    action = event()
    action.update(cash_div_tax=0., stk_div=1., stk_co_rate=1.,
                  pay_date=None, div_listdate="20260109")
    result = replay(tmp_path, action, [("20260105", 10), ("20260106", 10),
        ("20260107", 5), ("20260108", 5), ("20260109", 5)])
    equity = result.equity.set_index("trade_date")
    assert equity.loc["20260107", "unsettled_stock_value"] == 5000
    assert equity.loc["20260109", "unsettled_stock_value"] == 0
    assert equity.equity.tolist() == [10000.] * 5
    assert result.reconciliation["actual_quantity"] == 2000
    assert result.reconciliation["matched"]


def test_total_stock_ratio_not_added_to_components_and_conflicts_rejected():
    action = event()
    action.update(stk_div=.4, stk_co_rate=.4, div_listdate="20260109")
    assert official_replay_actions(pd.DataFrame([action])).iloc[0].stock_ratio == .4
    action["stk_co_rate"] = .3
    with pytest.raises(ValueError, match="CONFLICTING_STOCK"):
        official_replay_actions(pd.DataFrame([action]))


def test_unimplemented_and_future_known_actions_not_accepted():
    action = event()
    action["div_proc"] = "预案"
    assert official_replay_actions(pd.DataFrame([action])).empty
    action.update(div_proc="实施", imp_ann_date="20260108")
    with pytest.raises(ValueError, match="TIMELINE"):
        official_replay_actions(pd.DataFrame([action]))


def test_dividend_right_survives_sale_after_record_date(tmp_path):
    targets = pd.DataFrame([dict(date="20260105", symbol="AAA", target_weight=1.),
                            dict(date="20260106", symbol="AAA", target_weight=0.)])
    result = replay(tmp_path, event(), [("20260105", 10), ("20260106", 10),
        ("20260107", 9), ("20260108", 9), ("20260109", 9)], targets=targets)
    equity = result.equity.set_index("trade_date")
    assert equity.loc["20260107", "cash"] == 9000
    assert equity.loc["20260107", "receivable_cash"] == 1000
    assert equity.loc["20260109", "cash"] == 10000
    assert result.reconciliation["actual_quantity"] == 0
    assert result.reconciliation["matched"]


def test_locked_stock_rights_do_not_trigger_duplicate_purchase(tmp_path):
    action = event()
    action.update(cash_div_tax=0., stk_div=1., stk_co_rate=1.,
                  pay_date=None, div_listdate="20260109")
    snapshots = {}

    def provider(date, snapshot):
        snapshots[date] = snapshot
        return pd.DataFrame([dict(date=date, symbol="AAA", target_weight=1.)])

    result = replay(tmp_path, action, [("20260105", 10), ("20260106", 10),
        ("20260107", 5), ("20260108", 5), ("20260109", 5)],
        targets=pd.DataFrame(columns=["date", "symbol", "target_weight"]), provider=provider)
    assert snapshots["20260107"]["cash"] == 0
    assert snapshots["20260107"]["positions"][0]["quantity"] == 1000
    assert snapshots["20260107"]["unsettled_stock_rights"] == {"AAA": 1000}
    assert len(result.fills) == 1
    assert result.reconciliation["actual_quantity"] == 2000
