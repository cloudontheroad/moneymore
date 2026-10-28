import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.execution_replay import run_execution_replay
from moneymore.research.baseline_v2_continuous_replay import (
    EMPTY,
    MonthlyOpeningPlans,
    audit_ledger,
    prepare_market,
)


def test_carry_is_valuation_only_and_requires_proof():
    f = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "trade_date": ["20240131", "20240201"],
            "open": [10.0, float("nan")],
            "close": [10.0, float("nan")],
            "not_yet_listed": [False, False],
            "price_ok": [True, False],
            "verified_full_session_suspension": [False, True],
            "limits_ok": [True, True],
            "up_limit": [11.0, 11.0],
            "down_limit": [9.0, 9.0],
        }
    )
    actions = pd.DataFrame(columns=["symbol", "ex_date"])
    bars = prepare_market(f, actions, 2)
    assert bars.iloc[1]["close"] == 10
    assert not bars.iloc[1].can_buy and not bars.iloc[1].can_sell
    with pytest.raises(ValueError, match="UNVERIFIED_MISSING"):
        prepare_market(f.assign(verified_full_session_suspension=False), actions, 2)
    with pytest.raises(ValueError, match="EX_DATE"):
        prepare_market(f, pd.DataFrame({"symbol": ["A"], "ex_date": [pd.Timestamp("20240201")]}), 2)


def test_retry_waits_for_blocked_asset_not_daily_portfolio_churn(monkeypatch):
    import moneymore.research.baseline_v2_continuous_replay as module

    cfg = BacktestConfig(
        initial_cash=10000,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=2,
        lot_size=100,
        max_position_weight=0.6,
        max_gross_exposure=1,
        max_drawdown=0.15,
    )
    targets = pd.DataFrame(
        {"date": [pd.Timestamp("20240131")] * 2, "symbol": ["A", "B"], "target_weight": [0.5, 0.5]}
    )
    strategy = MonthlyOpeningPlans(
        targets,
        cfg,
        {"turnover_l1": 2, "solver_time_limit": 10, "solver_threads": None},
        "20240301",
    )
    called = []

    def planner(*args, **kwargs):
        called.append(1)
        return {"status": "PASS_REFERENCE_QUOTES_ONLY", "orders": []}

    monkeypatch.setattr(module, "plan_integer_execution", planner)
    strategy.closing("20240131", {})
    snapshot = {
        "equity": 10000,
        "cash": 8000,
        "positions": [{"symbol": "A", "quantity": 100}, {"symbol": "B", "quantity": 100}],
    }
    quotes = pd.DataFrame(
        {"open": [10.0, 10.0], "can_buy": [True, False], "can_sell": [True, False]},
        index=["A", "B"],
    )
    strategy.opening("20240201", snapshot, quotes)
    assert len(called) == 1
    assert strategy.opening("20240202", snapshot, quotes)["status"].startswith("WAITING")
    assert len(called) == 1
    quotes.loc["B", ["can_buy", "can_sell"]] = True
    strategy.opening("20240205", snapshot, quotes)
    assert len(called) == 2


def test_monthly_opening_uses_next_open_no_daily_rebalance_and_daily_ledger_matches(tmp_path):
    config = BacktestConfig(
        initial_cash=100000,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=2,
        lot_size=100,
        max_position_weight=0.6,
        max_gross_exposure=1,
        max_drawdown=0.15,
    )
    # Other tests can already have initialized HiGHS's process-global thread pool.
    # The production replay CLI is a fresh process and explicitly uses two threads.
    protocol = {"turnover_l1": 2.0, "solver_time_limit": 10, "solver_threads": None}
    targets = pd.DataFrame(
        {"date": [pd.Timestamp("20240131")], "symbol": ["A"], "target_weight": [0.5]}
    )
    strategy = MonthlyOpeningPlans(targets, config, protocol, "20240205")
    bars = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(d),
                "symbol": "A",
                "open": o,
                "close": c,
                "can_buy": True,
                "can_sell": True,
            }
            for d, o, c in [
                ("20240131", 10, 10),
                ("20240201", 20, 20),
                ("20240202", 20, 100),
                ("20240205", 100, 100),
            ]
        ]
    )
    actions = pd.DataFrame(
        [
            {
                "symbol": "A",
                "action_key": "A:DIV",
                "record_date": pd.Timestamp("20240201"),
                "ex_date": pd.Timestamp("20240202"),
                "cash_per_share": 1.0,
                "stock_ratio": 0.0,
                "cash_pay_date": pd.Timestamp("20240205"),
                "stock_list_date": pd.NaT,
            }
        ]
    )
    db = tmp_path / "isolated.sqlite3"
    result = run_execution_replay(
        bars=bars,
        targets=EMPTY,
        config=config,
        database=db,
        strategy_id="test",
        account_id="test",
        closing_target_provider=strategy.closing,
        opening_plan_provider=strategy.opening,
        corporate_actions=actions,
        accrue_action_receivables=True,
    )
    assert len(result.fills) == 1
    assert result.fills[0]["trade_date"] == "20240201"
    assert result.fills[0]["price"] == pytest.approx(20 * 1.0002)
    assert result.equity.receivable_cash.iloc[2] > 0
    assert result.equity.receivable_cash.iloc[3] == 0
    proof, ledger, _ = audit_ledger(db, result.equity, bars, actions, config.initial_cash)
    assert proof.equity_error.abs().max() < 0.01
    assert len(ledger) == 1
    assert result.reconciliation["matched"]
