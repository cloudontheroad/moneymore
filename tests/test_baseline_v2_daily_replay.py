import pandas as pd

from moneymore.config import BacktestConfig
from moneymore.execution_replay import run_execution_replay
from moneymore.research.baseline_v2_continuous_replay import EMPTY
from moneymore.research.baseline_v2_daily_replay import DailyOpeningPlans, diagnostic_performance


def test_diagnostic_performance_uses_initial_cash_and_previous_year_end():
    equity = pd.DataFrame(
        {
            "trade_date": ["20240131", "20241231", "20250102", "20251231"],
            "equity": [90.0, 110.0, 88.0, 121.0],
            "gross_exposure": [0.0, 1.0, 0.8, 0.9],
        }
    )
    result = diagnostic_performance(equity, [{"fee": 1.0, "price": 5.0, "quantity": 2}], 100)
    assert abs(result["cumulative_return"] - 0.21) < 1e-12
    assert abs(result["maximum_drawdown"] - 0.2) < 1e-12
    assert abs(result["period_returns_by_calendar_year"]["2025"] - 0.1) < 1e-12
    assert result["fees"] == 1
    assert result["gross_traded_notional"] == 10


def test_actual_fills_drive_exit_and_input_fault_clears_old_target(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=5)
    keys = dates.strftime("%Y%m%d").tolist()
    symbols_by_day = [["A", "B"], ["A", "B"], ["B", "C"], ["C"], ["B", "C"]]
    rows, pool = [], []
    for date, key, symbols in zip(dates, keys, symbols_by_day, strict=True):
        for rank, symbol in enumerate(symbols):
            rows.append({"date": date, "symbol": symbol, "score": 2.0 - rank})
        for symbol in ["B", "C"] if key >= keys[2] else ["A", "B"]:
            pool.append({"signal_date": key, "security_id": symbol})
    policy = {
        "portfolio": {
            "top_k": 2,
            "exit_rank": 2,
            "max_replacements": 1,
            "gross_exposure": 1.0,
            "minimum_weight": 0.4,
            "maximum_weight": 0.6,
            "correlation_lookback": 40,
            "cluster_correlation_threshold": 0.99,
            "maximum_cluster_members": 2,
            "weighting_method": "rank_linear",
        }
    }
    config = BacktestConfig(
        initial_cash=100000,
        max_position_weight=0.6,
        max_gross_exposure=1.0,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=2,
        lot_size=100,
        max_drawdown=0.15,
    )
    protocol = {"turnover_l1": 2.0, "solver_time_limit": 10, "solver_threads": None}
    prices = pd.DataFrame(10.0, index=dates, columns=["A", "B", "C"])
    daily = DailyOpeningPlans(
        pd.DataFrame(rows),
        pd.DataFrame(pool),
        prices,
        policy,
        config,
        protocol,
        keys,
        keys[-1],
        {"A", "B", "C"},
    )
    bars = pd.DataFrame(
        [
            {"date": d, "symbol": s, "open": 10.0, "close": 10.0, "can_buy": True, "can_sell": True}
            for d in dates
            for s in ["A", "B", "C"]
        ]
    )
    result = run_execution_replay(
        bars=bars,
        targets=EMPTY,
        config=config,
        database=tmp_path / "paper.sqlite3",
        strategy_id="daily-test",
        account_id="daily-test",
        closing_target_provider=daily.closing,
        opening_plan_provider=daily.opening,
    )
    assert daily.events[1]["actual_incumbents"]["A"] > 0
    exit_event = next(e for e in daily.events if e["signal_date"] == keys[2])
    assert exit_event["universe_exits"] == ["A"] and "A" not in exit_event["weights"]
    assert any(
        f["side"] == "SELL" and f["symbol"] == "A" and f["trade_date"] == keys[3]
        for f in result.fills
    )
    fault = daily.events[-1]
    assert fault["status"] == "BLOCKED_HELD_INPUT_HOLD_AND_ALERT" and "B" in fault["input_faults"]
    assert daily.context == {}
    assert not any(f["trade_date"] == keys[-1] for f in result.fills)
