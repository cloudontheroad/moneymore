from types import SimpleNamespace

import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.execution.paper import PaperBroker
from moneymore.execution_replay import run_execution_replay
from moneymore.research.baseline_v2_continuous_replay import EMPTY
from moneymore.research.baseline_v2_platform_risk import (
    PlatformRiskDailyPlans,
    PlatformRiskReductionPlans,
)


@pytest.mark.parametrize("reduce", [False, True])
def test_native_risk_escalation_persists_and_never_auto_recovers(tmp_path, reduce):
    dates = pd.bdate_range("2024-01-02", periods=5)
    keys = dates.strftime("%Y%m%d").tolist()
    scores, pool = [], []
    for i, date in enumerate(dates):
        for symbol in ("A", "B"):
            scores.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "score": 2 if symbol == ("A" if i < 2 else "B") else 1,
                }
            )
            pool.append({"signal_date": keys[i], "security_id": symbol})
    policy = {
        "portfolio": {
            "top_k": 1,
            "exit_rank": 1,
            "max_replacements": 1,
            "gross_exposure": 1,
            "minimum_weight": 1,
            "maximum_weight": 1,
            "correlation_lookback": 40,
            "cluster_correlation_threshold": 0.99,
            "maximum_cluster_members": 1,
            "weighting_method": "rank_linear",
        }
    }
    config = BacktestConfig(
        initial_cash=100000,
        max_position_weight=1,
        max_drawdown=0.15,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=0,
        lot_size=100,
        max_gross_exposure=1,
    )
    prices = pd.DataFrame({"A": [10, 10, 8, 11, 12], "B": [10] * 5}, index=dates)
    database = tmp_path / "risk.sqlite3"
    planner_class = PlatformRiskReductionPlans if reduce else PlatformRiskDailyPlans
    planner = planner_class(
        pd.DataFrame(scores),
        pd.DataFrame(pool),
        prices,
        policy,
        config,
        # HiGHS thread pool is process-global; other tests initialize the default.
        # Fresh-process research CLIs explicitly use two threads.
        {"turnover_l1": 2, "solver_time_limit": 10, "solver_threads": None},
        keys,
        keys[-1],
        {"A", "B"},
        risk_database=database,
        account_id="test",
    )
    bars = pd.DataFrame(
        [
            {
                "date": day,
                "symbol": symbol,
                "open": prices.loc[day, symbol],
                "close": prices.loc[day, symbol],
                "can_buy": True,
                "can_sell": True,
            }
            for day in dates
            for symbol in ("A", "B")
        ]
    )
    result = run_execution_replay(
        bars=bars,
        targets=EMPTY,
        config=config,
        database=database,
        strategy_id="test",
        account_id="test",
        closing_target_provider=planner.closing,
        opening_plan_provider=planner.opening,
    )
    assert len(result.fills) == (2 if reduce else 1), result.summary["opening_plan_audits"]
    assert result.fills[0]["trade_date"] == keys[1]
    if reduce:
        assert result.fills[1]["side"] == "SELL"
        assert result.fills[1]["trade_date"] == keys[2]
        assert result.equity.iloc[-1].gross_exposure == 0
    state = PaperBroker(database).risk_state("test")
    assert state["effective_state"] == "SELL_ONLY"
    if not reduce:  # Holding rebounds; native gate demands explicit recovery.
        assert state["recovery_required"]
    assert sum(
        p["plan"]["status"] == "ACCOUNT_RISK_BLOCKED" for p in result.summary["opening_plan_audits"]
    ) == (0 if reduce else 3)


def test_liquidation_respects_halt_and_unlisted_rights_and_retries():
    planner = object.__new__(PlatformRiskReductionPlans)
    planner.config = SimpleNamespace(
        slippage_bps=2,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
    )
    planner.context = {"signal_date": "20240102"}
    snapshot = {
        "cash": 1000,
        "positions": [{"symbol": "A", "quantity": 103}],
        "unsettled_stock_rights": {"A": 10},
    }
    quotes = pd.DataFrame({"open": [10.0], "can_sell": [False]}, index=["A"])
    state = {"effective_state": "SELL_ONLY"}
    blocked = planner.liquidation_plan("20240103", snapshot, quotes, state)
    assert blocked["orders"] == []
    assert blocked["target_quantities"] == {"A": 103}
    quotes.loc["A", "can_sell"] = True
    retry = planner.liquidation_plan("20240104", snapshot, quotes, state)
    assert retry["orders"][0]["quantity"] == 103
    assert retry["locked_rights_not_sold"] == {"A": 10}
    assert retry["target_quantities"] == {"A": 0}


def test_suspended_never_constructs_an_order(monkeypatch):
    planner = object.__new__(PlatformRiskReductionPlans)
    monkeypatch.setattr(
        planner, "evaluate_opening_risk", lambda *_: {"effective_state": "SUSPENDED"}
    )
    assert planner.opening("20240103", {}, pd.DataFrame())["orders"] == []
