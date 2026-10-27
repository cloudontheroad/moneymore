from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.execution_replay import run_execution_replay


def _config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=100_000,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=2,
        lot_size=100,
        max_position_weight=0.5,
        max_gross_exposure=0.8,
        max_drawdown=0.15,
    )


@pytest.mark.parametrize("corrupt_style_proof", [False, True])
def test_opening_integer_plan_reprices_and_reconciles(tmp_path, corrupt_style_proof):
    from moneymore.research.integer_execution import plan_integer_execution

    context = {}
    observed = []

    def close(date, snapshot):
        context["signal_date"] = date
        return pd.DataFrame([dict(date=date, symbol="AAA", target_weight=0.4)])

    def opening(date, snapshot, quotes):
        assert list(quotes.columns) == ["open", "can_buy", "can_sell"]
        observed.append(snapshot)
        if not context:
            return {"status": "NO_PRIOR_SIGNAL", "orders": []}
        quantities = {p["symbol"]: p["quantity"] for p in snapshot["positions"]}
        plan = plan_integer_execution(
            pd.Series({"AAA": 0.4}),
            quotes.open,
            quantities,
            quantities,
            snapshot["cash"],
            snapshot["equity"],
            _config(),
            pd.DataFrame({"style": [0.0, 1.0]}, index=["AAA", "BBB"]),
            pd.Series({"style": 2.0}),
            turnover=1.0,
        )
        assert plan["status"] == "PASS_REFERENCE_QUOTES_ONLY"
        return {
            **plan,
            "signal_date": context["signal_date"],
            "execution_style_proof": {
                "standardized": {"AAA": {"style": -1.0}},
                "limits": {"style": 0.1 if corrupt_style_proof else 2.0},
            },
        }

    bars = pd.DataFrame(
        [
            dict(date=d, symbol="AAA", open=o, close=c)
            for d, o, c in [
                ("20260105", 10.0, 100.0),
                ("20260106", 20.0, 21.0),
                ("20260107", 22.0, 23.0),
            ]
        ]
    )
    expected = (
        pytest.raises(ValueError, match="actual opening style boundary failed")
        if corrupt_style_proof
        else nullcontext()
    )
    with expected:
        result = run_execution_replay(
            bars=bars,
            targets=pd.DataFrame(columns=["date", "symbol", "target_weight"]),
            config=_config(),
            database=tmp_path / "integer.sqlite3",
            strategy_id="integer",
            account_id="integer",
            closing_target_provider=close,
            opening_plan_provider=opening,
        )
    if corrupt_style_proof:
        return
    assert observed[1]["equity"] == 100_000
    assert result.fills[0]["trade_date"] == "20260106"
    assert result.fills[0]["quantity"] == 2000
    assert result.reconciliation["matched"]
    assert all(
        a["execution_check"]["status"] == "MATCHED"
        for a in result.summary["opening_plan_audits"][1:]
    )
    assert all(
        a["execution_check"]["actual_style_exposures"]["style"] == pytest.approx(-1.0)
        for a in result.summary["opening_plan_audits"][1:]
    )


def test_blocked_opening_plan_never_submits_closing_targets(tmp_path):
    bars = pd.DataFrame(
        [dict(date=d, symbol="AAA", open=10.0, close=10.0) for d in ("20260105", "20260106")]
    )
    result = run_execution_replay(
        bars=bars,
        targets=pd.DataFrame(columns=["date", "symbol", "target_weight"]),
        config=_config(),
        database=tmp_path / "blocked.sqlite3",
        strategy_id="blocked",
        account_id="blocked",
        closing_target_provider=lambda date, snapshot: pd.DataFrame(
            [dict(date=date, symbol="AAA", target_weight=0.5)]
        ),
        opening_plan_provider=lambda *args: {"status": "INFEASIBLE", "orders": []},
    )
    assert not result.orders
    assert not result.fills
    assert result.equity.iloc[-1]["cash"] == 100_000


def test_dynamic_targets_use_replay_holdings_and_next_open(tmp_path: Path):
    seen = []

    def provider(date, snapshot):
        seen.append((date, sum(p["quantity"] for p in snapshot["positions"])))
        return pd.DataFrame([dict(date=pd.Timestamp(date), symbol="AAA", target_weight=0.5)])

    bars = pd.DataFrame(
        [
            dict(date=d, symbol="AAA", open=10.0, close=10.0)
            for d in ("2026-01-05", "2026-01-06", "2026-01-07")
        ]
    )
    result = run_execution_replay(
        bars=bars,
        targets=pd.DataFrame(columns=["date", "symbol", "target_weight"]),
        config=_config(),
        database=tmp_path / "dynamic.sqlite3",
        strategy_id="dynamic",
        account_id="isolated",
        closing_target_provider=provider,
    )
    assert seen[0][1] == 0
    assert seen[1][1] > 0
    assert result.fills[0]["trade_date"] == "20260106"
    assert result.reconciliation["matched"]


def test_replay_uses_next_open_costs_t_plus_one_and_reconciles(tmp_path: Path) -> None:
    bars = pd.DataFrame(
        [
            {"date": "2026-01-05", "symbol": "AAA", "open": 10, "close": 10},
            {"date": "2026-01-06", "symbol": "AAA", "open": 10, "close": 11},
            {"date": "2026-01-07", "symbol": "AAA", "open": 11, "close": 11},
            {"date": "2026-01-08", "symbol": "AAA", "open": 12, "close": 12},
        ]
    )
    targets = pd.DataFrame(
        [
            {"date": "2026-01-05", "symbol": "AAA", "target_weight": 0.5},
            {"date": "2026-01-06", "symbol": "AAA", "target_weight": 0.0},
        ]
    )
    result = run_execution_replay(
        bars=bars,
        targets=targets,
        config=_config(),
        database=tmp_path / "replay.sqlite3",
        strategy_id="replay",
        account_id="replay",
    )
    assert result.fills[0]["trade_date"] == "20260106"
    assert result.fills[0]["price"] == pytest.approx(10.002)
    assert result.fills[1]["trade_date"] == "20260107"
    assert result.fills[1]["price"] == pytest.approx(10.9978)
    assert result.summary["transaction_cost"] > 10
    assert result.summary["reconciled"] is True


def test_replay_defers_limit_up_without_losing_order(tmp_path: Path) -> None:
    bars = pd.DataFrame(
        [
            {"date": "2026-01-05", "symbol": "AAA", "open": 10, "close": 10},
            {
                "date": "2026-01-06",
                "symbol": "AAA",
                "open": 11,
                "close": 11,
                "can_buy": False,
            },
            {"date": "2026-01-07", "symbol": "AAA", "open": 10, "close": 10},
        ]
    )
    targets = pd.DataFrame([{"date": "2026-01-05", "symbol": "AAA", "target_weight": 0.5}])
    result = run_execution_replay(
        bars=bars,
        targets=targets,
        config=_config(),
        database=tmp_path / "limit.sqlite3",
        strategy_id="replay",
        account_id="replay",
    )
    assert result.summary["deferred_count"] == 1
    assert result.fills[0]["trade_date"] == "20260107"
