import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("report_ashare_three_year")


def test_historical_industry_does_not_use_future_classification(report):
    panel = pd.DataFrame(
        {"date": ["20230101", "20240101"], "symbol": ["A", "A"], "industry": ["OLD", "NEW"]}
    )
    frame = pd.DataFrame(
        {"date": ["20231201", "20240201"], "symbol": ["A", "A"], "weight": [0.1, 0.2]}
    )
    got = report.historical_industry(frame, panel)
    assert got.industry.tolist() == ["OLD", "NEW"]
    with pytest.raises(ValueError, match="MISSING_HISTORICAL_INDUSTRY"):
        report.historical_industry(frame.assign(date="20221201"), panel)


def test_drawdown_can_start_with_first_purchase_cost(report):
    frame = pd.DataFrame(
        {"date": ["20231101", "20231102", "20231103"], "equity": [99.0, 80.0, 102.0]}
    )
    got = report.drawdown_interval(frame, 100.0, "20231031")
    assert got["peak_date"] == "20231031"
    assert got["trough_date"] == "20231102"
    assert got["net_pnl"] == -20.0
    assert got["recovery_date"] == "20231103"


def test_year_subperiods_include_first_day_and_chain(report):
    frame = pd.DataFrame(
        {
            "date": ["20231228", "20231229", "20240102", "20240103"],
            "equity": [98.0, 110.0, 105.0, 121.0],
        }
    )
    got = report.subperiods(frame, 100.0, "Y")
    assert got["2023"]["net_return"] == pytest.approx(0.1)
    assert got["2024"]["net_return"] == pytest.approx(0.1)
    assert got["2023"]["maximum_drawdown"] == pytest.approx(0.02)
    assert got["2024"]["maximum_drawdown"] == pytest.approx(1 - 105 / 110)


def test_lightgbm_thread_bound_does_not_change_forecasts(report):
    import lightgbm as lgb

    module = importlib.import_module("train_ashare_three_year")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 5))
    y = rng.normal(size=200)
    model = lgb.train(
        {"objective": "regression", "num_threads": 2, "verbose": -1},
        lgb.Dataset(x, label=y),
        num_boost_round=5,
    )
    original = lgb.Booster.predict
    expected = model.predict(x)
    with module.bounded_lightgbm_prediction(2):
        actual = model.predict(x)
    np.testing.assert_array_equal(actual, expected)
    assert lgb.Booster.predict is original


@pytest.mark.parametrize("action", ["CASH_DIVIDEND", "STOCK_DIVIDEND"])
def test_attribution_separates_pending_cash_and_stock_rights(report, tmp_path, action):
    import sqlite3

    days = ["20231101", "20231102", "20231103"]
    cash_action = action == "CASH_DIVIDEND"
    equity = pd.DataFrame(
        {
            "date": days,
            "equity": [100.0, 100.0, 103.0 if cash_action else 110.0],
            "cash": [0.0, 0.0, 1.0 if cash_action else 0.0],
        }
    )
    holdings = pd.DataFrame(
        {
            "date": days,
            "symbol": ["A"] * 3,
            "weight": [1.0, 0.99 if cash_action else 0.5, 102 / 103 if cash_action else 1.0],
        }
    )
    fills = pd.DataFrame(
        {
            "date": [days[0]],
            "symbol": ["A"],
            "side": ["BUY"],
            "notional": [100.0],
            "fee": [0.0],
            "slippage": [0.0],
            "quantity": [1],
            "price": [100.0],
        }
    )
    ledger = pd.DataFrame(
        {
            "trade_date": [days[-1]],
            "symbol": ["A"],
            "action_type": [action],
            "cash_amount": [1.0 if cash_action else 0.0],
            "share_quantity": [0 if cash_action else 1],
        }
    )
    with sqlite3.connect(tmp_path / "paper.sqlite3") as db:
        ledger.to_sql("corporate_action_ledger", db, index=False)
    panel = pd.DataFrame({"date": ["20231031"], "symbol": ["A"], "industry": ["SECTOR"]})
    result = report.attribution(tmp_path, equity, fills, holdings, panel, 100.0, "20231031")
    assert result["maximum_daily_reconciliation_error"] < 1e-8
    assert result["unsettled_rights_total_pnl"] == pytest.approx(0)
    assert result["industry_net_pnl"]["SECTOR"] == pytest.approx(3.0 if cash_action else 10.0)
    assert result["cash_dividends_received_gross"] == (1.0 if cash_action else 0.0)
