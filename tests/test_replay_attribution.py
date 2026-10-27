import numpy as np
import pandas as pd

from moneymore.research.replay_attribution import (
    economic_daily_return,
    exposure_snapshot,
    reconstruct_cash,
    reconstruct_positions,
)


def test_holdings_reconstruct_fills_and_bonus_not_final_position_backfill():
    fills = pd.DataFrame(
        [
            {"trade_date": "1", "symbol": "A", "side": "BUY", "quantity": 100},
            {"trade_date": "3", "symbol": "A", "side": "SELL", "quantity": 50},
        ]
    )
    ledger = pd.DataFrame(
        [{"trade_date": "2", "symbol": "A", "action_type": "STOCK_DIVIDEND", "share_quantity": 20}]
    )
    assert reconstruct_positions(fills, ledger, ["0", "1", "2", "3"]) == {
        "0": {},
        "1": {"A": 100},
        "2": {"A": 120},
        "3": {"A": 70},
    }


def test_official_actions_remove_spurious_ex_date_loss():
    result = economic_daily_return(
        pd.Series({"A": 10.0, "B": 10.0}),
        pd.Series({"A": 9.0, "B": 5.0}),
        pd.Series({"B": 1.0}),
        pd.Series({"A": 1.0}),
    )
    np.testing.assert_allclose(result, 0)


def test_cash_reconstruction_keeps_dividends_and_fees_separate():
    fills = pd.DataFrame(
        [
            {"trade_date": "1", "side": "BUY", "quantity": 100, "price": 10.0, "fee": 5.0},
            {"trade_date": "3", "side": "SELL", "quantity": 100, "price": 11.0, "fee": 6.0},
        ]
    )
    ledger = pd.DataFrame(
        [{"trade_date": "2", "action_type": "CASH_DIVIDEND", "cash_amount": 20.0}]
    )
    assert reconstruct_cash(fills, ledger, ["0", "1", "2", "3"], 2000.0) == {
        "0": 2000.0,
        "1": 995.0,
        "2": 1015.0,
        "3": 2109.0,
    }


def test_stock_sleeve_style_not_confused_with_cash_exposure():
    features = pd.DataFrame({"style": [0.0, 1.0, 2.0]}, index=list("ABC"))
    a = exposure_snapshot(
        {"C": 10}, pd.Series(10.0, index=list("ABC")), 100.0, features, list("ABC")
    )
    b = exposure_snapshot(
        {"C": 10}, pd.Series(10.0, index=list("ABC")), 200.0, features, list("ABC")
    )
    assert a["active_stock_sleeve_exposures"] == b["active_stock_sleeve_exposures"]
    assert a["actual_gross"] == 2 * b["actual_gross"]
    assert (
        exposure_snapshot(
            {"D": 10}, pd.Series(10.0, index=list("ABC")), 100.0, features, list("ABC")
        )["status"]
        == "MISSING_HELD_STYLE"
    )


def test_round_lot_can_break_neutral_continuous_target():
    features = pd.DataFrame({"style": [-1.0, 1.0]}, index=["A", "B"])
    close = pd.Series({"A": 1000.0, "B": 10.0})
    target = exposure_snapshot({"A": 5, "B": 500}, close, 10000.0, features, ["A", "B"])
    actual = exposure_snapshot({"B": 500}, close, 10000.0, features, ["A", "B"])
    assert abs(target["active_stock_sleeve_exposures"]["style"]) < 1e-12
    assert actual["active_stock_sleeve_exposures"]["style"] == 1.0
