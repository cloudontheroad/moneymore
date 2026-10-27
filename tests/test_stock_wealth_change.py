import pandas as pd
import pytest

from moneymore.research.replay_attribution import stock_wealth_change


def test_buy_and_sale_cash_flows_and_fees():
    fills = pd.DataFrame(
        [{"symbol": "A", "side": "BUY", "quantity": 100, "price": 10.0, "fee": 5.0}]
    )
    assert stock_wealth_change({}, {"A": 1100.0}, fills, pd.DataFrame()) == {"A": 95.0}
    fills.loc[0, "side"] = "SELL"
    assert stock_wealth_change({"A": 900.0}, {}, fills, pd.DataFrame()) == {"A": 95.0}


def test_dividend_receivable_and_payment_not_double_counted():
    empty = pd.DataFrame()
    assert stock_wealth_change({"A": 1000.0}, {"A": 900.0 + 100.0}, empty, empty) == {"A": 0.0}
    ledger = pd.DataFrame([{"symbol": "A", "action_type": "CASH_DIVIDEND", "cash_amount": 100.0}])
    assert stock_wealth_change({"A": 1000.0}, {"A": 900.0}, empty, ledger) == {"A": 0.0}


def test_stock_rights_settlement_does_not_create_profit():
    ledger = pd.DataFrame([{"symbol": "A", "action_type": "STOCK_DIVIDEND", "cash_amount": 0.0}])
    assert stock_wealth_change({"A": 1000.0}, {"A": 1000.0}, pd.DataFrame(), ledger) == {"A": 0.0}


def test_nonfinite_pnl_rejected():
    with pytest.raises(ValueError, match="NONFINITE"):
        stock_wealth_change({}, {"A": float("nan")}, pd.DataFrame(), pd.DataFrame())
