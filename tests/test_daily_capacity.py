import pandas as pd
import pytest

from moneymore.research.daily_capacity import daily_participation


def data():
    fills = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": "20240102", "side": "BUY", "quantity": 100, "price": 10},
            {"symbol": "A", "trade_date": "20240102", "side": "SELL", "quantity": 100, "price": 10},
        ]
    )
    market = pd.DataFrame([{"symbol": "A", "trade_date": "20240102", "vol": 100, "amount": 100}])
    return fills, market


def test_units_and_gross_aggregation_not_netting():
    rows, summary = daily_participation(*data())
    assert rows.iloc[0].share_participation == pytest.approx(0.02)
    assert rows.iloc[0].notional_participation == pytest.approx(0.02)
    assert summary["opening_depth_verified"] is False
    assert summary["fills"] == 2


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 1e308])
def test_bad_liquidity_not_filled_or_treated_as_zero_participation(value):
    fills, market = data()
    market["vol"] = market.vol.astype(float)
    market.loc[0, "vol"] = value
    rows, summary = daily_participation(fills, market)
    assert summary["missing_or_invalid_symbol_days"] == 1
    assert pd.isna(rows.iloc[0].share_participation)


def test_duplicate_liquidity_rejected():
    fills, market = data()
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        daily_participation(fills, pd.concat([market, market]))


def test_missing_day_is_not_forward_filled():
    fills, market = data()
    market.loc[0, "trade_date"] = "20240101"
    _, summary = daily_participation(fills, market)
    assert summary["missing_or_invalid_symbol_days"] == 1


def test_invalid_fill_rejected():
    fills, market = data()
    fills["quantity"] = fills.quantity.astype(float)
    fills.loc[0, "quantity"] = 1.5
    with pytest.raises(ValueError, match="NONINTEGER"):
        daily_participation(fills, market)
