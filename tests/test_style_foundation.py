import pandas as pd
import pytest
from moneymore.research.style_foundation import economic_labels


def inputs():
    days = pd.bdate_range("2026-01-01", periods=7).strftime("%Y%m%d").tolist()
    daily = pd.DataFrame({"ts_code": ["A"]*7, "trade_date": days, "close": [10, 11, 12, 13, 14, 15, 16]})
    adj = daily[["ts_code", "trade_date"]].assign(adj_factor=1.)
    return days, daily, adj


def test_forward_return_sign_horizon_and_no_fill():
    days, daily, adj = inputs()
    labels = economic_labels(daily, adj, days)
    assert labels.loc[(pd.Timestamp(days[0]), "A")] == .5
    assert pd.isna(labels.loc[(pd.Timestamp(days[2]), "A")])
    missing = economic_labels(daily.drop(index=5), adj, days)
    assert pd.isna(missing.loc[(pd.Timestamp(days[0]), "A")])


def test_adjustment_invariance_and_duplicate_rejection():
    days, daily, adj = inputs()
    scaled = adj.assign(adj_factor=adj.adj_factor*10)
    pd.testing.assert_series_equal(economic_labels(daily, adj, days), economic_labels(daily, scaled, days))
    with pytest.raises(ValueError):
        economic_labels(pd.concat([daily, daily.iloc[:1]]), adj, days)
