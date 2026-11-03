import numpy as np
import pandas as pd
from moneymore.research.style_price_features import price_features, named_price_risk_exposures, registered_liquidity_on_calendar


def test_adjustment_removes_split_not_real_price_move_and_features_causal():
    price = np.full(100, 100.)
    price[70:] = 50.
    data = pd.DataFrame({"open": price, "high": price, "low": price, "close": price,
        "vol": 100., "amount": 1000., "adj_factor": np.r_[np.ones(70), np.full(30, 2.)]})
    assert price_features(data, "raw").close_return.iloc[70] == -.5
    assert price_features(data, "adjusted").close_return.iloc[70] == 0
    adjusted = data.copy()
    adjusted.loc[90:, "close"] = 1e6
    pd.testing.assert_frame_equal(price_features(data).iloc[:90], price_features(adjusted).iloc[:90])


def test_risk_matrix_ignores_extra_caps_and_does_not_fill_missing():
    rng = np.random.default_rng(17)
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .01, (130, 40)), axis=0)), columns=[f"S{i}" for i in range(40)])
    prices.iloc[-10, 0] = np.nan
    cap = pd.Series(100., index=[*prices.columns, "EXTRA"])
    result, meta = named_price_risk_exposures(prices, cap)
    assert len(result) == 40
    assert result.loc["S0"].isna().all()
    assert result.loc["S1"].notna().all()
    assert meta["complete_risk_assets"] == 39


def test_registered_liquidity_preserves_calendar_gaps_and_converts_percent():
    dates = pd.bdate_range("2026-01-01", periods=21).strftime("%Y%m%d").tolist()
    basic = pd.DataFrame({"date": dates, "symbol": "S", "turnover_rate": 2.})
    full, _ = registered_liquidity_on_calendar(basic, dates, ["S"])
    assert full.turnover_mean_20.iloc[:19].isna().all()
    assert np.isclose(full.turnover_mean_20.iloc[-1], .02)
    broken, aligned = registered_liquidity_on_calendar(basic.drop(index=5), dates, ["S"])
    assert len(aligned) == 21
    assert broken.turnover_mean_20.isna().all()  # no compressed 20-row history


def test_registered_liquidity_is_causal_and_grouped():
    dates = pd.bdate_range("2026-01-01", periods=25).strftime("%Y%m%d").tolist()
    basic = pd.concat([pd.DataFrame({"date": dates, "symbol": s, "turnover_rate": rate})
        for s, rate in [("A", 1.), ("B", 3.)]], ignore_index=True)
    before, _ = registered_liquidity_on_calendar(basic, dates, ["A", "B"])
    basic.loc[basic.date.gt(dates[20]), "turnover_rate"] = 999.
    after, _ = registered_liquidity_on_calendar(basic, dates, ["A", "B"])
    pd.testing.assert_frame_equal(before.loc[before.date.le(pd.Timestamp(dates[20]))],
        after.loc[after.date.le(pd.Timestamp(dates[20]))])
    assert np.isclose(before.loc[before.symbol.eq("B"), "turnover_mean_20"].iloc[-1], .03)
