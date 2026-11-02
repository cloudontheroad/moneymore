import numpy as np
import pandas as pd
from moneymore.research.style_price_features import price_features, named_price_risk_exposures


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
