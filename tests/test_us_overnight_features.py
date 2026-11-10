import pandas as pd
import pytest

from moneymore.us_overnight.features import ALL_FEATURE_COLUMNS, daily_stock_features


def test_daily_features_label_next_open_without_lookahead_in_predictors():
    dates = pd.date_range("2025-01-01", periods=220, tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": dates,
            "open": range(100, 320),
            "high": range(102, 322),
            "low": range(99, 319),
            "close": range(101, 321),
            "volume": [1_000_000] * 220,
        }
    )
    result = daily_stock_features(bars, cost_bps=20)
    assert result.loc[218, "label_next_open_net"] == pytest.approx(319 / 319 - 1 - 0.002)
    assert result.loc[218, "label_open_to_next_open_net"] == pytest.approx(
        319 / 318 - 1 - 0.002
    )
    changed = bars.copy()
    changed.loc[219, ["high", "low", "close", "volume"]] = [999, 1, 500, 99]
    rerun = daily_stock_features(changed, cost_bps=20)
    columns = ["ret_20d", "atr_14", "relative_volume_20", "distance_ma_200"]
    pd.testing.assert_series_equal(result.loc[218, columns], rerun.loc[218, columns])


def test_mature_github_feature_library_is_complete():
    assert len(ALL_FEATURE_COLUMNS) == 180
    assert {"KMID", "RSQR60", "WVMA20", "FINRL_RSI30", "FINRL_ADX30"}.issubset(
        ALL_FEATURE_COLUMNS
    )
