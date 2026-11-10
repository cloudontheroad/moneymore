from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.data import (
    FixtureMarketData,
    FutuOpenDReadOnly,
    normalize_futu_kline,
)
from moneymore.us_overnight.pipeline import (
    generate_daily_candidates,
    overnight_label,
    window_vwap,
    write_candidates,
)

NY = ZoneInfo("America/New_York")


def inputs():
    symbols = ["BBB", "AAA", "CCC"]
    universe = pd.DataFrame(
        {
            "symbol": symbols,
            "session_date": ["2026-11-05"] * 3,
            "known_at": ["2026-11-05T15:40:00-05:00"] * 3,
            "price": [10.0] * 3,
            "median_dollar_volume_20d": [30e6] * 3,
            "listing_days": [200] * 3,
            "security_type": ["STOCK"] * 3,
        }
    )
    features = pd.DataFrame(
        {
            "symbol": symbols,
            "observed_at": ["2026-11-05T15:45:00-05:00"] * 3,
            "ret_1d": [-0.03] * 3,
            "ret_3d": [-0.12] * 3,
            "close_position": [0.8] * 3,
            "last_30m_return": [0.01] * 3,
        }
    )
    return features, universe


def test_no_lookahead_and_stable_tie_order():
    features, universe = inputs()
    future = features.iloc[[0]].assign(symbol="ZZZ", observed_at="2026-11-05T15:51:00-05:00")
    universe = pd.concat([universe, universe.iloc[[0]].assign(symbol="ZZZ")], ignore_index=True)
    result = generate_daily_candidates(
        pd.concat([features, future]),
        universe,
        date(2026, 11, 5),
        datetime(2026, 11, 5, 15, 50, tzinfo=NY),
        USOvernightConfig(),
    )
    assert result.symbol.tolist() == ["AAA", "BBB", "CCC"]
    assert "ZZZ" not in set(result.symbol)


def test_timezone_boundary_and_missing_data():
    features, universe = inputs()
    features.loc[0, "ret_1d"] = None
    with pytest.raises(ValueError, match="timezone-aware"):
        generate_daily_candidates(
            features,
            universe,
            date(2026, 11, 5),
            datetime.fromisoformat("2026-11-05T15:50:00"),
            USOvernightConfig(),
        )
    result = generate_daily_candidates(
        features,
        universe,
        date(2026, 11, 5),
        datetime(2026, 11, 6, 4, 50, tzinfo=ZoneInfo("Asia/Shanghai")),
        USOvernightConfig(),
    )
    assert "BBB" not in set(result.symbol)


def test_vwap_cost_and_output(tmp_path):
    bars = pd.DataFrame(
        {
            "timestamp": ["2026-11-05T20:55:00Z", "2026-11-05T20:56:00Z"],
            "price": [10.0, 12.0],
            "volume": [100, 300],
        }
    )
    value = window_vwap(
        bars, datetime(2026, 11, 5, 15, 55, tzinfo=NY), datetime(2026, 11, 5, 16, 0, tzinfo=NY)
    )
    assert value == 11.5
    assert overnight_label(10, 11, 20) == pytest.approx(0.098)
    config = USOvernightConfig(output_root=tmp_path / "isolated")
    paths = write_candidates(pd.DataFrame({"symbol": ["AAA"]}), date(2026, 11, 5), config)
    assert all(path.exists() for path in paths)


def test_offline_fixture_and_opend_probe():
    fake = FixtureMarketData({"stock_basicinfo": pd.DataFrame({"code": ["US.AAPL"]})})
    assert fake.stock_basicinfo().code.tolist() == ["US.AAPL"]
    probe = FutuOpenDReadOnly(port=1).probe(timeout=0.01)
    assert probe["status"] == "OFFLINE"
    assert probe["read_only"] is True


def test_futu_kline_normalization_uses_turnover_vwap_and_new_york_time():
    raw = pd.DataFrame(
        {
            "time_key": ["2026-11-04 15:55:00"],
            "open": [10.0],
            "close": [10.2],
            "high": [10.3],
            "low": [9.9],
            "volume": [200],
            "turnover": [2020.0],
        }
    )
    result = normalize_futu_kline(raw)
    assert result.loc[0, "price"] == pytest.approx(10.1)
    assert result.loc[0, "price_source"] == "TURNOVER_DIV_VOLUME"
    assert str(result.loc[0, "timestamp"]) == "2026-11-04 20:55:00+00:00"


def test_futu_kline_normalization_falls_back_when_reported_vwap_is_outside_bar():
    raw = pd.DataFrame(
        {
            "time_key": ["2026-11-04 15:55:00"],
            "open": [10.0],
            "close": [10.2],
            "high": [10.3],
            "low": [9.9],
            "volume": [200],
            "turnover": [1800.0],
        }
    )
    result = normalize_futu_kline(raw)
    assert result.loc[0, "price"] == pytest.approx(10.1)
    assert result.loc[0, "price_source"] == "OHLC4_FALLBACK"
