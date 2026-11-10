from pathlib import Path

import pandas as pd

from scripts.build_us_overnight_investable_panel import eligibility_from_bars


def test_eligibility_uses_only_bars_before_signal(tmp_path: Path) -> None:
    dates = pd.date_range("2025-01-02", periods=122, freq="B", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": dates,
        "open": [10.0] * len(dates),
        "high": [10.0] * len(dates),
        "low": [10.0] * len(dates),
        "close": [10.0] * len(dates),
        "volume": [3_000_000.0] * len(dates),
        "cash_dividend": [0.0] * len(dates),
        "symbol": ["US.TEST"] * len(dates),
    })
    path = tmp_path / "US_TEST.parquet"
    frame.to_parquet(path, index=False)
    eligibility, _ = eligibility_from_bars(path)
    assert not bool(eligibility.iloc[119].investable)
    assert bool(eligibility.iloc[120].investable)
    assert eligibility.iloc[120].prior_close == 10.0
    assert eligibility.iloc[120].median_dollar_volume_20d == 30_000_000.0
    assert eligibility.iloc[120].correct_open_to_next_open_label == -0.002


def test_new_listing_large_next_open_gap_is_not_investable(tmp_path: Path) -> None:
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-15", periods=3, freq="B", tz="UTC"),
        "open": [200.0, 297.5, 285.0],
        "high": [200.0, 297.5, 295.0],
        "low": [200.0, 270.0, 270.0],
        "close": [200.0, 270.0, 270.0],
        "volume": [400.0, 7_300.0, 11_700.0],
        "cash_dividend": [0.0, 0.0, 0.0],
        "symbol": ["US.NEW"] * 3,
    })
    path = tmp_path / "US_NEW.parquet"
    frame.to_parquet(path, index=False)
    eligibility, gaps = eligibility_from_bars(path)
    assert not eligibility.investable.any()
    assert len(gaps) == 1
    assert gaps.iloc[0].open_to_next_open > 0.48
    assert abs(eligibility.iloc[0].correct_open_to_next_open_label - 0.4855) < 1e-12
