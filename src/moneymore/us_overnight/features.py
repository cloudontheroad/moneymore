from __future__ import annotations

import numpy as np
import pandas as pd

from .github_features import MATURE_GITHUB_FEATURE_COLUMNS, mature_github_features

LEGACY_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_20d",
    "falling_deceleration",
    "previous_overnight",
    "intraday_return",
    "close_position",
    "range_pct",
    "atr_14",
    "realized_vol_20",
    "relative_volume_20",
    "log_dollar_volume",
    "distance_ma_20",
    "distance_ma_50",
    "distance_ma_200",
    "drawdown_20",
    "drawdown_60",
]

# A compact, non-redundant training set drawn from the complete 162-feature
# mature-project library.  The full library remains available for screening.
MATURE_MODEL_FEATURE_COLUMNS = [
    "KMID", "KLEN", "KUP2", "KLOW2", "KSFT2", "VWAP0",
    "ROC5", "ROC10", "ROC20", "ROC60", "MA5", "MA20", "MA60",
    "STD5", "STD20", "BETA20", "RSQR20", "RESI5", "RESI20",
    "QTLU20", "QTLD20", "RANK20", "RSV5", "RSV20", "IMXD20",
    "CORR5", "CORR20", "CORD5", "CORD20", "CNTD5", "CNTD20",
    "SUMD5", "SUMD20", "VMA5", "VMA20", "WVMA5", "WVMA20",
    "VSUMD5", "VSUMD20", "FINRL_MACD", "FINRL_RSI30", "FINRL_CCI30",
    "FINRL_ADX30",
]
FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS + MATURE_MODEL_FEATURE_COLUMNS
ALL_FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS + MATURE_GITHUB_FEATURE_COLUMNS


def daily_stock_features(frame: pd.DataFrame, cost_bps: float = 20.0) -> pd.DataFrame:
    """Point-in-time daily features and forward labels; predictors stop at row t."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"daily bars missing columns: {sorted(missing)}")
    result = frame.sort_values("timestamp").copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    open_ = pd.to_numeric(result["open"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce")
    cash_dividend = pd.to_numeric(
        result.get("cash_dividend", pd.Series(0.0, index=result.index)), errors="coerce"
    ).fillna(0.0)
    previous_close = close.shift(1)
    for days in (1, 3, 5, 20):
        result[f"ret_{days}d"] = close.pct_change(days, fill_method=None)
    result["falling_deceleration"] = (result["ret_1d"] - result["ret_3d"] / 3).clip(lower=0) * (
        -result["ret_1d"]
    ).clip(lower=0)
    result["previous_overnight"] = open_.div(previous_close) - 1
    result["intraday_return"] = close.div(open_) - 1
    bar_range = (high - low).replace(0, np.nan)
    result["close_position"] = (close - low).div(bar_range)
    result["range_pct"] = bar_range.div(previous_close)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    result["atr_14"] = true_range.rolling(14).mean().div(close)
    result["realized_vol_20"] = result["ret_1d"].rolling(20).std() * np.sqrt(252)
    result["relative_volume_20"] = volume.div(volume.shift(1).rolling(20).mean())
    result["log_dollar_volume"] = np.log1p(close * volume)
    for days in (20, 50, 200):
        result[f"distance_ma_{days}"] = close.div(close.rolling(days).mean()) - 1
    for days in (20, 60):
        result[f"drawdown_{days}"] = close.div(close.rolling(days).max()) - 1
    result["label_next_open_net"] = (
        (open_.shift(-1) + cash_dividend.shift(-1)).div(close)
        - 1
        - cost_bps / 10_000
    )
    # If this row is trade session t, this is the return from buying at open(t)
    # and selling at open(t+1).  A panel intended for a pre-open decision must
    # shift every predictor by one session while leaving this label unshifted.
    result["label_open_to_next_open_net"] = (
        (open_.shift(-1) + cash_dividend.shift(-1)).div(open_)
        - 1
        - cost_bps / 10_000
    )
    result["label_end"] = result["timestamp"].shift(-1)
    mature = mature_github_features(result)
    result = pd.concat([result, mature], axis=1)
    return result
