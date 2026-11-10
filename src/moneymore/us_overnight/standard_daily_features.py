"""Standard daily technical, volume-price and mathematical trend indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def standard_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return conventional indicators using data through each input row."""
    data = frame.sort_values("timestamp")
    open_ = data["open"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)
    result = pd.DataFrame(index=data.index)
    result["timestamp"] = data["timestamp"]
    result["symbol"] = data["symbol"]

    # MACD/PPO/TRIX trend and derivatives.
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    result["macd_norm"] = macd / close
    result["macd_signal_norm"] = signal / close
    result["macd_hist_norm"] = hist / close
    result["macd_hist_d1"] = hist.diff() / close
    result["macd_hist_d2"] = hist.diff().diff() / close
    result["ppo"] = 100 * (ema12 / ema26 - 1)
    ema1 = close.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    result["trix15"] = ema3.pct_change(fill_method=None) * 100

    # RSI, stochastic K/D/J, Williams %R and CCI.
    for window in (6, 14, 28):
        result[f"rsi_{window}"] = _rsi(close, window) / 100
    low9, high9 = low.rolling(9).min(), high.rolling(9).max()
    rsv = 100 * (close - low9) / (high9 - low9).replace(0, np.nan)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    result["kdj_k"] = k / 100
    result["kdj_d"] = d / 100
    result["kdj_j"] = (3 * k - 2 * d) / 100
    result["kdj_cross"] = (k - d) / 100
    for window in (14, 28):
        highest, lowest = high.rolling(window).max(), low.rolling(window).min()
        result[f"willr_{window}"] = -100 * (highest - close) / (highest - lowest).replace(0, np.nan)
    typical = (high + low + close) / 3
    for window in (14, 20):
        mean = typical.rolling(window).mean()
        deviation = (typical - mean).abs().rolling(window).mean()
        result[f"cci_{window}"] = (typical - mean) / (0.015 * deviation.replace(0, np.nan))

    # Bollinger, Donchian and Keltner channels.
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    atr20 = true_range.rolling(20).mean()
    for window in (10, 20, 60):
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()
        result[f"boll_pos_{window}"] = (close - middle) / (2 * std).replace(0, np.nan)
        result[f"boll_width_{window}"] = 4 * std / middle
        result[f"boll_width_d1_{window}"] = result[f"boll_width_{window}"].diff()
    result["donchian_pos_20"] = (close - low.rolling(20).min()) / (
        high.rolling(20).max() - low.rolling(20).min()
    ).replace(0, np.nan)
    result["keltner_pos_20"] = (close - close.ewm(span=20, adjust=False).mean()) / (
        2 * atr20
    ).replace(0, np.nan)

    # Directional movement / ADX and Aroon.
    plus_dm = high.diff().where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    atr14 = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    result["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean() / 100
    result["di_spread14"] = (plus_di - minus_di) / 100
    result["adx_d1"] = result["adx14"].diff()
    for window in (14, 25):
        result[f"aroon_up_{window}"] = high.rolling(window).apply(
            lambda values, size=window: (np.argmax(values) + 1) / size, raw=True
        )
        result[f"aroon_down_{window}"] = low.rolling(window).apply(
            lambda values, size=window: (np.argmin(values) + 1) / size, raw=True
        )

    # Volume-price and money-flow indicators.
    signed = np.sign(close.diff()).fillna(0)
    obv = (signed * volume).cumsum()
    result["obv_slope_5"] = obv.diff(5) / volume.rolling(20).mean()
    result["obv_slope_20"] = obv.diff(20) / volume.rolling(20).mean()
    money_flow = typical * volume
    positive = money_flow.where(typical.diff() > 0, 0.0)
    negative = money_flow.where(typical.diff() < 0, 0.0)
    for window in (14, 20):
        ratio = positive.rolling(window).sum() / negative.rolling(window).sum().replace(0, np.nan)
        result[f"mfi_{window}"] = (100 - 100 / (1 + ratio)) / 100
    multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    result["cmf_20"] = (multiplier * volume).rolling(20).sum() / volume.rolling(20).sum()
    result["price_volume_corr_20"] = close.pct_change(fill_method=None).rolling(20).corr(
        volume.pct_change(fill_method=None)
    )
    result["up_down_volume_ratio_20"] = volume.where(close.diff() > 0, 0).rolling(20).sum() / (
        volume.where(close.diff() < 0, 0).rolling(20).sum().replace(0, np.nan)
    )
    result["amihud_20"] = (
        close.pct_change(fill_method=None).abs() / (close * volume).replace(0, np.nan)
    ).rolling(20).mean()

    # Multi-scale returns and discrete derivatives of log price.
    log_close = np.log(close)
    for window in (1, 2, 3, 5, 10, 20, 60):
        result[f"logret_{window}"] = log_close.diff(window)
    for smooth in (3, 5, 10, 20):
        trend = log_close.rolling(smooth).mean()
        result[f"trend_d1_{smooth}"] = trend.diff()
        result[f"trend_d2_{smooth}"] = trend.diff().diff()
    result["candle_body"] = (close - open_) / previous
    result["upper_shadow"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / previous
    result["lower_shadow"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / previous
    return result

