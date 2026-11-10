"""Point-in-time features independently implemented from mature GitHub designs.

The 158-factor layout follows Microsoft Qlib Alpha158.  MACD/RSI/CCI/ADX are
the default technical state used by FinRL.  No Qlib or FinRL runtime code is
called here, so the live OpenD pipeline remains small and auditable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

WINDOWS = (5, 10, 20, 30, 60)


def _last_rank(values: np.ndarray) -> float:
    return float(pd.Series(values).rank(pct=True).iloc[-1])


def _last_extreme(values: np.ndarray, maximum: bool) -> float:
    index = int(np.nanargmax(values) if maximum else np.nanargmin(values))
    return float(len(values) - 1 - index)


def _regression(close: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    slope = close.rolling(window).apply(
        lambda y: float(np.dot(x - x_mean, y - np.mean(y)) / x_var), raw=True
    )
    variance = close.rolling(window).var(ddof=0)
    covariance = slope * x_var / window
    rsquare = covariance.pow(2).div((x_var / window) * variance.replace(0, np.nan))
    mean = close.rolling(window).mean()
    residual = close - (mean + slope * (window - 1 - x_mean))
    return slope, rsquare.clip(0, 1), residual


def mature_github_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Qlib-158 plus FinRL's four default indicators, using data through t."""
    data = frame.sort_values("timestamp")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    if "price" in data:
        vwap = pd.to_numeric(data["price"], errors="coerce")
    elif "turnover" in data:
        vwap = pd.to_numeric(data["turnover"], errors="coerce").div(volume.replace(0, np.nan))
    else:
        vwap = (open_ + high + low + close) / 4
    eps = 1e-12
    bar_range = high - low
    result = pd.DataFrame(index=data.index)
    result["KMID"] = (close - open_) / open_
    result["KLEN"] = bar_range / open_
    result["KMID2"] = (close - open_) / (bar_range + eps)
    result["KUP"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / open_
    result["KUP2"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / (bar_range + eps)
    result["KLOW"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / open_
    result["KLOW2"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / (bar_range + eps)
    result["KSFT"] = (2 * close - high - low) / open_
    result["KSFT2"] = (2 * close - high - low) / (bar_range + eps)
    result["OPEN0"], result["HIGH0"], result["LOW0"] = open_ / close, high / close, low / close
    result["VWAP0"] = vwap / close
    delta, volume_delta = close.diff(), volume.diff()
    price_ratio = close / close.shift(1)
    volume_ratio_log = np.log(volume.div(volume.shift(1)).replace([np.inf, -np.inf], np.nan) + 1)
    for window in WINDOWS:
        suffix = str(window)
        roll_close, roll_volume = close.rolling(window), volume.rolling(window)
        roll_high, roll_low = high.rolling(window), low.rolling(window)
        slope, rsquare, residual = _regression(close, window)
        result[f"ROC{suffix}"] = close.shift(window) / close
        result[f"MA{suffix}"] = roll_close.mean() / close
        result[f"STD{suffix}"] = roll_close.std(ddof=0) / close
        result[f"BETA{suffix}"] = slope / close
        result[f"RSQR{suffix}"] = rsquare
        result[f"RESI{suffix}"] = residual / close
        result[f"MAX{suffix}"] = roll_high.max() / close
        result[f"MIN{suffix}"] = roll_low.min() / close
        result[f"QTLU{suffix}"] = roll_close.quantile(0.8) / close
        result[f"QTLD{suffix}"] = roll_close.quantile(0.2) / close
        result[f"RANK{suffix}"] = roll_close.apply(_last_rank, raw=True)
        rolling_min, rolling_max = roll_low.min(), roll_high.max()
        result[f"RSV{suffix}"] = (close - rolling_min) / (rolling_max - rolling_min + eps)
        imax = roll_high.apply(lambda x: _last_extreme(x, True), raw=True) / window
        imin = roll_low.apply(lambda x: _last_extreme(x, False), raw=True) / window
        result[f"IMAX{suffix}"], result[f"IMIN{suffix}"] = imax, imin
        result[f"IMXD{suffix}"] = imax - imin
        result[f"CORR{suffix}"] = close.rolling(window).corr(np.log1p(volume))
        result[f"CORD{suffix}"] = price_ratio.rolling(window).corr(volume_ratio_log)
        positive, negative = delta.gt(0).astype(float), delta.lt(0).astype(float)
        result[f"CNTP{suffix}"] = positive.rolling(window).mean()
        result[f"CNTN{suffix}"] = negative.rolling(window).mean()
        result[f"CNTD{suffix}"] = result[f"CNTP{suffix}"] - result[f"CNTN{suffix}"]
        denominator = delta.abs().rolling(window).sum() + eps
        result[f"SUMP{suffix}"] = delta.clip(lower=0).rolling(window).sum() / denominator
        result[f"SUMN{suffix}"] = (-delta.clip(upper=0)).rolling(window).sum() / denominator
        result[f"SUMD{suffix}"] = result[f"SUMP{suffix}"] - result[f"SUMN{suffix}"]
        result[f"VMA{suffix}"] = roll_volume.mean() / (volume + eps)
        result[f"VSTD{suffix}"] = roll_volume.std(ddof=0) / (volume + eps)
        weighted = price_ratio.sub(1).abs() * volume
        result[f"WVMA{suffix}"] = weighted.rolling(window).std(ddof=0) / (
            weighted.rolling(window).mean() + eps
        )
        volume_denominator = volume_delta.abs().rolling(window).sum() + eps
        result[f"VSUMP{suffix}"] = volume_delta.clip(lower=0).rolling(window).sum() / volume_denominator
        result[f"VSUMN{suffix}"] = (-volume_delta.clip(upper=0)).rolling(window).sum() / volume_denominator
        result[f"VSUMD{suffix}"] = result[f"VSUMP{suffix}"] - result[f"VSUMN{suffix}"]
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    result["FINRL_MACD"] = macd - macd.ewm(span=9, adjust=False).mean()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(30).mean() / loss.rolling(30).mean().replace(0, np.nan)
    result["FINRL_RSI30"] = 100 - 100 / (1 + rs)
    typical = (high + low + close) / 3
    mean_typical = typical.rolling(30).mean()
    mad = typical.rolling(30).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    result["FINRL_CCI30"] = (typical - mean_typical) / (0.015 * mad.replace(0, np.nan))
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(30).mean()
    plus_di = 100 * plus_dm.rolling(30).mean() / atr
    minus_di = 100 * minus_dm.rolling(30).mean() / atr
    result["FINRL_ADX30"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).rolling(30).mean()
    return result.replace([np.inf, -np.inf], np.nan)


MATURE_GITHUB_FEATURE_COLUMNS = [
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    "OPEN0", "HIGH0", "LOW0", "VWAP0",
    *[f"{group}{window}" for group in (
        "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
        "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD",
        "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
    ) for window in WINDOWS],
    "FINRL_MACD", "FINRL_RSI30", "FINRL_CCI30", "FINRL_ADX30",
]
