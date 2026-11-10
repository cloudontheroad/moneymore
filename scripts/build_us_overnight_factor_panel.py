"""Add causal overnight, market-regime, sector-proxy and calendar factor groups."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sync_us_yahoo_adjusted_history import download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/us_overnight_paper_only/daily_yahoo_adjusted"
BASE = ROOT / "data/us_overnight_corrected/features/historical_panel_61_lag1.parquet"
OUT = ROOT / "data/us_overnight_corrected/features"

MARKET = ["SPY", "QQQ", "IWM", "DIA", "^VIX", "^VIX3M", "^TNX"]
SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


def stock_factors(path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(path).sort_values("timestamp")
    close = bars["close"].astype(float)
    open_ = bars["open"].astype(float)
    overnight = open_ / close.shift(1) - 1
    intraday = close / open_ - 1
    result = bars[["timestamp", "symbol"]].copy()
    for window in (5, 20, 60):
        result[f"overnight_mean_{window}"] = overnight.rolling(window).mean().shift(1)
        result[f"overnight_std_{window}"] = overnight.rolling(window).std().shift(1)
        result[f"overnight_skew_{window}"] = overnight.rolling(window).skew().shift(1)
        result[f"overnight_positive_{window}"] = (overnight > 0).rolling(window).mean().shift(1)
        result[f"intraday_mean_{window}"] = intraday.rolling(window).mean().shift(1)
        result[f"intraday_std_{window}"] = intraday.rolling(window).std().shift(1)
        result[f"overnight_minus_intraday_{window}"] = (
            overnight.rolling(window).mean() - intraday.rolling(window).mean()
        ).shift(1)
    result["gap_abs_mean_20"] = overnight.abs().rolling(20).mean().shift(1)
    result["gap_jump_rate_60"] = (overnight.abs() > 0.03).rolling(60).mean().shift(1)
    # Multi-session right-side setups.  These encode state transitions rather
    # than merely yesterday's return, and are shifted below to remain causal.
    candle_up = close > open_
    close_up = close > close.shift(1)
    for window in (3, 5, 10):
        prior_all_down = (~close_up).shift(1).rolling(window).sum() == window
        result[f"first_up_after_{window}_down"] = (candle_up & prior_all_down).astype(float).shift(1)
        result[f"trend_efficiency_{window}"] = (
            close.diff(window).abs() / close.diff().abs().rolling(window).sum()
        ).shift(1)
        result[f"up_day_ratio_{window}"] = close_up.rolling(window).mean().shift(1)
    for window in (5, 10, 20, 60):
        moving = close.rolling(window).mean()
        result[f"first_close_above_ma_{window}"] = (
            (close > moving) & (close.shift(1) <= moving.shift(1))
        ).astype(float).shift(1)
        result[f"ma_slope_{window}"] = moving.pct_change(5, fill_method=None).shift(1)
    for window in (20, 60, 120):
        prior_high = close.shift(1).rolling(window).max()
        result[f"breakout_{window}"] = (close / prior_high - 1).shift(1)
        result[f"near_high_{window}"] = (close / close.rolling(window).max()).shift(1)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    result["trend_stack_5_20_60"] = (
        (close > ma20) & (ma20 > ma60) & (ma20 > ma20.shift(5))
    ).astype(float).shift(1)
    result["pullback_reversal_uptrend"] = (
        (close > ma60)
        & (close.shift(1) < close.shift(2))
        & (close > close.shift(1))
        & candle_up
    ).astype(float).shift(1)
    volume = bars["volume"].astype(float)
    result["breakout_volume_confirm_20"] = (
        (close > close.shift(1).rolling(20).max())
        * (volume / volume.shift(1).rolling(20).mean())
    ).shift(1)
    return result


def context() -> pd.DataFrame:
    cache = OUT / "context"
    cache.mkdir(parents=True, exist_ok=True)
    series: dict[str, pd.Series] = {}
    for ticker in [*MARKET, *SECTORS]:
        target = cache / f"{ticker.replace('^', 'IDX_')}.parquet"
        if not target.exists():
            frame = download(f"US.{ticker}", "2023-01-01", "2026-11-06")
            frame.to_parquet(target, index=False)
        frame = pd.read_parquet(target).sort_values("timestamp")
        series[ticker] = frame.set_index("timestamp")["close"].astype(float)
    prices = pd.DataFrame(series).sort_index()
    returns = prices.pct_change(fill_method=None)
    factors = pd.DataFrame(index=prices.index)
    for ticker in MARKET:
        safe = ticker.replace("^", "")
        factors[f"market_ret1_{safe}"] = returns[ticker].shift(1)
        factors[f"market_ret5_{safe}"] = prices[ticker].pct_change(5, fill_method=None).shift(1)
        factors[f"market_vol20_{safe}"] = returns[ticker].rolling(20).std().shift(1)
    for ticker in SECTORS:
        factors[f"sector_ret1_{ticker}"] = returns[ticker].shift(1)
        factors[f"sector_ret5_{ticker}"] = prices[ticker].pct_change(5, fill_method=None).shift(1)
    factors["breadth_sector_positive_1d"] = (returns[SECTORS] > 0).mean(axis=1).shift(1)
    factors["breadth_sector_positive_5d"] = (prices[SECTORS].pct_change(5) > 0).mean(axis=1).shift(1)
    return factors.reset_index()


def main() -> int:
    base = pd.read_parquet(BASE)
    parts = [stock_factors(path) for path in sorted(RAW.glob("US_*.parquet"))]
    stock = pd.concat(parts, ignore_index=True)
    stock_columns = [column for column in stock if column not in {"timestamp", "symbol"}]
    for column in stock_columns:
        stock[column] = stock.groupby("timestamp")[column].rank(pct=True) - 0.5
    panel = base.merge(stock, on=["timestamp", "symbol"], how="left")
    regime = context()
    panel = panel.merge(regime, on="timestamp", how="left")
    local_date = pd.to_datetime(panel["timestamp"], utc=True).dt.tz_convert("America/New_York")
    weekday = local_date.dt.dayofweek
    panel["calendar_dow_sin"] = np.sin(2 * np.pi * weekday / 5)
    panel["calendar_dow_cos"] = np.cos(2 * np.pi * weekday / 5)
    panel["calendar_month_end"] = local_date.dt.is_month_end.astype(float)
    panel["calendar_month_start"] = local_date.dt.is_month_start.astype(float)
    panel["calendar_quarter_end"] = local_date.dt.is_quarter_end.astype(float)
    panel["calendar_opex_week"] = ((local_date.dt.day >= 15) & (local_date.dt.day <= 21)).astype(float)

    trend_markers = (
        "first_up_", "trend_efficiency_", "up_day_ratio_", "first_close_above_",
        "ma_slope_", "breakout_", "near_high_", "trend_stack_", "pullback_reversal_"
    )
    trend_columns = [
        column for column in stock_columns
        if column.startswith(trend_markers) or column == "breakout_volume_confirm_20"
    ]
    overnight_columns = [column for column in stock_columns if column not in trend_columns]
    groups = {
        "base": [column for column in base if column not in {"timestamp", "feature_asof", "symbol", "label_next_open_net"}],
        "overnight_history": overnight_columns,
        "right_side_trend": trend_columns,
        "market_regime": [column for column in regime if column != "timestamp" and not column.startswith("sector_")],
        "sector_regime": [column for column in regime if column.startswith("sector_")],
        "calendar": [column for column in panel if column.startswith("calendar_")],
    }
    target = OUT / "historical_panel_factor_groups_lag1.parquet"
    panel.to_parquet(target, index=False)
    (OUT / "factor_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    for group_name, columns in groups.items():
        if group_name == "base":
            continue
        (OUT / f"feature_columns_base_plus_{group_name}.json").write_text(
            json.dumps([*groups["base"], *columns], indent=2), encoding="utf-8"
        )
    print(json.dumps({"panel": str(target), "rows": len(panel), "groups": {k: len(v) for k, v in groups.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
