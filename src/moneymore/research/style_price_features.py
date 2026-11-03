"""Compact causal research inputs. Raw and adjusted paths share one formula."""
import numpy as np
import pandas as pd


def registered_liquidity_on_calendar(basic, calendar, symbols):
    """Existing liquidity descriptor, with calendar gaps kept unobserved.

    Input turnover_rate is provider percentage points; output is decimal.
    This is our registered 20-session mean, not a full Barra liquidity factor.
    """
    from moneymore.factors.builtin import build_default_registry

    if list(calendar) != sorted(set(calendar)) or len(symbols) != len(set(symbols)):
        raise ValueError("DUPLICATE_OR_UNORDERED_LIQUIDITY_GRID")
    grid = pd.MultiIndex.from_product([symbols, calendar], names=["symbol", "date"])
    aligned = basic.set_index(["symbol", "date"]).reindex(grid).reset_index()
    aligned["turnover_rate"] /= 100
    result = build_default_registry().compute(aligned, ["turnover_mean_20"])
    return result, aligned


def price_features(bars, basis="raw"):
    if basis not in ("raw", "adjusted"):
        raise ValueError("unknown basis")
    ratio = bars.adj_factor if basis == "adjusted" else 1.
    close = bars.close * ratio
    previous = close.shift(1)
    ret = close.pct_change(fill_method=None)
    return pd.DataFrame({
        "open_gap": bars.open * ratio / previous - 1,
        "high_gap": bars.high * ratio / previous - 1,
        "low_gap": bars.low * ratio / previous - 1,
        "close_return": ret,
        "volume_change": np.log1p(bars.vol).diff(),
        "amount_change": np.log1p(bars.amount).diff(),
        "return_5": close.pct_change(5, fill_method=None),
        "return_20": close.pct_change(20, fill_method=None),
        "price_to_ma_20": close / close.rolling(20).mean() - 1,
        "price_to_ma_60": close / close.rolling(60).mean() - 1,
        "volatility_20": ret.rolling(20).std(),
        "drawdown_60": close / close.rolling(60).max() - 1,
    }).replace([np.inf, -np.inf], np.nan)


def named_price_risk_exposures(prices, market_cap, lookback=120):
    market_cap = market_cap.reindex(prices.columns)
    returns = prices.pct_change(fill_method=None)
    window = returns.tail(lookback)
    complete = window.notna().all()
    proxy = window.loc[:, complete].mean(axis=1)
    if complete.sum() < 30 or proxy.var() <= 0:
        raise ValueError("insufficient complete market-risk observations")
    beta = pd.Series(np.nan, index=window.columns)
    beta.loc[complete] = window.loc[:, complete].apply(lambda x: x.cov(proxy)) / proxy.var()
    residual = window - pd.DataFrame(np.outer(proxy, beta), index=window.index, columns=window.columns)
    exposure = pd.DataFrame({"log_size": np.log(market_cap.where(market_cap > 0)),
        "momentum60": prices.pct_change(60, fill_method=None).iloc[-1],
        "volatility20": returns.tail(20).std(), "market_beta120": beta,
        "residual_volatility120": residual.std() * np.sqrt(252)})
    exposure.loc[~complete, :] = np.nan
    return exposure, {"market_proxy": "equal_weight_current_pool_complete_names_not_external_index",
        "complete_risk_assets": int(complete.sum()), "risk_days": len(window),
        "universe_is_historical_PIT": False}
