"""Reconstruct actual replay holdings and audit named, benchmark-relative styles."""

import numpy as np
import pandas as pd


def stock_wealth_change(prior_values, values, fills, corporate_ledger):
    """Exact PnL from economic stock values, trade cash flows and paid dividends.

    Values include ex-date cash/stock receivables until settlement. This is
    bookkeeping, not causal stock-selection attribution.
    """
    pnl = {
        s: values.get(s, 0.0) - prior_values.get(s, 0.0) for s in set(values) | set(prior_values)
    }
    for fill in fills.itertuples():
        if fill.side not in ("BUY", "SELL"):
            raise ValueError("UNKNOWN_TRADE_SIDE")
        flow = (1 if fill.side == "SELL" else -1) * fill.quantity * fill.price - fill.fee
        pnl[fill.symbol] = pnl.get(fill.symbol, 0.0) + flow
    for item in corporate_ledger.itertuples():
        if item.action_type == "CASH_DIVIDEND":
            pnl[item.symbol] = pnl.get(item.symbol, 0.0) + item.cash_amount
    if not np.isfinite(list(pnl.values())).all():
        raise ValueError("NONFINITE_STOCK_WEALTH_CHANGE")
    return pnl


def reconstruct_positions(fills, corporate_ledger, dates):
    holdings, snapshots = {}, {}
    for date in dates:
        for row in fills.loc[fills.trade_date.astype(str).eq(date)].itertuples():
            holdings[row.symbol] = holdings.get(row.symbol, 0) + (
                row.quantity if row.side == "BUY" else -row.quantity
            )
        for row in corporate_ledger.loc[
            corporate_ledger.trade_date.astype(str).eq(date)
        ].itertuples():
            if row.action_type == "STOCK_DIVIDEND":
                holdings[row.symbol] = holdings.get(row.symbol, 0) + row.share_quantity
        if any(q < 0 for q in holdings.values()):
            raise ValueError("NEGATIVE_RECONSTRUCTED_HOLDING")
        snapshots[date] = {s: int(q) for s, q in holdings.items() if q > 0}
    return snapshots


def reconstruct_cash(fills, corporate_ledger, dates, initial_cash):
    cash, result = float(initial_cash), {}
    for date in dates:
        for row in fills.loc[fills.trade_date.astype(str).eq(date)].itertuples():
            cash += (-1 if row.side == "BUY" else 1) * row.quantity * row.price - row.fee
        for row in corporate_ledger.loc[
            corporate_ledger.trade_date.astype(str).eq(date)
        ].itertuples():
            if row.action_type == "CASH_DIVIDEND":
                cash += row.cash_amount
        result[date] = cash
    return result


def exposure_snapshot(holdings, close, equity, features, pool):
    valid_pool = features.reindex(pool).replace([np.inf, -np.inf], np.nan).dropna()
    if valid_pool.empty or set(holdings) - set(valid_pool.index):
        return {
            "status": "MISSING_HELD_STYLE",
            "missing_symbols": sorted(set(holdings) - set(valid_pool.index)),
        }
    if equity <= 0 or not np.isfinite(close.reindex(list(holdings))).all():
        return {"status": "INVALID_MARK_OR_EQUITY"}
    z = (valid_pool - valid_pool.mean()) / valid_pool.std(ddof=0).replace(0, 1)
    values = pd.Series({s: q * close[s] for s, q in holdings.items()}, dtype=float)
    gross = float(values.sum() / equity)
    sleeve = values / values.sum() if values.sum() > 0 else values
    # Active STOCK-sleeve exposures separate stock selection from cash level.
    active = (sleeve.reindex(z.index, fill_value=0) - 1 / len(z)) @ z
    return {
        "status": "MEASURED",
        "actual_gross": gross,
        "style_pool_size": len(z),
        "active_stock_sleeve_exposures": active.to_dict(),
        "account_exposures": ((values.reindex(z.index, fill_value=0) / equity) @ z).to_dict(),
    }


def economic_daily_return(prior_close, close, stock_ratio, cash_per_share):
    """Raw-price total return proxy using official implemented ex-date actions."""
    result = (
        close * (1 + stock_ratio.reindex(close.index, fill_value=0))
        + cash_per_share.reindex(close.index, fill_value=0)
    ) / prior_close - 1
    return result.where(prior_close.gt(0) & close.gt(0)).replace([np.inf, -np.inf], np.nan)
