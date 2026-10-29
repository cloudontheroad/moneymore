"""Ex-post daily liquidity diagnostics, never an opening-auction fill guarantee.

Tushare daily units: vol=lots of 100 shares, amount=thousand CNY.
https://tushare.pro/document/2?doc_id=27
"""

import numpy as np
import pandas as pd


def daily_participation(fills, market):
    fills, market = fills.copy(), market.copy()
    for frame in (fills, market):
        frame["trade_date"] = pd.to_datetime(frame.trade_date, format="%Y%m%d").dt.strftime(
            "%Y%m%d"
        )
    if market.duplicated(["symbol", "trade_date"]).any():
        raise ValueError("AMBIGUOUS_DAILY_LIQUIDITY")
    if not fills.side.isin(["BUY", "SELL"]).all():
        raise ValueError("UNKNOWN_FILL_SIDE")
    for field in ("quantity", "price"):
        fills[field] = pd.to_numeric(fills[field], errors="raise")
        if not (np.isfinite(fills[field]) & fills[field].gt(0)).all():
            raise ValueError("INVALID_EXECUTED_FILL")
    if not fills.quantity.eq(fills.quantity.round()).all():
        raise ValueError("NONINTEGER_EXECUTED_QUANTITY")
    fills["notional_cny"] = fills.quantity * fills.price
    if not np.isfinite(fills.notional_cny).all():
        raise ValueError("EXECUTED_NOTIONAL_OVERFLOW")
    grouped = fills.groupby(["symbol", "trade_date"], as_index=False).agg(
        gross_shares=("quantity", "sum"),
        gross_notional_cny=("notional_cny", "sum"),
        fill_count=("side", "size"),
    )
    if not np.isfinite(grouped[["gross_shares", "gross_notional_cny"]]).all().all():
        raise ValueError("DAILY_EXECUTED_TOTAL_OVERFLOW")
    rows = grouped.merge(market, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    for field in ("vol", "amount"):
        rows[field] = pd.to_numeric(rows[field], errors="coerce")
    valid = (
        np.isfinite(rows.vol)
        & rows.vol.gt(0)
        & rows.vol.le(np.finfo(float).max / 100)
        & np.isfinite(rows.amount)
        & rows.amount.gt(0)
        & rows.amount.le(np.finfo(float).max / 1000)
    )
    rows["liquidity_status"] = np.where(
        valid, "DAILY_TOTAL_OBSERVED", "MISSING_OR_INVALID_DAILY_LIQUIDITY"
    )
    rows["market_shares"] = rows.vol.where(valid) * 100
    rows["market_notional_cny"] = rows.amount.where(valid) * 1000
    rows["share_participation"] = rows.gross_shares / rows.market_shares
    rows["notional_participation"] = rows.gross_notional_cny / rows.market_notional_cny
    rows["above_total_market_volume"] = valid & rows.share_participation.gt(1)
    summary = {
        "status": "DAILY_TOTAL_DIAGNOSTIC_ONLY_NOT_OPEN_CAPACITY_ACCEPTANCE",
        "symbol_days": len(rows),
        "fills": len(fills),
        "missing_or_invalid_symbol_days": int((~valid).sum()),
        "exceeds_total_daily_volume_symbol_days": int(rows.above_total_market_volume.sum()),
        "maximum_share_participation": float(rows.share_participation.max())
        if valid.any()
        else None,
        "maximum_notional_participation": float(rows.notional_participation.max())
        if valid.any()
        else None,
        "p99_share_participation": float(rows.share_participation.quantile(0.99))
        if valid.any()
        else None,
        "opening_depth_verified": False,
        "used_as_pretrade_input": False,
        "production_changed": False,
    }
    return rows, summary
