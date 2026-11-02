"""Independent, unfilled economic label reconstruction for foundation audits."""
import pandas as pd


def economic_labels(daily, adjustments, sessions, horizon=5):
    if horizon < 1:
        raise ValueError("positive horizon required")
    for frame in (daily, adjustments):
        if frame.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("duplicate economic input")
    combined = daily.merge(adjustments, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    combined["total_price"] = combined.close * combined.adj_factor
    prices = combined.pivot(index="trade_date", columns="ts_code", values="total_price").reindex(sessions)
    result = prices.shift(-horizon) / prices - 1
    result.index = pd.to_datetime(result.index)
    result.index.name = "datetime"
    result.columns.name = "instrument"
    return result.stack(future_stack=True).rename("rebuilt")
