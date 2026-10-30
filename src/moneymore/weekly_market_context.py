"""As-of market observations, not a new exposure formula or predictive model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def market_context(bars, calendar, sectors, *, as_of):
    cutoff = pd.Timestamp(as_of).strftime("%Y%m%d")
    sessions = sorted({str(d).replace("-", "")[:8] for d in calendar if str(d).replace("-", "")[:8] <= cutoff})
    if not sessions:
        raise ValueError("no completed exchange sessions")
    expected = sessions[-1]
    data = bars.loc[bars.ts_code.isin(sectors)].copy()
    data["trade_date"] = data.trade_date.astype(str)
    data = data.loc[data.trade_date <= expected]
    if data.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("duplicate daily bars")
    prices = data.pivot(index="trade_date", columns="ts_code", values="close").reindex(sessions)
    prices = prices.reindex(columns=sorted(sectors)).astype(float)
    prices = prices.where(prices > 0)
    returns = prices.pct_change(fill_method=None)
    valid = prices.loc[expected].notna()
    ret5 = prices.pct_change(5, fill_method=None)
    ret20 = prices.pct_change(20, fill_method=None)
    daily = returns.mean(axis=1)
    # This proxy describes today's pool. It is not a historical dynamic-pool benchmark.
    series = pd.DataFrame({
        "equal_weight_return": daily,
        "breadth_ma20": (prices > prices.rolling(20).mean()).where(prices.rolling(20).count() == 20).mean(axis=1),
        "breadth_ma60": (prices > prices.rolling(60).mean()).where(prices.rolling(60).count() == 60).mean(axis=1),
        "volatility20": daily.rolling(20).std() * np.sqrt(252),
        "observed_stocks": prices.notna().sum(axis=1),
    })
    groups = {}
    labels = pd.Series(sectors)
    for industry in sorted(labels.unique()):
        names = labels.index[labels == industry]
        groups[industry] = {
            "members": len(names),
            "observed": int(valid.reindex(names).sum()),
            "return5_mean": ret5.loc[expected, names].mean(),
            "return20_mean": ret20.loc[expected, names].mean(),
        }
    summary = {
        "as_of": cutoff, "expected_session": expected,
        "latest_observed": str(data.trade_date.max()) if len(data) else None,
        "requested": len(sectors), "observed": int(valid.sum()),
        "missing_symbols": sorted(valid.index[~valid].tolist()),
        "coverage": float(valid.mean()),
        "industries": groups,
        "macro": {"status": "NOT_CONNECTED", "used_in_training": False},
        "limitations": ["current-universe descriptive snapshot, not PIT benchmark",
                        "raw close returns may include corporate-action effects",
                        "missing bars are not forward filled; suspension not classified",
                        "macro publication-time inputs not connected"],
        "used_in_model": False,
    }
    # Strict JSON: missing values are null, never NaN/Infinity.
    import json
    summary = json.loads(pd.Series(summary).to_json(force_ascii=False))
    return summary, series
