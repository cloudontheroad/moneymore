"""Causal helpers for the isolated three-year rank-cap comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ashare_mainline import purged_split
from .simple_rules import ttm_at


def rolling_split(panel, quarter, validation_sessions=60, training_sessions=504):
    split = purged_split(panel, quarter, validation_sessions, 100)
    dates = sorted(panel.loc[split["train"], "date"].unique())
    split["train"] &= panel.date.ge(
        dates[-training_sessions] if len(dates) > training_sessions else dates[0]
    )
    return split


def prepared_reports(frame):
    frame = frame.copy()
    for name in ["ann_date", "f_ann_date", "end_date"]:
        if name in frame:
            frame[name] = pd.to_datetime(
                frame[name].astype("string"), format="%Y%m%d", errors="coerce"
            )
    if "report_type" in frame:
        frame = frame.loc[pd.to_numeric(frame.report_type, errors="coerce").isin([1, 4])]
    frame["available"] = frame[[c for c in ["ann_date", "f_ann_date"] if c in frame]].max(
        axis=1
    ) + pd.Timedelta(days=1)
    frame = frame.loc[
        frame.available.notna() & frame.end_date.notna() & frame.end_date.le(frame.available)
    ]
    if "update_flag" not in frame:
        frame["update_flag"] = "0"
    frame["update_flag"] = frame.update_flag.fillna("0").astype(str)
    return frame.sort_values(["ts_code", "available", "end_date", "update_flag"])


def financial_events(cashflow, indicators, income, max_age=200):
    """One row per symbol/publication event; TTM components all known at that event.

    No global latest-version deduplication: later corrections only take effect
    when announced. Vendor revisions without original versions remain a limitation.
    """
    groups = [
        {s: g for s, g in prepared_reports(f).groupby("ts_code", sort=False)}
        for f in [cashflow, indicators, income]
    ]
    rows = []
    for symbol in sorted(set(groups[0]) | set(groups[1]) | set(groups[2])):
        blank = pd.DataFrame(columns=["available", "end_date"])
        frames = [g.get(symbol, blank) for g in groups]
        dates = sorted(set().union(*(set(f.available) for f in frames)))
        for day in dates:
            c, f, i = [
                a.loc[a.available.le(day)].drop_duplicates("end_date", keep="last") for a in frames
            ]
            if f.empty:
                continue
            end = f.end_date.max()
            if (day - end).days > max_age:
                continue
            latest = f.set_index("end_date").loc[end]
            periods = (
                pd.period_range(end=end.to_period("Q"), periods=4, freq="Q")
                .to_timestamp(how="end")
                .normalize()
            )
            q = f.set_index("end_date").q_roe.reindex(periods)
            ocf = ttm_at(c, "n_cashflow_act", end) if "n_cashflow_act" in c else np.nan
            profit = ttm_at(i, "operate_profit", end) if "operate_profit" in i else np.nan
            rows.append(
                {
                    "symbol": symbol,
                    "available": day,
                    "report_end": end,
                    "fcff": ttm_at(f, "fcff", end),
                    "netdebt": latest.netdebt,
                    "operating_cash": ocf,
                    "cash_conversion": ocf / profit
                    if np.isfinite(profit) and profit > 0
                    else np.nan,
                    "quarter_roe_std": q.std(ddof=0) if q.notna().all() else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["available", "symbol"])


def attach_financial_events(panel, events, max_age=200):
    left = panel[["date", "symbol", "total_mv"]].copy()
    left["asof"] = pd.to_datetime(left.date, format="%Y%m%d")
    merged = pd.merge_asof(
        left.sort_values(["asof", "symbol"]),
        events.sort_values(["available", "symbol"]),
        left_on="asof",
        right_on="available",
        by="symbol",
        direction="backward",
    )
    fresh = (merged["asof"] - merged.report_end).dt.days.between(0, max_age)
    mv = merged.total_mv * 10000.0  # Tushare capitalization is 10,000 CNY.
    ev = mv + merged.netdebt
    merged["fcff_to_ev"] = (merged.fcff / ev.where(ev.gt(0))).where(fresh)
    merged["netdebt_to_mv"] = (merged.netdebt / mv.where(mv.gt(0))).where(fresh)
    merged["operating_cash_to_mv"] = (merged.operating_cash / mv.where(mv.gt(0))).where(fresh)
    for name in ["cash_conversion", "quarter_roe_std"]:
        merged[name] = merged[name].where(fresh)
    return merged[
        [
            "date",
            "symbol",
            "fcff_to_ev",
            "netdebt_to_mv",
            "operating_cash_to_mv",
            "cash_conversion",
            "quarter_roe_std",
            "available",
            "report_end",
        ]
    ]


def causal_volatility_scale(
    market_returns, signal_dates, calendar, calibration_start, calibration_end
):
    ret = market_returns.copy()
    ret.index = pd.to_datetime(ret.index)
    variance = ret.groupby(ret.index.to_period("M")).var(ddof=1)
    count = ret.groupby(ret.index.to_period("M")).count()
    variance = variance.where(count.ge(10))
    calibration = variance.loc[
        (variance.index >= pd.Period(pd.Timestamp(calibration_start), freq="M"))
        & (variance.index <= pd.Period(pd.Timestamp(calibration_end), freq="M"))
    ].dropna()
    if len(calibration) < 12:
        raise ValueError("INSUFFICIENT_PRE_EVALUATION_VARIANCE")
    c = float(calibration.median())
    cal = pd.DatetimeIndex(pd.to_datetime(calendar))
    month_ends = pd.Series(cal, index=cal.to_period("M")).groupby(level=0).max()
    result = {}
    for day in signal_dates:
        stamp = pd.Timestamp(day)
        month = stamp.to_period("M")
        completed = month if stamp == month_ends.loc[month] else month - 1
        v = variance.get(completed, np.nan)
        if pd.Timestamp(calibration_end) >= stamp or not np.isfinite(v) or v <= 0:
            raise ValueError("INVALID_VARIANCE_TIME_CONTRACT:" + day)
        result[day] = min(1.0, c / float(v))
    return result, c


def sequence_indices(frame, calendar, length):
    """Exact exchange sessions, not last N member observations across an exit gap.

    Missing histories use zero padding, never future rows or another instrument.
    Returns indices into a feature array with one leading all-zero padding row.
    """
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("DUPLICATE_SEQUENCE_KEYS")
    dates = pd.Index(calendar)
    positions = dates.get_indexer(frame.date)
    if (positions < 0).any():
        raise ValueError("SEQUENCE_DATE_OUTSIDE_CALENDAR")
    keys = pd.MultiIndex.from_frame(frame[["date", "symbol"]])
    columns = []
    for lag in range(length - 1, -1, -1):
        offset = positions - lag
        target = dates.take(np.maximum(offset, 0))
        index = keys.get_indexer(pd.MultiIndex.from_arrays([target, frame.symbol])) + 1
        index[offset < 0] = 0
        columns.append(index)
    return np.stack(columns, axis=1).astype("int32")
