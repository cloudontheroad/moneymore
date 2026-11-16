"""Isolated adapters for the native Qlib model; no broker or live-store writes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FUNDAMENTALS = ["ep", "bp", "roe", "sales_yoy", "roe_change", "sales_change"]
MOMENTUM = ["mom6_1", "mom12_1"]
AGGREGATES = [f"{name}_comparable" for name in ("value_quality", "improvement", "momentum")]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_keys(frame, keys=("date", "symbol")):
    if frame[list(keys)].isna().any().any() or frame.duplicated(list(keys)).any():
        raise ValueError("INVALID_PANEL_KEYS")


def validate_financial_dates(frame):
    """The reused panel applies conservative publication plus one calendar day."""
    dates = pd.to_datetime(frame.date, format="%Y%m%d")
    requirements = {
        "roe": ["roe_ann_date"],
        "sales_yoy": ["q_sales_yoy_ann_date"],
        "roe_change": ["roe_ann_date", "previous_roe_ann_date"],
        "sales_change": ["q_sales_yoy_ann_date", "previous_q_sales_yoy_ann_date"],
    }
    for value, sources in requirements.items():
        for source in sources:
            available = pd.to_datetime(frame[source], format="%Y%m%d", errors="coerce")
            valid = available.notna() & ((available + pd.Timedelta(days=1)) <= dates)
            if (frame[value].notna() & ~valid).any():
                raise ValueError(f"FINANCIAL_NOT_VISIBLE:{value}:{source}")


def market_context(bars, membership, minimum_coverage=0.8, calendar=None):
    """Daily known-membership proxy, not an official index or an exposure formula."""
    assert_keys(bars, ("trade_date", "ts_code"))
    assert_keys(membership)
    close = (
        bars.assign(adjusted=bars.close * bars.adj_factor)
        .pivot(index="trade_date", columns="ts_code", values="adjusted")
        .sort_index()
    )
    if calendar is not None:
        close = close.reindex(pd.Index(calendar).astype(str))
    close = close.where(close.gt(0))
    ret = close.pct_change(fill_method=None)
    keys = pd.MultiIndex.from_frame(membership[["date", "symbol"]])

    def member_values(wide):
        return wide.stack(future_stack=True).reindex(keys).to_numpy()

    sample = membership[["date", "symbol"]].copy()
    sample["return"] = member_values(ret)
    for window in (20, 60):
        ma = close.rolling(window, min_periods=window).mean()
        above = (close > ma).astype(float).where(close.notna() & ma.notna())
        sample[f"breadth{window}"] = member_values(above)
    groups = sample.groupby("date", sort=True)
    daily = groups[["return", "breadth20", "breadth60"]].mean()
    coverage = groups[["return", "breadth20", "breadth60"]].count().div(groups.size(), axis=0)
    daily = daily.where(coverage >= minimum_coverage)
    # Reindex to the exchange calendar: an entirely absent member day must break windows.
    daily = daily.reindex(
        close.index[(close.index >= sample.date.min()) & (close.index <= sample.date.max())]
    )
    r = daily["return"]
    result = pd.DataFrame(index=daily.index)
    result["market_return1"] = r
    for window in (5, 20, 60):
        result[f"market_return{window}"] = r.rolling(window).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )
    result["market_volatility20"] = r.rolling(20).std() * np.sqrt(252)
    result["market_drawdown60"] = r.rolling(60).apply(
        lambda x: np.prod(1 + x) / max(1.0, np.cumprod(1 + x).max()) - 1, raw=True
    )
    result["market_breadth20"] = daily.breadth20
    result["market_breadth60"] = daily.breadth60
    return result.rename_axis("date"), coverage.rename_axis("date")


def purged_split(panel, quarter, validation_sessions=60, minimum_train_sessions=100):
    """Three chronological partitions; endpoints cannot touch next partition."""
    period = pd.Period(quarter, freq="Q")
    start, end = period.start_time.strftime("%Y%m%d"), period.end_time.strftime("%Y%m%d")
    test = panel.date.between(start, end)
    if not test.any():
        raise ValueError("EMPTY_TEST_PERIOD")
    first_test = panel.loc[test, "date"].min()
    mature = panel.label_end_20.notna() & panel.label_end_20.lt(first_test)
    available_days = sorted(panel.loc[mature & panel.date.lt(first_test), "date"].unique())
    if len(available_days) < validation_sessions + minimum_train_sessions:
        raise ValueError("INSUFFICIENT_HISTORY")
    valid_start = available_days[-validation_sessions]
    valid = mature & panel.date.ge(valid_start) & panel.date.lt(first_test)
    train = (
        panel.label_end_20.notna() & panel.label_end_20.lt(valid_start) & panel.date.lt(valid_start)
    )
    if panel.loc[train, "date"].nunique() < minimum_train_sessions:
        raise ValueError("INSUFFICIENT_PURGED_TRAINING")
    return {"train": train, "valid": valid, "test": test}


def rank_target(frame):
    """Cross-sectional rank units, NEVER interpreted as predicted percentage return."""
    return (frame.groupby("date").label_20.rank(pct=True) - 0.5) * np.sqrt(12)


def fit_processor(train, columns, categorical=False):
    values = train[columns].replace([np.inf, -np.inf], np.nan)
    # Future values must not decide which features exist at this fold.
    active = values.columns[values.nunique(dropna=True).gt(1)].tolist()
    if not active:
        raise ValueError("NO_TRAINING_FEATURES")
    values = values[active]
    median = values.median()
    scale = (values.quantile(0.75) - values.quantile(0.25)) / 1.349
    scale = scale.where(scale.gt(1e-8), values.std()).fillna(1.0).clip(lower=1e-8)
    categories = sorted(train.industry.dropna().astype(str).unique()) if categorical else []
    return {
        "input_columns": list(columns),
        "active": active,
        "median": median,
        "scale": scale,
        "dropped_training_constant_or_missing": sorted(set(columns) - set(active)),
        "categories": categories,
        "categorical": categorical,
        "fit_first": str(train.date.min()),
        "fit_last": str(train.date.max()),
    }


def transform(processor, frame):
    active = processor["active"]
    values = frame[active].replace([np.inf, -np.inf], np.nan)
    missing = values.isna().astype("float32").add_prefix("missing__")
    normalized = (
        (values.fillna(processor["median"]) - processor["median"]) / processor["scale"]
    ).clip(-8, 8)
    result = pd.concat([normalized, missing], axis=1)
    if processor["categorical"]:
        for index, category in enumerate(processor["categories"]):
            result[f"industry_{index}"] = frame.industry.eq(category).astype(float)
        result["industry_unknown"] = (~frame.industry.isin(processor["categories"])).astype(float)
    result = result.astype("float32").reset_index(drop=True)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("NONFINITE_TRANSFORM")
    return result


class NativeDataset:
    """Qlib dataset contract adapter; inference does not need or inspect labels."""

    def __init__(self, frames):
        self.frames = frames

    def prepare(self, segments, col_set=None, **kwargs):
        if isinstance(segments, list):
            return tuple(self.prepare(segment, col_set=col_set) for segment in segments)
        frame = self.frames[segments]
        if col_set == "feature" and not isinstance(frame.columns, pd.MultiIndex):
            return frame
        return frame[col_set] if isinstance(col_set, str) else frame


def learning_frame(features, target):
    result = features.copy()
    result.columns = pd.MultiIndex.from_product([["feature"], result.columns])
    result[("label", "rank20")] = np.asarray(target)
    return result


def predict(model, kind, features):
    if kind == "ridge":
        return model.predict(features)
    return model.predict(NativeDataset({"test": features}), segment="test").to_numpy()


def prediction_diagnostics(frame, evaluation_end):
    """Daily signal diagnostics only; no executable P&L or Sharpe claim."""
    rows = []
    for date, group in frame.groupby("date", sort=True):
        ranked = group.sort_values(["score", "symbol"], ascending=[False, True])
        top = ranked.head(50)
        eligible = (
            group.label_end_20.notna()
            & group.label_end_20.le(evaluation_end)
            & group.label_20.notna()
        )
        evaluated = group.loc[eligible]
        ic = (
            evaluated.score.rank().corr(evaluated.label_20.rank())
            if len(evaluated) > 2 and evaluated.score.nunique() > 1
            else np.nan
        )
        board = top.symbol.str.startswith(("30", "68"))
        rows.append(
            {
                "date": date,
                "predicted_rows": len(group),
                "distinct_scores": group.score.nunique(),
                "score_std": group.score.std(),
                "evaluated_rows": len(evaluated),
                "rank_ic": ic,
                "top50_chinext_star_fraction": board.mean(),
                "top50_largest_industry_fraction": top.industry.fillna("UNKNOWN")
                .value_counts(normalize=True)
                .max(),
            }
        )
    return pd.DataFrame(rows)
