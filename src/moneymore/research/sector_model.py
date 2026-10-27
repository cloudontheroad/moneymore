from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest import run_daily_backtest
from ..config import BacktestConfig
from ..data.research import load_point_in_time_features
from ..data.store import ParquetStore
from ..factors import PreprocessConfig, build_default_registry, preprocess_cross_section
from .bank_model import (
    apply_bank_targets,
    load_bank_backtest_bars,
    topk_dropout_targets,
)
from .bank_timing import (
    VolatilityTargetRiskDegree,
    apply_risk_degree,
    build_bank_market_index,
)
from .metrics import performance_metrics, slice_equity


@dataclass(frozen=True)
class SectorDefinition:
    sector_id: str
    name: str
    etf_code: str
    style: str
    symbols: tuple[str, ...]
    source_weights: dict[str, float]
    factor_weights: dict[str, float]
    top_k: int
    exit_rank: int
    max_replacements: int
    risk_target: float


def build_sector_factor_panel(
    store: ParquetStore,
    definition: SectorDefinition,
) -> pd.DataFrame:
    registry = build_default_registry()
    names = list(definition.factor_weights)
    panels = []
    for symbol in definition.symbols:
        features = load_point_in_time_features(store, symbol)
        computed = registry.compute(features, names)
        computed["date"] = pd.to_datetime(computed["date"])
        computed["month"] = computed["date"].dt.to_period("M")
        computed = (
            computed.sort_values("date")
            .groupby("month", as_index=False)
            .tail(1)
            .drop(columns="month")
        )
        panels.append(computed)
    factors = pd.concat(panels, ignore_index=True)
    for name in names:
        if registry.get(name).direction.value == "low_is_better":
            factors[name] = -factors[name]
    processed = preprocess_cross_section(
        factors,
        names,
        PreprocessConfig(industry_column=None, minimum_assets=5),
    )
    processed["score"] = 0.0
    processed["active_weight"] = 0.0
    for name, weight in definition.factor_weights.items():
        available = processed[name].notna()
        processed.loc[available, "score"] += processed.loc[available, name] * weight
        processed.loc[available, "active_weight"] += weight
    processed["score"] /= processed["active_weight"].replace(0, pd.NA)
    processed["sector"] = definition.sector_id
    processed["style"] = definition.style
    return processed


def build_global_factor_panel(
    store: ParquetStore,
    symbols: list[str],
    factor_weights: dict[str, float],
    *,
    latest_only: bool = False,
) -> pd.DataFrame:
    """Score the entire candidate pool in one cross-section without sector buckets."""
    registry = build_default_registry()
    names = list(factor_weights)
    rows = []
    for symbol in sorted(set(symbols)):
        features = load_point_in_time_features(store, symbol)
        computed = registry.compute(features, names)
        if not computed.empty:
            computed["date"] = pd.to_datetime(computed["date"])
            computed["month"] = computed["date"].dt.to_period("M")
            computed = (
                computed.sort_values("date")
                .groupby("month", as_index=False)
                .tail(1)
                .drop(columns="month")
            )
            rows.append(computed.tail(1) if latest_only else computed)
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "score", "active_weight"])
    factors = pd.concat(rows, ignore_index=True)
    for name in names:
        if registry.get(name).direction.value == "low_is_better":
            factors[name] = -factors[name]
    processed = preprocess_cross_section(
        factors,
        names,
        PreprocessConfig(industry_column=None, minimum_assets=5),
    )
    processed["score"] = 0.0
    processed["active_weight"] = 0.0
    for name, weight in factor_weights.items():
        available = processed[name].notna()
        processed.loc[available, "score"] += processed.loc[available, name] * weight
        processed.loc[available, "active_weight"] += weight
    processed["score"] /= processed["active_weight"].replace(0, pd.NA)
    return processed


def build_global_factor_snapshot(
    store: ParquetStore,
    symbols: list[str],
    factor_weights: dict[str, float],
    *,
    audit_capture: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the live cross-section with bulk parquet reads.

    The former implementation opened five parquet tables once per symbol. At a
    1000-stock universe that meant thousands of full-file scans every evening.
    This path reads each source once and retains only the rolling history needed
    by the longest live factor.
    """
    requested = sorted(set(symbols))
    daily = store.read(
        "daily",
        columns=["ts_code", "trade_date", "close"],
        filters=[("ts_code", "in", requested)],
    )
    adjustment = store.read(
        "adj_factor",
        columns=["ts_code", "trade_date", "adj_factor"],
        filters=[("ts_code", "in", requested)],
    )
    panel = daily.merge(
        adjustment,
        on=["ts_code", "trade_date"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["ts_code", "trade_date"])
    panel = panel.groupby("ts_code", as_index=False, group_keys=False).tail(270)
    latest_factor = panel.groupby("ts_code")["adj_factor"].transform("last")
    panel["signal_close"] = (
        pd.to_numeric(panel["close"], errors="coerce")
        * pd.to_numeric(panel["adj_factor"], errors="coerce")
        / latest_factor
    )
    panel = panel.rename(columns={"ts_code": "symbol", "trade_date": "date"})
    panel["date"] = pd.to_datetime(panel["date"], format="%Y%m%d")
    latest_dates = panel.groupby("symbol")["date"].transform("max")
    latest_mask = panel["date"] == latest_dates

    market_columns = ["dv_ttm", "pb", "pe_ttm"]
    basic = store.read(
        "daily_basic",
        columns=["ts_code", "trade_date", *market_columns],
        filters=[("ts_code", "in", requested)],
    ).rename(columns={"ts_code": "symbol"})
    basic["date"] = pd.to_datetime(basic["trade_date"], format="%Y%m%d")
    latest_by_symbol = panel.loc[latest_mask, ["symbol", "date"]].rename(
        columns={"date": "latest_date"}
    )
    basic = basic.merge(latest_by_symbol, on="symbol", how="inner")
    basic = (
        basic.loc[basic["date"] < basic["latest_date"]]
        .sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .set_index("symbol")
    )
    for column in market_columns:
        panel[column] = np.nan
        panel.loc[latest_mask, column] = panel.loc[latest_mask, "symbol"].map(
            basic[column]
        )

    financial_columns = ["roe", "q_sales_yoy"]
    financial = store.read(
        "fina_indicator",
        columns=["ts_code", "ann_date", "end_date", *financial_columns],
        filters=[("ts_code", "in", requested)],
    ).rename(columns={"ts_code": "symbol"})
    financial["available_date"] = pd.to_datetime(
        financial["ann_date"], format="%Y%m%d"
    ) + pd.offsets.Day(1)
    financial = financial.merge(latest_by_symbol, on="symbol", how="inner")
    financial = (
        financial.loc[financial["available_date"] <= financial["latest_date"]]
        .sort_values(["symbol", "available_date", "end_date"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .set_index("symbol")
    )
    for column in financial_columns:
        panel[column] = np.nan
        panel.loc[latest_mask, column] = panel.loc[latest_mask, "symbol"].map(
            financial[column]
        )

    registry = build_default_registry()
    names = list(factor_weights)
    computed = registry.compute(panel, names)
    factors = computed.loc[computed["date"] == computed["date"].max()].copy()
    if audit_capture is not None:
        audit_capture["raw_factors"] = factors.copy()
    for name in names:
        if registry.get(name).direction.value == "low_is_better":
            factors[name] = -factors[name]
    processed = preprocess_cross_section(
        factors,
        names,
        PreprocessConfig(industry_column=None, minimum_assets=5),
    )
    processed["score"] = 0.0
    processed["active_weight"] = 0.0
    for name, weight in factor_weights.items():
        available = processed[name].notna()
        processed.loc[available, "score"] += processed.loc[available, name] * weight
        processed.loc[available, "active_weight"] += weight
    processed["score"] /= processed["active_weight"].replace(0, pd.NA)
    if audit_capture is not None:
        audit_capture["processed_factors"] = processed.copy()
    return processed


def research_sector(
    store: ParquetStore,
    definition: SectorDefinition,
    config: BacktestConfig,
    start_date: str = "2015-01-01",
    end_date: str = "2026-07-27",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = build_sector_factor_panel(store, definition)
    targets = topk_dropout_targets(
        scores[["date", "symbol", "score"]],
        top_k=definition.top_k,
        exit_rank=definition.exit_rank,
        max_replacements=definition.max_replacements,
        position_weight=0.75 / definition.top_k,
    )
    targets["sector"] = definition.sector_id
    bars = load_bank_backtest_bars(
        store, list(definition.symbols), start_date, end_date
    )
    base = apply_bank_targets(bars, targets)
    market = build_bank_market_index(bars)
    timing = VolatilityTargetRiskDegree(
        annual_target=definition.risk_target,
        minimum_degree=0.15,
        maximum_degree=0.75,
        strategy_id=f"{definition.sector_id}_vol_target",
    )
    degrees = timing.risk_degrees(market)
    signals = apply_risk_degree(base, degrees, base_gross_exposure=0.75)
    result = run_daily_backtest(
        signals, config.model_copy(update={"max_drawdown": 0.99})
    )
    rows = []
    for period, start, end in (
        ("sample_in", "2015-01-01", "2021-12-31"),
        ("sample_out", "2022-01-01", end_date),
    ):
        try:
            metrics = performance_metrics(slice_equity(result.equity, start, end))
        except ValueError:
            continue
        rows.append(
            {
                "sector": definition.sector_id,
                "period": period,
                **metrics,
                "fills": len(result.fills),
                "evidence_status": "CURRENT_CONSTITUENT_BIASED",
            }
        )
    equity = result.equity.copy()
    equity["sector"] = definition.sector_id
    return pd.DataFrame(rows), scores, targets, equity.merge(
        degrees, on="date", how="left", validate="one_to_one"
    )


def inverse_volatility_allocation(
    returns: pd.DataFrame,
    maximum_weight: float = 0.35,
    minimum_weight: float = 0.10,
) -> dict[str, float]:
    volatility = returns.std(ddof=1) * np.sqrt(242)
    raw = 1 / volatility.replace(0, np.nan)
    weights = (raw / raw.sum()).fillna(0.0)
    for _ in range(10):
        weights = weights.clip(lower=minimum_weight, upper=maximum_weight)
        weights /= weights.sum()
        if bool(
            ((weights >= minimum_weight - 1e-9) & (weights <= maximum_weight + 1e-9)).all()
        ):
            break
    return {str(key): float(value) for key, value in weights.items()}
