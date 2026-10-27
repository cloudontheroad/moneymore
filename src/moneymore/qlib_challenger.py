from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data.research import load_total_return_stock_bars
from .data.store import ParquetStore
from .portfolio_constructor import global_topk_portfolio, trailing_return_correlation

_LIVE_DATASET_CACHE: dict[tuple[object, ...], pd.DataFrame] = {}


@dataclass(frozen=True)
class ChallengerMetrics:
    model_id: str
    segment: str
    samples: int
    rank_ic: float
    rank_ic_ir: float
    top_k_excess_return: float
    average_turnover: float
    cost_adjusted_top_k_excess_return: float


class QlibPanelDataset:
    """Minimal DatasetH-compatible adapter backed by point-in-time local data."""

    def __init__(self, frame: pd.DataFrame, segments: dict[str, tuple[str, str]]) -> None:
        self.frame = frame.sort_index()
        self.segments = {
            name: slice(pd.Timestamp(start), pd.Timestamp(end))
            for name, (start, end) in segments.items()
        }

    def prepare(
        self,
        segment: str | slice,
        col_set: str | list[str] | None = None,
        data_key: str | None = None,
    ) -> pd.DataFrame:
        del data_key
        selected = self.segments[segment] if isinstance(segment, str) else segment
        dates = self.frame.index.get_level_values("datetime")
        result = self.frame.loc[(dates >= selected.start) & (dates <= selected.stop)]
        if col_set is None:
            return result
        return result.loc[:, col_set]


def build_challenger_dataset(
    store: ParquetStore,
    symbol_sectors: dict[str, str],
    sequence_length: int = 60,
    label_horizon: int = 5,
    require_label: bool = True,
    feature_count: int = 6,
    live_as_of: str | None = None,
) -> pd.DataFrame:
    from .challenger_diagnostics import align_trading_observations, trading_dates

    sessions = trading_dates(store, live_as_of or "99991231")
    cache_key: tuple[object, ...] | None = None
    if live_as_of is not None:
        cache_key = (
            str(store.root.resolve()),
            live_as_of,
            tuple(sorted(symbol_sectors)),
            sequence_length,
            label_horizon,
            require_label,
            feature_count,
        )
        cached = _LIVE_DATASET_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()
    panels = []
    live_bars: dict[str, pd.DataFrame] = {}
    if live_as_of is not None:
        requested = sorted(symbol_sectors)
        market = store.read(
            "daily",
            columns=[
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ],
            filters=[("ts_code", "in", requested)],
        )
        market = market.loc[market["trade_date"].astype(str) <= live_as_of].copy()
        market = (
            market.sort_values(["ts_code", "trade_date"])
            .groupby("ts_code", as_index=False, group_keys=False)
            .tail(sequence_length + 65)
            .rename(
                columns={
                    "trade_date": "date",
                    "open": "raw_open",
                    "high": "raw_high",
                    "low": "raw_low",
                    "close": "raw_close",
                }
            )
        )
        live_bars = {
            str(symbol): frame.copy() for symbol, frame in market.groupby("ts_code", sort=False)
        }
    for symbol, sector in sorted(symbol_sectors.items()):
        if live_as_of is None:
            bars = load_total_return_stock_bars(store, symbol).sort_values("date").copy()
        else:
            bars = live_bars.get(symbol, pd.DataFrame()).sort_values("date").copy()
            if bars.empty:
                continue
        bars = align_trading_observations(bars, sessions)
        close = bars["raw_close"].astype(float)
        previous = close.shift(1)
        feature_frame = pd.DataFrame(
            {
                "open_gap": bars["raw_open"].astype(float) / previous - 1,
                "high_gap": bars["raw_high"].astype(float) / previous - 1,
                "low_gap": bars["raw_low"].astype(float) / previous - 1,
                "close_return": close / previous - 1,
                "volume_change": np.log1p(bars["vol"].astype(float)).diff(),
                "amount_change": np.log1p(bars["amount"].astype(float)).diff(),
                "return_5": close.pct_change(5, fill_method=None),
                "return_20": close.pct_change(20, fill_method=None),
                "price_to_ma_20": close / close.rolling(20).mean() - 1,
                "price_to_ma_60": close / close.rolling(60).mean() - 1,
                "volatility_20": close.pct_change(fill_method=None).rolling(20).std(),
                "drawdown_60": close / close.rolling(60).max() - 1,
            }
        ).replace([np.inf, -np.inf], np.nan)
        if feature_count not in {6, 12}:
            raise ValueError("feature_count must be 6 or 12")
        feature_frame = feature_frame.iloc[:, :feature_count]
        feature_columns: dict[str, pd.Series] = {}
        for feature in feature_frame:
            for lag in range(sequence_length - 1, -1, -1):
                feature_columns[f"{feature}_lag{lag:02d}"] = feature_frame[feature].shift(lag)
        future_return = close.shift(-label_horizon) / close - 1
        panel = pd.DataFrame(feature_columns)
        panel["raw_label"] = future_return
        panel["datetime"] = pd.to_datetime(bars["date"])
        panel["instrument"] = symbol
        panel["sector"] = sector
        panels.append(panel)
    combined = pd.concat(panels, ignore_index=True)
    combined["label"] = combined["raw_label"] - combined.groupby(["datetime", "sector"])[
        "raw_label"
    ].transform("mean")
    feature_names = [column for column in combined.columns if "_lag" in column]
    required = [*feature_names, *(["label"] if require_label else [])]
    combined = combined.dropna(subset=required)
    # Live inference only scores the latest available cross-section.  Keeping
    # the preceding feature-building window here made every ensemble member
    # repeatedly index and carry tens of thousands of rows it could never use.
    if live_as_of is not None and not combined.empty:
        latest_live_date = combined["datetime"].max()
        combined = combined.loc[combined["datetime"] == latest_live_date].copy()
    combined = combined.set_index(["datetime", "instrument"]).sort_index()
    features = combined[feature_names].astype("float32")
    labels = combined[["label"]].astype("float32")
    features.columns = pd.MultiIndex.from_product([["feature"], feature_names])
    labels.columns = pd.MultiIndex.from_product([["label"], ["LABEL0"]])
    result = pd.concat([features, labels], axis=1)
    if cache_key is not None:
        _LIVE_DATASET_CACHE.clear()
        _LIVE_DATASET_CACHE[cache_key] = result.copy()
    return result


def evaluate_predictions(
    predictions: pd.Series,
    labels: pd.Series,
    model_id: str,
    segment: str,
    top_k: int,
    symbol_sectors: dict[str, str] | None = None,
    round_trip_cost_rate: float = 0.00106,
    rebalance_interval: int = 1,
    exit_rank_per_sector: int | None = None,
    max_replacements: int | None = None,
    minimum_weight: float = 0.05,
    maximum_weight: float = 0.15,
    price_panel: pd.DataFrame | None = None,
    correlation_lookback: int = 120,
    correlation_penalty: float = 0.0,
    cluster_correlation_threshold: float | None = None,
    maximum_cluster_members: int | None = None,
    correlations: dict[pd.Timestamp, pd.DataFrame] | None = None,
    weighting_method: str = "rank_linear",
    confidence_temperature: float = 1.0,
    confidence_shrinkage: float = 0.35,
) -> ChallengerMetrics:
    aligned = pd.concat([predictions.rename("score"), labels.rename("label")], axis=1).dropna()
    daily_ic = (
        aligned.groupby(level="datetime")
        .apply(
            lambda frame: frame["score"].corr(frame["label"], method="spearman"),
            include_groups=False,
        )
        .dropna()
    )
    selections: list[set[str]] = []
    top_returns: list[float] = []
    held: set[str] = set()
    held_weights: dict[str, float] = {}
    for day_number, (date, day) in enumerate(aligned.groupby(level="datetime")):
        instruments = day.index.get_level_values("instrument").astype(str)
        working = day.assign(instrument=instruments)
        if day_number % rebalance_interval != 0 and held:
            selected = working.loc[working["instrument"].isin(held)]
        else:
            chosen, held_weights, ranked = global_topk_portfolio(
                working,
                held,
                symbol_column="instrument",
                top_k=top_k,
                exit_rank=int(exit_rank_per_sector or top_k),
                max_replacements=int(max_replacements or top_k),
                minimum_weight=min(minimum_weight, 1.0 / top_k),
                maximum_weight=max(maximum_weight, 1.0 / top_k),
                correlation=(
                    correlations.get(pd.Timestamp(date))
                    if correlations is not None
                    else trailing_return_correlation(price_panel, date, correlation_lookback)
                    if price_panel is not None
                    else None
                ),
                correlation_penalty=correlation_penalty,
                cluster_correlation_threshold=cluster_correlation_threshold,
                maximum_cluster_members=maximum_cluster_members,
                weighting_method=weighting_method,
                confidence_temperature=confidence_temperature,
                confidence_shrinkage=confidence_shrinkage,
            )
            selected = ranked.loc[ranked["instrument"].astype(str).isin(chosen)]
        held = set(selected["instrument"])
        selections.append(held)
        top_returns.append(
            float(
                sum(
                    float(row["label"]) * held_weights[str(row["instrument"])]
                    for row in selected.to_dict("records")
                )
            )
        )
    turnovers = [
        len(current.symmetric_difference(previous)) / max(len(current) + len(previous), 1)
        for previous, current in pairwise(selections)
    ]
    average_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    top_k_excess_return = float(np.mean(top_returns)) if top_returns else 0.0
    return ChallengerMetrics(
        model_id=model_id,
        segment=segment,
        samples=len(aligned),
        rank_ic=float(daily_ic.mean()) if not daily_ic.empty else 0.0,
        rank_ic_ir=(
            float(daily_ic.mean() / daily_ic.std(ddof=1))
            if len(daily_ic) > 1 and daily_ic.std(ddof=1)
            else 0.0
        ),
        top_k_excess_return=top_k_excess_return,
        average_turnover=average_turnover,
        cost_adjusted_top_k_excess_return=(
            top_k_excess_return - average_turnover * round_trip_cost_rate
        ),
    )


def challenger_universe(root: Path, store: ParquetStore) -> dict[str, str]:
    try:
        from .strategy_universe import active_strategy_universe

        active = active_strategy_universe(store)
        return dict(zip(active["symbol"].astype(str), active["industry"].astype(str)))
    except (FileNotFoundError, ValueError):
        pass
    config = yaml.safe_load((root / "configs" / "sector_models.yaml").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for sector, item in config["universes"].items():
        for symbol in item["holdings"]:
            result.setdefault(symbol, sector)
    membership = store.read(
        "universe_membership",
        filters=[("universe", "==", "bank_cn")],
    )
    latest = membership.loc[
        pd.to_datetime(membership["date"]) == pd.to_datetime(membership["date"]).max()
    ]
    result.update({str(symbol): "bank" for symbol in latest["ts_code"]})
    return result


def save_challenger_artifact(payload: dict[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable challenger artifact differs: {target}")
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
    return target


def metrics_payload(metrics: ChallengerMetrics) -> dict[str, object]:
    return asdict(metrics)


def rank_blend_predictions(
    predictions: list[tuple[pd.Series, float]],
) -> pd.Series:
    """Blend heterogeneous models on comparable daily percentile ranks."""
    if not predictions:
        raise ValueError("at least one prediction series is required")
    total_weight = sum(max(float(weight), 0.0) for _, weight in predictions)
    if total_weight <= 0:
        raise ValueError("prediction weights must contain a positive value")
    blended = None
    for series, weight in predictions:
        ranked = series.groupby(level="datetime").rank(pct=True, method="average")
        contribution = ranked * (max(float(weight), 0.0) / total_weight)
        blended = contribution if blended is None else blended.add(contribution, fill_value=0.0)
    assert blended is not None
    return blended.sort_index()


def research_gate_diagnostics(
    research: dict[str, Any], fallback_config: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Evaluate the immutable protocol stored with a candidate artifact."""
    protocol = research.get("protocol") or fallback_config
    model_id = str(protocol["model_id"])
    gate = protocol["research_gate"]
    metrics = {str(row.get("model_id")): row for row in research.get("metrics", [])}
    metric = metrics.get(model_id, {})
    stability = research.get("stability", {})
    failures: list[str] = []
    checks = (
        (int(metric.get("samples", 0)) >= int(gate["minimum_samples"]), "MINIMUM_SAMPLES"),
        (float(metric.get("rank_ic", -1)) >= float(gate["minimum_rank_ic"]), "RANK_IC"),
        (float(metric.get("rank_ic_ir", -1)) >= float(gate["minimum_rank_ic_ir"]), "RANK_IC_IR"),
        (
            float(metric.get("cost_adjusted_top_k_excess_return", -1))
            > float(gate["minimum_cost_adjusted_excess_return"]),
            "COST_ADJUSTED_EXCESS_RETURN",
        ),
        (int(stability.get("seed_count", 0)) >= int(gate["minimum_seed_count"]), "SEED_COUNT"),
        (
            float(stability.get("positive_seed_ratio", 0))
            >= float(gate["minimum_positive_seed_ratio"]),
            "SEED_STABILITY",
        ),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    if str(protocol.get("protocol_version", "legacy")) == "challenger_v2":
        if int(research.get("universe_size", 0)) < int(gate.get("minimum_universe_size", 0)):
            failures.append("UNIVERSE_SIZE")
        lightgbm = metrics.get("qlib_lightgbm_alpha360_v1", {})
        if (
            gate.get("require_positive_lightgbm_rank_ic", False)
            and float(lightgbm.get("rank_ic", -1)) <= 0
        ):
            failures.append("CROSS_MODEL_LIGHTGBM_RANK_IC")
    return not failures, failures


def evaluate_forward_observations(
    root: Path,
    store: ParquetStore,
    label_horizon: int,
) -> dict[str, object]:
    """Compatibility entry point for the active model's bounded evaluator."""
    from .challenger_diagnostics import (
        evaluate_forward_reports,
        prediction_reports,
        trading_dates,
        write_diagnostic,
    )

    as_of = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    calendar = trading_dates(store, as_of)
    reports = prediction_reports(root / "state/qlib-challenger-shadow", as_of)
    symbols = sorted({str(score["instrument"]) for report in reports for score in report["scores"]})
    first = min((report["trade_date"] for report in reports), default=as_of)
    market = store.read(
        "daily",
        columns=["ts_code", "trade_date", "close"],
        filters=[
            ("trade_date", ">=", first),
            ("trade_date", "<=", as_of),
            ("ts_code", "in", symbols),
        ],
    )
    payload = evaluate_forward_reports(
        reports, market, calendar, as_of=as_of, horizon=label_horizon
    )
    write_diagnostic(root / "state/qlib-challenger/forward-evaluation.json", payload)
    return payload
