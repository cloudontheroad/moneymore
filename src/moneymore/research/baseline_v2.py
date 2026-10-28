"""Isolated baseline scoring ablations. No recommendation or broker writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..factors import PreprocessConfig, build_default_registry, preprocess_cross_section


def consistent_scores(raw, weights, *, neutralize_size=False, minimum_assets=5):
    """One fixed formula on an explicitly disclosed complete-data cohort.

    Missing observations are NOT zero-filled, median-imputed or reweighted.
    Exclusions are research evidence, not permission to liquidate live holdings.
    """
    if not weights or any(not np.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("INVALID_FACTOR_WEIGHTS")
    names = list(weights)
    frame = raw.copy()
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("DUPLICATE_SCORE_ROWS")
    frame[names] = (
        frame[names]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = frame[names].notna().all(axis=1)
    if neutralize_size:
        if "total_mv" not in frame:
            raise ValueError("MISSING_SIZE_CONTROL")
        frame["total_mv"] = pd.to_numeric(frame.total_mv, errors="coerce")
        valid &= np.isfinite(frame.total_mv) & frame.total_mv.gt(0)
    rejected = frame.loc[~valid, ["date", "symbol"]].copy()
    rejected["reason"] = [
        ";".join(
            [name for name in names if pd.isna(frame.loc[i, name])]
            + (
                ["total_mv"]
                if neutralize_size
                and not valid.loc[i]
                and (not np.isfinite(frame.loc[i, "total_mv"]) or frame.loc[i, "total_mv"] <= 0)
                else []
            )
        )
        for i in rejected.index
    ]
    frame = frame.loc[valid].copy()
    if frame.empty or frame.groupby("date").size().min() < minimum_assets:
        raise ValueError("INSUFFICIENT_COMPLETE_CROSS_SECTION")
    if neutralize_size and frame.groupby("date").total_mv.nunique().min() < 2:
        raise ValueError("DEGENERATE_SIZE_CONTROL")
    registry = build_default_registry()
    for name in names:
        if registry.get(name).direction.value == "low_is_better":
            frame[name] = -frame[name]
    processed = preprocess_cross_section(
        frame,
        names,
        PreprocessConfig(
            neutralize_size=neutralize_size, industry_column=None, minimum_assets=minimum_assets
        ),
    )
    total = sum(weights.values())
    processed["active_weight"] = total
    processed["score"] = sum(processed[name] * (weight / total) for name, weight in weights.items())
    for name, weight in weights.items():
        processed[f"effective_weight__{name}"] = weight / total
        processed[f"contribution__{name}"] = processed[name] * weight / total
    processed["rank"] = processed.groupby("date").score.rank(ascending=False, method="first")
    return processed, rejected


def run_baseline_v2_research(root, store, symbols, as_of):
    """Same date/universe/weights; separate data repair and size-control ablations."""
    import hashlib
    from datetime import UTC, datetime
    from uuid import uuid4

    import yaml

    from ..challenger_diagnostics import trading_dates, write_diagnostic
    from ..data.fundamental_research import (
        financial_features_as_of,
        financial_quality,
        research_financial_frame,
    )
    from .sector_model import build_global_factor_snapshot

    pd.to_datetime(as_of, format="%Y%m%d", errors="raise")
    policy = yaml.safe_load((root / "configs/sector_models.yaml").read_text(encoding="utf-8"))[
        "global_selection"
    ]
    weights = policy["factors"]
    staged = research_financial_frame(store)
    financial = financial_features_as_of(staged, as_of)
    quality = financial_quality(staged, symbols, as_of, per_field=True)
    excluded = set(quality["missing_symbols"] + quality["stale_symbols"])
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    directory = root / "state/baseline-v2" / as_of / run_id
    directory.mkdir(parents=True, exist_ok=False)

    class SnapshotStore:
        def __init__(self, repaired):
            self.repaired = repaired

        def read(self, table, **kwargs):
            if table == "fina_indicator" and self.repaired:
                return financial[kwargs["columns"]].copy()
            frame = store.read(table, **kwargs)
            column = "ann_date" if table == "fina_indicator" else "trade_date"
            if column in frame:
                frame = frame.loc[frame[column].astype(str).le(as_of)]
            return frame

    legacy = build_global_factor_snapshot(SnapshotStore(False), symbols, weights)
    capture = {}
    repaired_legacy = build_global_factor_snapshot(
        SnapshotStore(True), symbols, weights, audit_capture=capture
    )
    raw = capture["raw_factors"]
    if raw.empty or pd.Timestamp(raw.date.max()).strftime("%Y%m%d") != as_of:
        raise ValueError("MARKET_DATE_MISMATCH")
    raw_symbols = set(raw.symbol)
    stale_market = sorted(set(symbols) - raw_symbols)
    raw = raw.loc[~raw.symbol.isin(excluded)].copy()
    calendar = [day for day in trading_dates(store, as_of) if day <= as_of][-253:]
    if len(calendar) != 253 or calendar[-1] != as_of:
        raise ValueError("INCOMPLETE_EXCHANGE_CALENDAR")
    filters = [
        ("ts_code", "in", sorted(set(symbols))),
        ("trade_date", ">=", calendar[0]),
        ("trade_date", "<=", as_of),
    ]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    history = daily.merge(adj, on=["ts_code", "trade_date"], validate="one_to_one")
    price = pd.to_numeric(history.close, errors="coerce")
    adjustment = pd.to_numeric(history.adj_factor, errors="coerce")
    valid_history = history.loc[
        history.trade_date.astype(str).isin(calendar)
        & np.isfinite(price)
        & price.gt(0)
        & np.isfinite(adjustment)
        & adjustment.gt(0)
    ]
    counts = valid_history.groupby("ts_code").trade_date.nunique()
    continuous_symbols = set(counts.loc[counts.eq(253)].index)
    history_excluded = sorted(set(raw.symbol) - continuous_symbols)
    raw = raw.loc[raw.symbol.isin(continuous_symbols)].copy()
    basic = store.read(
        "daily_basic",
        columns=["ts_code", "trade_date", "total_mv", "pe_ttm", "dv_ttm"],
        filters=[("ts_code", "in", sorted(set(symbols)))],
    )
    basic = basic.loc[basic.trade_date.astype(str).lt(as_of)].copy()
    basic = basic.sort_values(["ts_code", "trade_date"]).groupby("ts_code").tail(1)
    # Explicit conservative age guard; never use future or indefinitely carried metadata.
    age = (pd.Timestamp(as_of) - pd.to_datetime(basic.trade_date, format="%Y%m%d")).dt.days
    basic = basic.loc[age.between(0, 10)].set_index("ts_code")
    raw["total_mv"] = raw.symbol.map(basic.total_mv)
    # Shared cohort so the size ablation cannot gain merely by changing its stock pool.
    size_valid = np.isfinite(pd.to_numeric(raw.total_mv, errors="coerce")) & raw.total_mv.gt(0)
    size_excluded = raw.loc[~size_valid, "symbol"].tolist()
    common = raw.loc[size_valid].copy()
    plain, rejected = consistent_scores(common, weights)
    controlled, _ = consistent_scores(common, weights, neutralize_size=True)
    legacy.to_parquet(directory / "legacy.parquet", index=False)
    repaired_legacy.to_parquet(directory / "repaired_data_legacy_formula.parquet", index=False)
    raw.to_parquet(directory / "raw.parquet", index=False)
    financial.to_parquet(directory / "financial_provenance.parquet", index=False)
    rejected.to_parquet(directory / "factor_exclusions.parquet", index=False)
    result = {
        "as_of_date": as_of,
        "version": "baseline_v2_scoring_contract_v1",
        "status": "RESEARCH_ONLY",
        "orders_enabled": False,
        "automatic_promotion": False,
        "weights": weights,
        "missing_policy": "EXPLICIT_COMPLETE_COHORT_NO_REWEIGHT",
        "financial_quality": quality,
        "financial_exclusions": sorted(excluded),
        "size_exclusions": size_excluded,
        "factor_exclusions": rejected.to_dict("records"),
        "requested": len(set(symbols)),
        "scored": len(plain),
        "stale_market_exclusions": stale_market,
        "history_exclusions": history_excluded,
        "promotion_gate": "BLOCKED_PENDING_SEMANTICS_AND_FORWARD_VALIDATION",
        "evidence_path": str(directory),
        "input_semantics": {
            "nonpositive_pe_excluded_by_legacy_factor": int(
                pd.to_numeric(basic.pe_ttm, errors="coerce").le(0).sum()
            ),
            "missing_pe_requires_evidence_not_loss_assumption": int(basic.pe_ttm.isna().sum()),
            "missing_dividend_yield_not_assumed_zero": int(basic.dv_ttm.isna().sum()),
            "earnings_yield_rule": "LEGACY_POSITIVE_PE_ONLY_UNCHANGED_FOR_ABLATION",
            "history_rule": "253_VALID_CONSECUTIVE_EXCHANGE_SESSIONS_NO_FILL",
        },
        "industry_constraints": False,
        "data_only_ablation": {
            "legacy_scored": len(legacy),
            "repaired_data_legacy_formula_scored": len(repaired_legacy),
            "scope": "SAME_LEGACY_FORMULA_AND_UNIVERSE_CHANGE_FINANCIAL_SOURCE_ONLY",
        },
        "variants": {},
        "limitations": [
            "CURRENT_UNIVERSE_SNAPSHOT_NOT_HISTORICAL_PERFORMANCE",
            "EXCLUSIONS_NOT_LIVE_LIQUIDATION_INSTRUCTIONS",
            "RANK_NORMALIZATION_CAN_REINTRODUCE_SIZE_EXPOSURE",
        ],
    }
    for name, scores in (("repaired_fixed_weights", plain), ("repaired_size_control", controlled)):
        scores.to_parquet(directory / (name + ".parquet"), index=False)
        result["variants"][name] = {
            "top10": scores.sort_values("rank")
            .head(10)[["symbol", "score", "rank"]]
            .to_dict("records"),
            "score_log_size_spearman": float(
                scores.score.corr(np.log(scores.total_mv), method="spearman")
            ),
        }
    # Encode dates in diagnostic exclusions without allowing NaN JSON payloads.
    result["factor_exclusions"] = [
        {**r, "date": str(r["date"])} for r in result["factor_exclusions"]
    ]
    result["artifacts"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.glob("*.parquet")
    }
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(root / "state/baseline-v2/latest.json", result)
    return result
