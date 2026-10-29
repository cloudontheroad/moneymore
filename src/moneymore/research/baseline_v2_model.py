"""Versioned factor baseline: comparable signals, explicit normalized-score imputation.

No raw financial zero filling, stock-board quotas or learned performance claims.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..challenger_diagnostics import trading_dates, write_diagnostic
from ..data.fundamental_research import (
    financial_features_as_of,
    financial_quality,
    research_financial_frame,
)
from ..factors import PreprocessConfig, build_default_registry, preprocess_cross_section
from ..strategy_universe import active_strategy_universe
from .baseline_v2_financials import common_period_features
from .baseline_v2_industry import industry_as_of
from .sector_model import build_global_factor_snapshot


def comparable_scores(raw, policy, *, size_control=True, industry_control=True):
    """QLib Fillna(0) applies ONLY after cross-sectional normalization.

    Optional missing valuation is neutral signal with explicit missing indicators,
    NOT evidence of zero dividend/earnings. Core quality and price cannot be imputed.
    Industry dummies remove incomparable factor level effects, not enforce budgets.
    """
    from qlib.data.dataset.processor import Fillna

    weights = policy["factors"]
    names = list(weights)
    if (
        not weights
        or any(not np.isfinite(w) or w <= 0 for w in weights.values())
        or not set(policy["required_factors"]).issubset(weights)
    ):
        raise ValueError("INVALID_BASELINE_V2_POLICY")
    frame = raw.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.empty or frame.duplicated(["date", "symbol"]).any():
        raise ValueError("EMPTY_OR_DUPLICATE_V2_CROSS_SECTION")
    frame[names] = (
        frame[names]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    if not frame[policy["required_factors"]].notna().all().all():
        raise ValueError("MISSING_CORE_FACTOR_NO_IMPUTATION")
    if len(frame) < int(policy["minimum_cross_section_assets"]):
        raise ValueError("INSUFFICIENT_V2_CROSS_SECTION")
    if size_control:
        size = pd.to_numeric(frame.total_mv, errors="coerce")
        if not (np.isfinite(size) & size.gt(0)).all() or size.nunique() < 2:
            raise ValueError("INVALID_SIZE_CONTROL")
    if industry_control:
        if frame.industry.isna().any() or frame.industry.astype(str).str.strip().eq("").any():
            raise ValueError("MISSING_INDUSTRY_CONTROL")
        counts = frame.industry.map(frame.industry.value_counts())
        frame["control_industry"] = frame.industry.where(
            counts.ge(int(policy["minimum_industry_assets"])), "POOLED_SMALL_INDUSTRIES"
        )
    missing = frame[names].isna()
    registry = build_default_registry()
    for name in names:
        if registry.get(name).direction.value == "low_is_better":
            frame[name] = -frame[name]
    control = "control_industry" if industry_control else None
    if control:
        parameter_count = frame[control].nunique() + int(size_control)
        # Every factor's OLS must be estimable; never silently fall back to unadjusted data.
        if any(frame[name].notna().sum() < max(5, parameter_count + 2) for name in names):
            raise ValueError("FACTOR_CONTROL_SAMPLE_INSUFFICIENT")
    processed = preprocess_cross_section(
        frame,
        names,
        PreprocessConfig(neutralize_size=size_control, industry_column=control, minimum_assets=5),
    )
    normalized_missing = processed[names].isna()
    processed[names] = Fillna(fill_value=0)(processed[names].copy())
    total = sum(weights.values())
    processed["score"] = 0.0
    processed["active_weight"] = total
    processed["observed_weight"] = sum((~missing[name]) * w for name, w in weights.items()) / total
    for name, weight in weights.items():
        processed[f"missing__{name}"] = missing[name].to_numpy()
        processed[f"neutral_signal__{name}"] = normalized_missing[name].to_numpy()
        processed[f"effective_weight__{name}"] = weight / total
        processed[f"contribution__{name}"] = processed[name] * weight / total
        processed["score"] += processed[f"contribution__{name}"]
    if policy.get("validated_linear_model") and policy.get("experimental_linear_model"):
        raise ValueError("AMBIGUOUS_LINEAR_MODEL_RELEASE_SCOPE")
    learned = policy.get("validated_linear_model") or policy.get("experimental_linear_model")
    if learned:
        coefficients = learned.get("coefficients", {})
        if (
            learned.get("native_model") != "qlib.contrib.model.linear.LinearModel"
            or set(coefficients) != set(names)
            or not (learned.get("validation_evidence_sha256") or (
                learned.get("scope") == "FORWARD_STYLE_REPAIR_EXPERIMENT_NOT_PROMOTION"
                and learned.get("evidence_sha256")
                and policy.get("real_orders_enabled") is False
                and policy.get("automatic_promotion") is False
            ))
            or not np.isfinite(list(coefficients.values())).all()
            or sum(abs(float(c)) for c in coefficients.values()) <= 1e-12
        ):
            raise ValueError("INVALID_VALIDATED_LINEAR_MODEL")
        intercept = float(learned.get("intercept", 0))
        if not np.isfinite(intercept):
            raise ValueError("INVALID_LINEAR_INTERCEPT")
        processed["score"] = intercept
        processed["intercept_contribution"] = intercept
        norm = sum(abs(float(c)) for c in coefficients.values())
        processed["observed_weight"] = (
            sum((~missing[f]) * abs(float(c)) for f, c in coefficients.items()) / norm
        )
        for name, coefficient in coefficients.items():
            processed[f"effective_weight__{name}"] = float(coefficient)
            processed[f"contribution__{name}"] = processed[name] * float(coefficient)
            processed["score"] += processed[f"contribution__{name}"]
    processed["rank"] = processed.score.rank(ascending=False, method="first")
    return processed


def require_weight_release_evidence(report):
    """No flag named 'fit succeeded' may be confused with permission to release."""
    if (
        report.get("status") != "PASS_RESEARCH_ONLY"
        or report.get("gate_failures")
        or report.get("real_orders_enabled") is not False
        or report.get("release_gate") != "CONTINUOUS_PIT_COST_AND_STYLE_VALIDATED"
    ):
        raise ValueError("WEIGHT_VALIDATION_NOT_RELEASE_APPROVED")


def build_v2_snapshot(root: Path, store, as_of: str, policy=None):
    """Return versioned data/control ablations, exclusions and source provenance."""
    policy = policy or yaml.safe_load(
        (root / "configs/baseline_v2.yaml").read_text(encoding="utf-8")
    )
    if policy.get("experimental_linear_model"):
        from .security_identity import SecurityIdentity
        from .style_v2_experiment import OverlayStore, validate_experiment

        validate_experiment(root, policy)
        overlay = policy["input_overlay"]
        store = OverlayStore(store, overlay["path"], overlay["sha256"], SecurityIdentity(overlay["identity_policy"]))
    if policy.get("validated_linear_model"):
        learned = policy["validated_linear_model"]
        evidence_path = Path(learned.get("validation_evidence_path", "")).resolve()
        if not evidence_path.is_relative_to((root / "state/baseline-v2-validation").resolve()) or not evidence_path.is_file():
            raise ValueError("INVALID_WEIGHT_RELEASE_EVIDENCE_PATH")
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != learned.get("validation_evidence_sha256"):
            raise ValueError("WEIGHT_RELEASE_EVIDENCE_HASH_MISMATCH")
        require_weight_release_evidence(json.loads(evidence_path.read_text(encoding="utf-8")))
    universe = active_strategy_universe(store)
    if universe.effective_date.astype(str).max() > as_of:
        raise ValueError("FUTURE_UNIVERSE_NOT_ALLOWED")
    symbols = sorted(set(universe.symbol))
    staged = research_financial_frame(store)
    if policy.get("financial_period_rule") == "COMMON_DISCLOSED_QUARTER":
        financial = common_period_features(staged, as_of)
        good = set(financial.loc[financial[["roe", "q_sales_yoy"]].notna().all(axis=1), "ts_code"])
        financial_excluded = set(symbols) - good
    else:
        financial = financial_features_as_of(staged, as_of)
        quality = financial_quality(staged, symbols, as_of, per_field=True)
        financial_excluded = set(quality["missing_symbols"] + quality["stale_symbols"])

    class PITStore:
        def read(self, table, **kwargs):
            if table == "fina_indicator":
                return financial[kwargs["columns"]].copy()
            frame = store.read(table, **kwargs)
            if table == "daily_basic":
                try:
                    official = store.read("baseline_v2_daily_basic")
                except FileNotFoundError:
                    official = pd.DataFrame()
                if not official.empty:
                    columns = kwargs.get("columns") or list(frame)
                    extra = official.loc[
                        official.ts_code.isin(symbols) & official.trade_date.astype(str).le(as_of)
                    ]
                    frame = pd.concat(
                        [frame, extra.reindex(columns=columns)], ignore_index=True
                    ).drop_duplicates(["ts_code", "trade_date"], keep="last")
            if "trade_date" in frame:
                return frame.loc[frame.trade_date.astype(str).le(as_of)]
            return frame

    capture = {}
    legacy_repaired = build_global_factor_snapshot(
        PITStore(), symbols, policy["factors"], audit_capture=capture
    )
    raw = capture["raw_factors"].copy()
    if raw.empty or pd.Timestamp(raw.date.max()).strftime("%Y%m%d") != as_of:
        raise ValueError("V2_MARKET_DATE_MISMATCH")
    sessions = int(policy["history_sessions"])
    calendar = [d for d in trading_dates(store, as_of) if d <= as_of][-sessions:]
    if len(calendar) != sessions or calendar[-1] != as_of:
        raise ValueError("V2_INCOMPLETE_CALENDAR")
    filters = [
        ("ts_code", "in", symbols),
        ("trade_date", ">=", calendar[0]),
        ("trade_date", "<=", as_of),
    ]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    joined = daily.merge(adj, on=["ts_code", "trade_date"], validate="one_to_one")
    joined["price"] = pd.to_numeric(joined.close, errors="coerce") * pd.to_numeric(
        joined.adj_factor, errors="coerce"
    )
    valid = joined.loc[
        np.isfinite(joined.price)
        & joined.price.gt(0)
        & joined.trade_date.astype(str).isin(calendar)
    ]
    prices = valid.pivot(index="trade_date", columns="ts_code", values="price").reindex(calendar)
    continuous = set(prices.columns[prices.notna().all()])
    basic = PITStore().read(
        "daily_basic",
        columns=["ts_code", "trade_date", "total_mv", "pe_ttm", "dv_ttm"],
        filters=[("ts_code", "in", symbols), ("trade_date", "<", as_of)],
    )
    basic = basic.loc[basic.trade_date.astype(str).lt(as_of)]
    basic = (
        basic.sort_values(["ts_code", "trade_date"]).groupby("ts_code").tail(1).set_index("ts_code")
    )
    age = (pd.Timestamp(as_of) - pd.to_datetime(basic.trade_date, format="%Y%m%d")).dt.days
    basic = basic.loc[age.between(0, int(policy["market_metadata_age_days"]))]
    metadata = universe.set_index("symbol")
    raw["total_mv"] = raw.symbol.map(basic.total_mv)
    raw["industry"] = raw.symbol.map(metadata.industry)
    if policy.get("industry_control_source") == "SW_L1_EFFECTIVE_INTERVALS":
        mapping = industry_as_of(store.read("baseline_v2_industry_membership"), as_of)
        raw["industry"] = raw.symbol.map(mapping.l1_name)
    raw["name"] = raw.symbol.map(metadata.name)
    raw["board"] = raw.symbol.map(metadata.board)
    reasons = {}
    for symbol in symbols:
        row = raw.loc[raw.symbol.eq(symbol)]
        why = []
        if symbol in financial_excluded:
            why.append("MISSING_OR_STALE_FINANCIALS")
        if symbol not in continuous:
            why.append("INCOMPLETE_253_SESSION_HISTORY")
        if row.empty:
            why.append("STALE_MARKET_CROSS_SECTION")
        else:
            if not row[policy["required_factors"]].notna().all().all():
                why.append("MISSING_CORE_FACTOR")
            if not np.isfinite(row.total_mv).all() or not row.total_mv.gt(0).all():
                why.append("MISSING_OR_STALE_MARKET_METADATA")
            if row.industry.isna().any():
                why.append("MISSING_INDUSTRY")
        if why:
            reasons[symbol] = why
    eligible = raw.loc[~raw.symbol.isin(reasons)].reset_index(drop=True)
    variants = {
        "fixed_weight_neutral_missing": comparable_scores(
            eligible, policy, size_control=False, industry_control=False
        ),
        "size_comparable": comparable_scores(
            eligible, policy, size_control=True, industry_control=False
        ),
        "industry_size_comparable": comparable_scores(eligible, policy),
    }
    return (
        policy,
        variants,
        {
            "raw": raw,
            "financial_provenance": financial,
            "repaired_legacy": legacy_repaired,
            "prices": prices,
        },
        reasons,
    )


def refresh_official_basic(root, store, date):
    """One bounded official daily snapshot; isolated table, no production overwrite."""
    from datetime import UTC, datetime

    from dotenv import load_dotenv

    from ..data.tushare_provider import TushareProvider

    try:
        existing = store.read("baseline_v2_daily_basic", filters=[("trade_date", "=", date)])
    except FileNotFoundError:
        existing = pd.DataFrame()
    if not existing.empty:
        return {"status": "CACHED", "rows": len(existing), "date": date}
    load_dotenv(root / ".env", override=False)
    # Uses the adapter's authenticated client without printing or persisting its credential.
    data = TushareProvider()._client.daily_basic(
        trade_date=date, fields="ts_code,trade_date,close,pb,pe_ttm,dv_ttm,total_mv"
    )
    required = {"ts_code", "trade_date", "close", "pb", "pe_ttm", "dv_ttm", "total_mv"}
    if data is None or data.empty or not required.issubset(data) or len(data) >= 6000:
        raise ValueError("INVALID_OR_POSSIBLY_TRUNCATED_OFFICIAL_BASIC")
    if not data.trade_date.eq(date).all() or data.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("OFFICIAL_BASIC_DATE_OR_DUPLICATE_ERROR")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    store.save_snapshot(
        "baseline_v2_daily_basic", data, "tushare", ["ts_code", "trade_date"], date + "_" + stamp
    )
    return {"status": "FETCHED", "rows": len(data), "date": date}


def archive_v2(root, as_of, policy, variants, capture, exclusions):
    from datetime import UTC, datetime
    from uuid import uuid4

    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    directory = root / "state/baseline-v2-model" / as_of / run
    directory.mkdir(parents=True, exist_ok=False)
    # Price matrix is archived with its date index, no synthetic future observations.
    for name, frame in {**capture, **variants}.items():
        frame.to_parquet(directory / (name + ".parquet"), index=name == "prices")
    report = {
        "version": policy["version"],
        "as_of_date": as_of,
        "policy": policy,
        "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        "status": "FORWARD_PAPER_EXPERIMENT",
        "real_orders_enabled": False,
        "automatic_promotion": False,
        "excluded": exclusions,
        "eligible": len(next(iter(variants.values()))),
        "path": str(directory),
        "variants": {},
        "limitations": [
            "FROZEN_LEARNED_RESEARCH_NOT_PROMOTION_VALIDATED" if policy.get("experimental_linear_model") else "NOT_TRAINED_OR_RETURN_OPTIMAL",
            "CURRENT_UNIVERSE_NOT_HISTORICAL_BACKTEST",
            "NEUTRAL_IMPUTATION_IS_MODEL_ASSUMPTION_NOT_RAW_ZERO",
            "NO_BOARD_ALLOCATION_TARGET",
        ],
    }
    for name, frame in variants.items():
        top = frame.sort_values("rank").head(int(policy["portfolio"]["top_k"]))
        tech = frame.symbol.str.startswith(("688", "689", "300", "301"))
        contribution_columns = ["contribution__" + f for f in policy["factors"]]
        report["variants"][name] = {
            "top": top[["symbol", "name", "industry", "board", "score", "observed_weight"]].to_dict(
                "records"
            ),
            "top_star_chinext_count": int(
                top.symbol.str.startswith(("688", "689", "300", "301")).sum()
            ),
            "score_size_correlation": float(
                frame.score.corr(np.log(frame.total_mv), method="spearman")
            ),
            "missing_counts": {f: int(frame[f"missing__{f}"].sum()) for f in policy["factors"]},
            "star_chinext_eligible": int(tech.sum()),
            "best_star_chinext_rank": float(frame.loc[tech, "rank"].min()) if tech.any() else None,
            "star_chinext_top10": frame.loc[tech]
            .sort_values("rank")
            .head(10)[["symbol", "name", "rank", "score", "observed_weight", *contribution_columns]]
            .to_dict("records"),
            "star_chinext_mean_factor_contributions": frame.loc[tech, contribution_columns]
            .mean()
            .to_dict(),
            "other_board_mean_factor_contributions": frame.loc[~tech, contribution_columns]
            .mean()
            .to_dict(),
        }
    report["artifacts"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.parquet")
    }
    report["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_diagnostic(directory / "report.json", report)
    write_diagnostic(root / "state/baseline-v2-model/latest.json", report)
    return report
