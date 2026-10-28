"""Frozen daily ranking diagnostic. No fitting, forward labels, targets or trading."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from ..challenger_diagnostics import trading_dates, write_diagnostic
from ..data.fundamental_research import research_financial_frame
from .baseline_v2_financials import common_period_features, required_period
from .baseline_v2_industry import industry_as_of
from .baseline_v2_model import comparable_scores
from .security_identity import SecurityIdentity


def logical_hash(frame):
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()


def score_pair(raw, policy, weight_report):
    names = list(policy["factors"])
    fixed = weight_report["fixed_weights"]
    coefficients = weight_report["diagnostic_coefficients_not_released"]
    if policy.get("validated_linear_model") or policy["factors"] != fixed:
        raise ValueError("FROZEN_FIXED_WEIGHT_POLICY_MISMATCH")
    if (
        set(coefficients) != set(names)
        or not np.isfinite(list(coefficients.values())).all()
        or sum(abs(float(c)) for c in coefficients.values()) <= 1e-12
    ):
        raise ValueError("INVALID_FROZEN_RESEARCH_COEFFICIENTS")
    out = comparable_scores(raw, policy, size_control=True, industry_control=True)
    out["score_fixed"] = out.score
    # Features are already direction-adjusted and normalized exactly once by the shared scorer.
    # Omit the unavailable constant intercept: it does not affect a cross-sectional ranking.
    out["score_learned"] = out[names].to_numpy() @ np.array([coefficients[n] for n in names])
    for variant in ("fixed", "learned"):
        order = out.sort_values(["score_" + variant, "symbol"], ascending=[False, True]).index
        out.loc[order, "rank_" + variant] = np.arange(1, len(out) + 1)
    return out


def run_daily_scores(root, store):
    base = root / "state/baseline-v2-validation/daily-inputs"
    audit_path = base / "identity-coverage-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("signals") != 463 or not audit.get("identity_policy"):
        raise ValueError("COMPLETE_IDENTITY_COVERAGE_AUDIT_REQUIRED")
    identity = SecurityIdentity(audit["identity_policy"])
    identity_report = json.loads((base / "identity-audit.json").read_text(encoding="utf-8"))
    if identity_report.get("blocking_scope_conflicts") != 0:
        raise ValueError("IDENTITY_SOURCE_CONFLICTS_BLOCK_SCORING")
    pool_path = Path(identity_report["output_directory"]) / "identity-top1000-preview.parquet"
    pool = pd.read_parquet(pool_path)
    if len(pool) != 463000 or pool.duplicated(["signal_date", "security_id"]).any():
        raise ValueError("DAILY_POOL_IDENTITY_DUPLICATE_OR_INCOMPLETE")
    expected_path = Path(audit["output_directory"]) / "sample-exclusions.parquet"
    if hashlib.sha256(expected_path.read_bytes()).hexdigest() != audit["details_sha256"]:
        raise ValueError("COVERAGE_DETAILS_HASH_MISMATCH")
    expected = pd.read_parquet(expected_path)
    policy_path = root / "configs/baseline_v2.yaml"
    weights_path = root / "state/baseline-v2-validation/weight-validation.json"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    calendar = trading_dates(store, "20261028")
    signals = sorted(pool.signal_date.unique())
    start = audit["identity_scope_start"]
    dates = [d for d in calendar if start <= d <= signals[-1]]
    symbols = sorted(pool.security_id.unique())
    source_symbols = set(symbols)
    for row in identity.changes:
        if row["old"] in source_symbols:
            source_symbols.add(row["new"])
    filters = [
        ("ts_code", "in", sorted(source_symbols)),
        ("trade_date", ">=", start),
        ("trade_date", "<=", signals[-1]),
    ]
    raw = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    raw = identity.market_rows(raw, "trade_date", ["close"])
    adj = identity.market_rows(adj, "trade_date", ["adj_factor"])
    close = raw.pivot(index="trade_date", columns="ts_code", values="close").reindex(
        index=dates, columns=symbols
    )
    factors = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex(
        index=dates, columns=symbols
    )
    finances = identity.market_rows(
        research_financial_frame(store),
        "ann_date",
        ["roe", "q_sales_yoy"],
        record_keys=["ann_date", "end_date", "growth_source"],
    )
    industry = identity.industry_rows(store.read("baseline_v2_industry_membership"))
    inputs = {
        "raw_close_matrix": close,
        "source_adjustment_matrix": factors,
        "financial_records": finances,
        "industry_intervals": industry,
    }
    for name, frame in inputs.items():
        if logical_hash(frame) != audit["source_logical_hashes"][name]:
            raise ValueError("FROZEN_INPUT_CHANGED:" + name)
    prices = (close * factors).where(
        np.isfinite(close) & close.gt(0) & np.isfinite(factors) & factors.gt(0)
    )
    returns = prices.pct_change(fill_method=None)
    core = {
        "momentum_252_21": prices.shift(21) / prices.shift(252) - 1,
        "return_120": prices / prices.shift(120) - 1,
        "volatility_60": returns.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(242),
        "drawdown_120": prices / prices.rolling(120, min_periods=120).max() - 1,
    }
    continuous = prices.notna().rolling(253, min_periods=253).sum().eq(253)
    directory = (
        base
        / "daily-score-runs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    directory.mkdir(parents=True)
    frozen = {
        "policy": policy,
        "weight_report": weights,
        "coverage_audit": audit,
        "identity_audit": identity_report,
        "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "weight_report_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }
    write_diagnostic(directory / "frozen-inputs.json", frozen)
    frames, diagnostics = [], []
    for count, signal in enumerate(signals, 1):
        raw = pool.loc[pool.signal_date.eq(signal)].rename(columns={"ts_code": "symbol"}).copy()
        if len(raw) != 1000 or not raw.trade_date.eq(calendar[calendar.index(signal) - 1]).all():
            raise ValueError("INVALID_EXACT_PRIOR_SESSION_POOL")
        raw["date"] = pd.Timestamp(signal)
        raw["earnings_yield"] = 1 / pd.to_numeric(raw.pe_ttm, errors="coerce").where(
            raw.pe_ttm.gt(0)
        )
        raw["book_to_price"] = 1 / pd.to_numeric(raw.pb, errors="coerce").replace(0, np.nan)
        raw["dividend_yield_ttm"] = pd.to_numeric(raw.dv_ttm, errors="coerce") / 100
        for field, matrix in core.items():
            raw[field] = raw.symbol.map(matrix.loc[signal])
        finance = common_period_features(finances, signal).set_index("ts_code")
        for field, name in [("roe", "roe"), ("q_sales_yoy", "quarterly_revenue_growth")]:
            valid = finance[field + "_end_date"].astype(str).eq(required_period(signal))
            raw[name] = raw.symbol.map(finance[field].where(valid))
        raw["industry"] = raw.symbol.map(industry_as_of(industry, signal).l1_name)
        eligible = raw[policy["required_factors"]].notna().all(axis=1)
        eligible &= raw.symbol.map(continuous.loc[signal]).fillna(False) & raw.industry.notna()
        wanted = set(
            expected.loc[expected.date.eq(signal) & expected.reason.eq("ELIGIBLE"), "symbol"]
        )
        if set(raw.loc[eligible, "symbol"]) != wanted:
            raise ValueError("ELIGIBILITY_DOES_NOT_RECONCILE:" + signal)
        scored = score_pair(raw.loc[eligible], policy, weights)
        columns = [
            "date",
            "symbol",
            "industry",
            "total_mv",
            "observed_weight",
            *policy["factors"],
            *["missing__" + f for f in policy["factors"]],
            "score_fixed",
            "score_learned",
            "rank_fixed",
            "rank_learned",
        ]
        frames.append(scored[columns])
        fixed = set(scored.loc[scored.rank_fixed.le(10), "symbol"])
        learned = set(scored.loc[scored.rank_learned.le(10), "symbol"])
        entry = {
            "date": signal,
            "eligible": len(scored),
            "diagnostic_top10_overlap": len(fixed & learned),
            "score_rank_correlation": scored.score_fixed.corr(
                scored.score_learned, method="spearman"
            ),
        }
        for variant in ["fixed", "learned"]:
            top = scored.loc[scored["rank_" + variant].le(10)]
            entry[variant + "_growth_board_fraction"] = float(
                top.symbol.str.startswith(("30", "68")).mean()
            )
        diagnostics.append(entry)
        if count % 50 == 0:
            print(
                json.dumps(
                    {"status": "SCORING", "completed_days": count, "required_days": len(signals)}
                ),
                flush=True,
            )
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(directory / "daily-scores.parquet", index=False)
    pd.DataFrame(diagnostics).to_parquet(directory / "daily-diagnostics.parquet", index=False)
    report = {
        "status": "DAILY_RANKING_DIAGNOSTIC_NOT_RELEASED",
        "output_directory": str(directory),
        "days": len(signals),
        "rows": len(panel),
        "minimum_daily_eligible": min(d["eligible"] for d in diagnostics),
        "mean_diagnostic_top10_overlap": float(
            np.mean([d["diagnostic_top10_overlap"] for d in diagnostics])
        ),
        "fixed_mean_diagnostic_growth_board_fraction": float(
            np.mean([d["fixed_growth_board_fraction"] for d in diagnostics])
        ),
        "learned_mean_diagnostic_growth_board_fraction": float(
            np.mean([d["learned_growth_board_fraction"] for d in diagnostics])
        ),
        "panel_sha256": hashlib.sha256(
            (directory / "daily-scores.parquet").read_bytes()
        ).hexdigest(),
        "real_orders_enabled": False,
        "production_changed": False,
        "limitations": [
            "RESEARCH_COEFFICIENTS_NOT_RELEASE_APPROVED",
            "TOP10_DIAGNOSTIC_NOT_ACTUAL_PORTFOLIO",
            "RANKING_SCORE_NOT_CALIBRATED_RETURN_FORECAST",
            "NO_DAILY_EXECUTION_OR_PERFORMANCE_RESULT",
            "SOURCE_ADJUSTMENT_ECONOMIC_AUDIT_NOT_COMPLETED_FOR_FULL_DAILY_UNION",
            "ORIGINAL_FINANCIAL_VINTAGES_UNPROVEN",
        ],
    }
    write_diagnostic(directory / "report.json", report)
    write_diagnostic(base / "daily-scores.json", report)
    return report
