"""PIT-universe, purged Qlib linear-weight research. Fail closed on evidence gaps.

Only research artifacts are written. Never edits a running model, ledger or NAV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..challenger_diagnostics import trading_dates, write_diagnostic
from ..data.fundamental_research import research_financial_frame
from ..qlib_challenger import QlibPanelDataset
from .baseline_v2_financials import common_period_features, required_period
from .baseline_v2_industry import industry_as_of
from .baseline_v2_model import comparable_scores


def build_monthly_validation_panel(root, store, protocol, schedule):
    directory = root / "state/baseline-v2-validation"
    directory.mkdir(parents=True, exist_ok=True)
    model_policy = yaml.safe_load((root / "configs/baseline_v2.yaml").read_text(encoding="utf-8"))
    basic = store.read("baseline_v2_daily_basic")
    dates = set(basic.trade_date)
    missing = [d for _, d in schedule if d not in dates]
    if missing:
        raise ValueError("MISSING_MONTHLY_OFFICIAL_SNAPSHOTS:" + str(len(missing)))
    snapshots = {}
    for signal, source in schedule:
        frame = basic.loc[basic.trade_date.eq(source)].copy()
        codes = frame.ts_code.astype(str)
        frame = (
            frame.loc[
                (
                    (codes.str.endswith(".SH") & codes.str.startswith(("60", "68")))
                    | (codes.str.endswith(".SZ") & codes.str.startswith(("00", "30")))
                )
                & pd.to_numeric(frame.total_mv, errors="coerce").gt(0)
            ]
            .sort_values(["total_mv", "ts_code"], ascending=[False, True])
            .head(1000)
        )
        if len(frame) != 1000 or frame.ts_code.duplicated().any():
            raise ValueError("INVALID_HISTORICAL_TOP1000:" + signal)
        snapshots[signal] = frame
    symbols = sorted(set().union(*(set(f.ts_code) for f in snapshots.values())))
    calendar = [d for d in trading_dates(store, protocol["as_of"]) if d <= protocol["as_of"]]
    first_idx = calendar.index(schedule[0][0])
    calendar = calendar[max(0, first_idx - 270) :]
    filters = [
        ("ts_code", "in", symbols),
        ("trade_date", ">=", calendar[0]),
        ("trade_date", "<=", protocol["as_of"]),
    ]
    daily = store.read("daily", columns=["ts_code", "trade_date", "open", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    joined = daily.merge(adj, on=["ts_code", "trade_date"], validate="one_to_one")
    for col in ("open", "close"):
        joined[col] = pd.to_numeric(joined[col], errors="coerce") * pd.to_numeric(
            joined.adj_factor, errors="coerce"
        )
    prices = joined.pivot(index="trade_date", columns="ts_code", values="close").reindex(calendar)
    opens = joined.pivot(index="trade_date", columns="ts_code", values="open").reindex(calendar)
    prices = prices.where(np.isfinite(prices) & prices.gt(0))
    opens = opens.where(np.isfinite(opens) & opens.gt(0))
    del joined, daily, adj
    returns = prices.pct_change(fill_method=None)
    core = {
        "momentum_252_21": prices.shift(21) / prices.shift(252) - 1,
        "return_120": prices / prices.shift(120) - 1,
        "volatility_60": returns.rolling(60, min_periods=60).std(ddof=1) * np.sqrt(242),
        "drawdown_120": prices / prices.rolling(120, min_periods=120).max() - 1,
    }
    continuous = prices.notna().rolling(253, min_periods=253).sum().eq(253)
    horizon = int(protocol["label_horizon"])
    future = prices.shift(-horizon) / opens.shift(-1) - 1
    future_valid = (
        prices.notna().rolling(horizon, min_periods=horizon).sum().shift(-horizon).eq(horizon)
    )
    future = future.where(future_valid)
    staged = research_financial_frame(store)
    membership = store.read("baseline_v2_industry_membership")
    if not json.loads(
        (root / "state/baseline-v2-validation/industry-download.json").read_text(encoding="utf-8")
    ).get("complete"):
        raise ValueError("INCOMPLETE_HISTORICAL_INDUSTRY_DOWNLOAD")
    frames, coverage = [], []
    for signal, _ in schedule:
        raw = snapshots[signal].rename(columns={"ts_code": "symbol"}).copy()
        raw["date"] = pd.Timestamp(signal)
        raw["earnings_yield"] = 1 / pd.to_numeric(raw.pe_ttm, errors="coerce").where(
            raw.pe_ttm.gt(0)
        )
        raw["book_to_price"] = 1 / pd.to_numeric(raw.pb, errors="coerce").replace(0, np.nan)
        raw["dividend_yield_ttm"] = pd.to_numeric(raw.dv_ttm, errors="coerce") / 100
        for factor, matrix in core.items():
            raw[factor] = raw.symbol.map(matrix.loc[signal])
        financial = common_period_features(staged, signal).set_index("ts_code")
        for field, name in (("roe", "roe"), ("q_sales_yoy", "quarterly_revenue_growth")):
            valid = financial[field + "_end_date"].astype(str).eq(required_period(signal))
            raw[name] = raw.symbol.map(financial[field].where(valid))
        # Historical intervals, not today's label carried backwards.
        industry = industry_as_of(membership, signal)
        raw["industry"] = raw.symbol.map(industry.l1_name)
        eligible = raw[model_policy["required_factors"]].notna().all(axis=1) & raw.symbol.map(
            continuous.loc[signal]
        ).fillna(False)
        eligible &= raw.industry.notna()
        count = int(eligible.sum())
        coverage.append(
            {
                "date": signal,
                "eligible": count,
                "requested": 1000,
                "coverage": count / 1000,
                "industry_eligible": int(raw.industry.notna().sum()),
                "financial_eligible": int(
                    raw[["roe", "quarterly_revenue_growth"]].notna().all(axis=1).sum()
                ),
            }
        )
        if count < int(protocol["minimum_month_assets"]):
            print(
                json.dumps({"date": signal, "status": "LOW_COVERAGE", "eligible": count}),
                flush=True,
            )
            continue
        scored = comparable_scores(
            raw.loc[eligible], model_policy, size_control=True, industry_control=True
        )
        scored["forward_return"] = scored.symbol.map(future.loc[signal])
        idx = calendar.index(signal) + horizon
        scored["label_end"] = calendar[idx] if idx < len(calendar) else None
        # Do not use October market outcomes to tune this October deployment.
        if str(scored.label_end.iloc[0]) > protocol["end"]:
            scored["forward_return"] = np.nan
        scored["label"] = (scored.forward_return.rank(pct=True) - 0.5) * 2
        frames.append(scored)
        print(json.dumps({"date": signal, "status": "BUILT", "eligible": count}), flush=True)
    if not frames:
        raise ValueError("NO_VALID_TRAINING_MONTHS")
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(directory / "monthly-panel.parquet", index=False)
    pd.DataFrame(coverage).to_parquet(directory / "coverage.parquet", index=False)
    metadata_report = {
        "protocol": protocol,
        "model_policy": model_policy,
        "months": len(frames),
        "rows": len(panel),
        "historical_universe": "OFFICIAL_MONTHLY_PRIOR_SESSION_TOP1000_RECONSTRUCTED",
        "industry_control": True,
        "industry_source": "SW_L1_EFFECTIVE_INTERVALS",
        "current_industry_not_used_in_training": True,
        "financial_vintage": "ANNOUNCEMENT_FILTERED_LATEST_PROVIDER_REVISIONS_NOT_ORIGINAL_VINTAGE_ATTESTED",
        "financial_period_rule": "COMMON_DISCLOSED_QUARTER_NO_MIXED_DURATION_ROE",
        "coverage": coverage,
        "panel_sha256": hashlib.sha256(
            (directory / "monthly-panel.parquet").read_bytes()
        ).hexdigest(),
        "real_orders_enabled": False,
    }
    write_diagnostic(directory / "panel-manifest.json", metadata_report)
    return {k: v for k, v in metadata_report.items() if k not in {"coverage", "model_policy"}}


def purged_split(panel, start, end):
    """Outcome timestamps, not only feature timestamps, must precede split end."""
    return panel.loc[
        panel.date.between(pd.Timestamp(start), pd.Timestamp(end))
        & panel.label_end.astype(str).le(end)
        & panel.label.notna()
    ].copy()


def fit_qlib_linear(frame, names, alpha):
    from qlib.contrib.model.linear import LinearModel

    index = pd.MultiIndex.from_arrays([frame.date, frame.symbol], names=["datetime", "instrument"])
    data = pd.DataFrame(
        frame[names].to_numpy(float),
        index=index,
        columns=pd.MultiIndex.from_product([["feature"], names]),
    )
    data[("label", "target")] = frame.label.to_numpy(float)
    dataset = QlibPanelDataset(
        data, {"train": (str(frame.date.min().date()), str(frame.date.max().date()))}
    )
    model = LinearModel(estimator="ridge", alpha=float(alpha), fit_intercept=True)
    model.fit(dataset)
    return model


def daily_metrics(frame, prediction, *, cost_bps=20):
    """Signal-only independent baskets, not continuous executable account returns."""
    frame = frame.assign(prediction=np.asarray(prediction))
    rows = []
    for date, group in frame.groupby("date"):
        selected = group.nlargest(10, "prediction")
        rows.append(
            {
                "date": str(date.date()),
                "rank_ic": group.prediction.corr(group.label, method="spearman"),
                "basket_excess_proxy": selected.forward_return.mean()
                - group.forward_return.mean()
                - cost_bps / 10000,
            }
        )
    return pd.DataFrame(rows)


def block_lower_bound(values, samples=2000, seed=1729, comparisons=3):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 6:
        return None
    rng = np.random.default_rng(seed)
    # Circular three-month blocks preserve local dependence; multiple-grid correction.
    starts = rng.integers(0, len(values), size=(samples, int(np.ceil(len(values) / 3))))
    indices = ((starts[..., None] + np.arange(3)) % len(values)).reshape(samples, -1)[
        :, : len(values)
    ]
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025 / comparisons))


def validate_weights(root, protocol):
    import shutil
    from datetime import UTC, datetime
    from uuid import uuid4

    directory = root / "state/baseline-v2-validation"
    panel = pd.read_parquet(directory / "monthly-panel.parquet")
    manifest = json.loads((directory / "panel-manifest.json").read_text(encoding="utf-8"))
    if (
        hashlib.sha256((directory / "monthly-panel.parquet").read_bytes()).hexdigest()
        != manifest["panel_sha256"]
    ):
        raise ValueError("VALIDATION_PANEL_HASH_MISMATCH")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    run_dir = directory / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("monthly-panel.parquet", "coverage.parquet", "panel-manifest.json"):
        shutil.copy2(directory / name, run_dir / name)
    names = list(manifest["model_policy"]["factors"])
    train = purged_split(panel, protocol["start"], protocol["train_end"])
    valid = purged_split(panel, protocol["valid_start"], protocol["valid_end"])
    test = purged_split(panel, protocol["test_start"], protocol["test_end"])
    if min(train.date.nunique(), valid.date.nunique(), test.date.nunique()) < 6:
        raise ValueError("INSUFFICIENT_PURGED_SPLIT_MONTHS")
    models, grid = {}, []
    for alpha in protocol["ridge_alphas"]:
        model = fit_qlib_linear(train, names, alpha)
        metrics = daily_metrics(valid, valid[names].to_numpy() @ model.coef_ + model.intercept_)
        grid.append({"alpha": alpha, "valid_rank_ic": float(metrics.rank_ic.mean())})
        models[alpha] = model
    if not all(np.isfinite(row["valid_rank_ic"]) for row in grid):
        raise ValueError("INVALID_VALIDATION_RANK_IC")
    chosen = max(grid, key=lambda x: (x["valid_rank_ic"], -x["alpha"]))["alpha"]
    model = models[chosen]
    learned = daily_metrics(test, test[names].to_numpy() @ model.coef_ + model.intercept_)
    old = daily_metrics(test, test.score.to_numpy())
    compare = learned.merge(old, on="date", suffixes=("_learned", "_fixed"), validate="one_to_one")
    compare["rank_ic_improvement"] = compare.rank_ic_learned - compare.rank_ic_fixed
    compare["basket_improvement_proxy"] = (
        compare.basket_excess_proxy_learned - compare.basket_excess_proxy_fixed
    )
    lower = block_lower_bound(
        compare.rank_ic_improvement,
        int(protocol["bootstrap_samples"]),
        int(protocol["seed"]),
        len(grid),
    )
    lower_basket = block_lower_bound(
        compare.basket_improvement_proxy,
        int(protocol["bootstrap_samples"]),
        int(protocol["seed"]),
        len(grid),
    )
    failures = []
    coverage = pd.DataFrame(manifest["coverage"])
    historical = coverage.loc[coverage.date.le(protocol["test_end"])]
    if historical.coverage.min() < protocol["minimum_universe_coverage"]:
        failures.append("HISTORICAL_TOP1000_FINANCIAL_OR_PRICE_COVERAGE")
    if len(compare) < int(protocol["minimum_test_months"]):
        failures.append("INSUFFICIENT_TEST_MONTHS")
    if lower is None or lower <= 0:
        failures.append("NO_ROBUST_RANK_IC_IMPROVEMENT")
    if lower_basket is None or lower_basket <= 0:
        failures.append("NO_ROBUST_BASKET_IMPROVEMENT")
    # Historical report revision times are not established by a present-day API pull.
    failures.extend(
        [
            "ORIGINAL_FINANCIAL_VINTAGES_NOT_ATTESTED",
            "CONTINUOUS_INTEGER_COST_BACKTEST_NOT_COMPLETED",
        ]
    )
    if not manifest.get("industry_control"):
        failures.append("LIVE_INDUSTRY_CONTROL_NOT_VALIDATED_BY_SIZE_ONLY_TRAINING")
    compare.to_parquet(directory / "holdout-comparison.parquet", index=False)
    compare.to_parquet(run_dir / "holdout-comparison.parquet", index=False)
    # Freeze alpha from 2023, then walk forward using only previously matured labels.
    rolling = []
    for date in sorted(test.date.unique()):
        past = panel.loc[
            panel.date.lt(date)
            & panel.label_end.astype(str).lt(pd.Timestamp(date).strftime("%Y%m%d"))
            & panel.label.notna()
        ]
        months = sorted(past.date.unique())[-36:]
        fitted = fit_qlib_linear(past.loc[past.date.isin(months)], names, chosen)
        future = test.loc[test.date.eq(date)]
        metrics = daily_metrics(future, future[names].to_numpy() @ fitted.coef_ + fitted.intercept_)
        rolling.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "train_end": str(past.label_end.max()),
                "weights": dict(zip(names, fitted.coef_.tolist(), strict=True)),
                **metrics.iloc[0].to_dict(),
            }
        )
    # Deployable model is NOT emitted until all release gates pass.
    result = {
        "run_id": run_id,
        "evidence_path": str(run_dir),
        "status": "BLOCKED_NOT_FINAL_V2" if failures else "PASS_RESEARCH_ONLY",
        "protocol": protocol,
        "gate_failures": failures,
        "selected_alpha": chosen,
        "validation_grid": grid,
        "train_months": int(train.date.nunique()),
        "valid_months": int(valid.date.nunique()),
        "test_months": len(compare),
        "fixed_weights": manifest["model_policy"]["factors"],
        "diagnostic_coefficients_not_released": dict(zip(names, model.coef_.tolist(), strict=True)),
        "test_mean_rank_ic_fixed": float(compare.rank_ic_fixed.mean()),
        "test_mean_rank_ic_learned": float(compare.rank_ic_learned.mean()),
        "rank_ic_improvement_ci_low": lower,
        "basket_improvement_proxy_ci_low": lower_basket,
        "rolling_test": rolling,
        "real_orders_enabled": False,
        "automatic_promotion": False,
        "production_policy_updated": False,
        "panel_sha256": manifest["panel_sha256"],
        "comparison_scope": "SAME_HISTORICAL_ELIGIBLE_POOL_AND_CONTROLS_NOT_ORIGINAL_LIVE_PORTFOLIO",
    }
    rolling_frame = pd.DataFrame(rolling).merge(old, on="date", suffixes=("_rolling", "_fixed"))
    result["rolling_rank_ic_improvement_ci_low"] = block_lower_bound(
        rolling_frame.rank_ic_rolling - rolling_frame.rank_ic_fixed,
        int(protocol["bootstrap_samples"]),
        int(protocol["seed"]),
        len(grid),
    )
    if (
        result["rolling_rank_ic_improvement_ci_low"] is None
        or result["rolling_rank_ic_improvement_ci_low"] <= 0
    ):
        failures.append("NO_ROBUST_ROLLING_RANK_IC_IMPROVEMENT")
    result["status"] = "BLOCKED_NOT_FINAL_V2" if failures else "PASS_RESEARCH_ONLY"
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    factor_diagnostics = {}
    rolling_weights = pd.DataFrame([r["weights"] for r in rolling])
    for name in names:
        details = {}
        for stage, frame in (("train", train), ("valid", valid), ("test", test)):
            ic = (
                frame.groupby("date")
                .apply(
                    lambda f, factor=name: f[factor].corr(f.label, method="spearman"),
                    include_groups=False,
                )
                .dropna()
            )
            details[stage] = {
                "mean_rank_ic": float(ic.mean()),
                "months": len(ic),
                "ci_low": block_lower_bound(
                    ic, int(protocol["bootstrap_samples"]), int(protocol["seed"]), len(names)
                ),
            }
        if not rolling_weights.empty:
            w = rolling_weights[name]
            details["rolling_coefficient"] = {
                "median": float(w.median()),
                "std": float(w.std()),
                "positive_fraction": float(w.gt(0).mean()),
            }
        missing_column = "missing__" + name
        details["test_observed_share"] = (
            float(1 - test[missing_column].mean()) if missing_column in test else None
        )
        factor_diagnostics[name] = details
    result["factor_diagnostics"] = factor_diagnostics
    result["test_years"] = {
        str(year): {
            "fixed_rank_ic": float(frame.rank_ic_fixed.mean()),
            "learned_rank_ic": float(frame.rank_ic_learned.mean()),
            "basket_improvement_proxy": float(frame.basket_improvement_proxy.mean()),
        }
        for year, frame in compare.groupby(compare.date.str[:4])
    }
    correlation = train[names].corr()
    result["train_factor_correlation"] = correlation.to_dict()
    write_diagnostic(run_dir / "weight-validation.json", result)
    write_diagnostic(directory / "weight-validation.json", result)
    return {k: v for k, v in result.items() if k != "rolling_test"}
