"""Current-as-of named controls and mature covariance estimators, no orders."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.research.baseline_v2_financials import common_period_features, required_period
from moneymore.research.style_risk_inputs import signed_inverse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("snapshot immutable")
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    source = root / "state/style-balance/input-ablation-20261102-a/named-exposures.parquet"
    raw = pd.read_parquet(source)
    as_of = "20261030"
    symbols = list(raw.index)
    basic = store.read("daily_basic", filters=[("trade_date", "==", as_of), ("ts_code", "in", symbols)])
    basic = basic.drop_duplicates("ts_code").set_index("ts_code").reindex(symbols)
    raw["book_to_price"] = signed_inverse(basic.pb)
    raw["earnings_yield"] = signed_inverse(basic.pe_ttm)
    raw["dividend_yield"] = basic.dv_ttm / 100
    raw["turnover_rate"] = basic.turnover_rate / 100
    finances = common_period_features(research_financial_frame(store), as_of).set_index("ts_code")
    for field, name in (("roe", "quality_roe"), ("q_sales_yoy", "growth_sales_yoy")):
        raw[name] = finances[field].where(finances[field+"_end_date"].astype(str).eq(required_period(as_of))).reindex(symbols) / 100
    days = [d for d in trading_dates(store, as_of) if d <= as_of][-121:]
    filters = [("ts_code", "in", symbols), ("trade_date", ">=", days[0]), ("trade_date", "<=", as_of)]
    bars = store.read("daily", columns=["ts_code", "trade_date", "close", "amount"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    bars = bars.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    bars["total_price"] = bars.close * bars.adj_factor
    prices = bars.pivot(index="trade_date", columns="ts_code", values="total_price").reindex(days)
    amount = bars.pivot(index="trade_date", columns="ts_code", values="amount").reindex(days)
    raw["log_amount20"] = np.log(amount.tail(20).mean().where(amount.tail(20).notna().sum().eq(20)).where(lambda x: x > 0) * 1000)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    numeric = raw.select_dtypes(include="number").columns.tolist()
    returns = prices.pct_change(fill_method=None).iloc[1:]
    complete_return = returns.notna().all()
    eligible = raw[numeric].notna().all(axis=1) & raw.industry.notna() & complete_return.reindex(raw.index, fill_value=False)
    selected = raw.loc[eligible].copy()
    if len(selected) < 100:
        raise ValueError("insufficient fully observed named risk universe")
    scaler = StandardScaler()
    controls = pd.DataFrame(scaler.fit_transform(selected[numeric]), index=selected.index, columns=numeric)
    industry = pd.get_dummies(selected.industry, prefix="industry", dtype=float)
    controls = pd.concat([controls, industry], axis=1)
    x = returns.reindex(columns=selected.index)
    estimator = LedoitWolf().fit(x)
    covariance = estimator.covariance_
    eigen = np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or eigen.min() < -1e-10:
        raise ValueError("covariance not finite PSD")
    # Qlib's statistical factor decomposition is kept separate from named
    # characteristic controls. PCA is not relabelled as value/quality factors.
    from qlib.model.riskmodel import StructuredCovEstimator
    pca = StructuredCovEstimator(num_factors=10, factor_model="pca", scale_return=False)
    loadings, factor_cov, specific = pca.predict(x, is_price=False, return_decomposed_components=True)
    reconstructed = loadings @ np.atleast_2d(factor_cov) @ loadings.T + np.diag(specific)
    if np.linalg.eigvalsh(reconstructed).min() < -1e-10 or (specific < 0).any():
        raise ValueError("Qlib factor decomposition invalid")
    args.output.mkdir(parents=True)
    raw.to_parquet(args.output / "raw-exposures.parquet")
    controls.to_parquet(args.output / "named-controls.parquet")
    pd.DataFrame(covariance, index=selected.index, columns=selected.index).to_parquet(args.output / "shrinkage-covariance.parquet")
    np.savez(args.output / "qlib-statistical-risk.npz", symbols=selected.index.to_numpy(str),
        loadings=loadings, factor_covariance=factor_cov, specific_variance=specific)
    (args.output / "scaler.pkl").write_bytes(pickle.dumps(scaler))
    finances.reindex(symbols).to_parquet(args.output / "financial-provenance.parquet")
    coverage = pd.DataFrame({"eligible": eligible, "missing_named_fields": raw[numeric].isna().apply(lambda r: ",".join(r.index[r]), axis=1),
        "missing_return_history": ~complete_return.reindex(raw.index, fill_value=False)})
    coverage.to_parquet(args.output / "coverage.parquet")
    report = {"status": "CURRENT_RISK_SNAPSHOT_VALIDATED_NOT_STRATEGY_PASS", "as_of": as_of,
        "orders_enabled": False, "requested": len(raw), "eligible": len(selected), "excluded": int((~eligible).sum()),
        "continuous_controls": numeric, "industry_controls": len(industry.columns),
        "missing_by_field": raw[numeric].isna().sum().to_dict(), "risk_sessions": len(x),
        "financial_period": required_period(as_of), "financial_visibility": "announcement_plus_one_day_common_period",
        "financial_provider_revisions_not_original_vintage_proven": True,
        "asset_covariance": {"estimator": "sklearn.LedoitWolf", "minimum_eigenvalue": float(eigen.min()), "shrinkage": float(estimator.shrinkage_)},
        "statistical_factor_covariance": {"estimator": "qlib.StructuredCovEstimator.PCA", "factor_count": int(loadings.shape[1]),
            "named_factor_returns": False},
        "source_exposures_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "coverage_bias_check": {"all_growth_board": int(raw.listing_board.isin(["STAR", "CHINEXT"]).sum()),
            "included_growth_board": int(selected.listing_board.isin(["STAR", "CHINEXT"]).sum()),
            "excluded_growth_board": int(raw.loc[~eligible].listing_board.isin(["STAR", "CHINEXT"]).sum())},
        "risk_universe_is_not_selection_universe": True,
        "skfolio_installed": __import__("importlib.metadata", fromlist=["version"]).version("skfolio"),
        "skfolio_characteristics_factor_model_available": hasattr(__import__("skfolio.prior", fromlist=["TimeSeriesFactorModel"]), "CharacteristicsFactorModel"),
        "limitations": ["snapshot-only, not historical style model or rolling factor returns",
            "no named-factor covariance/attribution claim; PCA and named controls are distinct",
            "normalization fitted on contemporaneous cross-section, not future dates",
            "no current model/scores/weights altered", "not a solver or constraint feasibility acceptance"]}
    (args.output / "risk-model.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "eligible": len(selected), "continuous_controls": len(numeric),
        "industry_controls": len(industry.columns), "covariance_psd": True}))


if __name__ == "__main__":
    main()
