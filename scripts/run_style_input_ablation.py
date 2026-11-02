"""Fixed-budget Ridge/LGBM input ablation; no new GRU, account or trading calls."""
import json
import os
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.qlib_challenger import challenger_universe, QlibPanelDataset
from moneymore.research.qlib_feature_preprocessing import fit_training_processor, transform_features
from moneymore.research.style_price_features import price_features, named_price_risk_exposures
from moneymore.research.baseline_v2_industry import industry_as_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "state/style-balance/input-ablation-20261102-a"


def main():
    OUT.mkdir(exist_ok=False)
    status = {"pid": os.getpid(), "status": "BUILDING_INPUTS", "orders_enabled": False, "completed": []}
    def save():
        status["updated_at"] = datetime.now(UTC).isoformat()
        (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    save()
    try:
        import lightgbm as lgb
        from qlib.contrib.model.linear import LinearModel
        store = ParquetStore(ROOT / "data")
        universe = challenger_universe(ROOT, store)
        as_of = "20261030"
        days = [d for d in trading_dates(store, as_of) if "20190101" <= d <= as_of]
        filters = [("ts_code", "in", list(universe)), ("trade_date", ">=", days[0]), ("trade_date", "<=", as_of)]
        market = store.read("daily", columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"], filters=filters)
        adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
        joined = market.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        frames = {"raw": [], "adjusted": []}
        price_parts = []
        for symbol, group in joined.groupby("ts_code", sort=True):
            group = group.set_index("trade_date").reindex(days)
            total_price = group.close * group.adj_factor
            label = total_price.shift(-5) / total_price - 1
            price_parts.append(total_price.rename(symbol))
            for basis in frames:
                f = price_features(group, basis)
                f.columns = pd.MultiIndex.from_product([["feature"], f.columns])
                f[("label", "TOTAL_RETURN")] = label
                f["datetime"] = pd.to_datetime(f.index)
                f["instrument"] = symbol
                frames[basis].append(f.reset_index(drop=True).set_index(["datetime", "instrument"]))
        raw = pd.concat(frames["raw"]).sort_index()
        adjusted = pd.concat(frames["adjusted"]).sort_index()
        # Same usable rows across all four input variants; never fill gaps.
        common = raw.dropna().index.intersection(adjusted.dropna().index)
        raw, adjusted = raw.loc[common], adjusted.loc[common]
        del frames
        sessions = pd.DatetimeIndex(pd.to_datetime(days))
        valid_start = pd.Timestamp("2023-01-01")
        first_valid = sessions[sessions >= valid_start][0]
        valid_first_index = sessions.get_loc(first_valid)
        first_test = sessions[sessions >= pd.Timestamp("2024-01-01")][0]
        test_first_index = sessions.get_loc(first_test)
        last_test_index = sessions.get_loc(sessions[sessions <= pd.Timestamp("2025-12-31")][-1])
        segments = {"train": ("2020-01-01", str(sessions[valid_first_index-6].date())),
            "valid": (str(first_valid.date()), str(sessions[test_first_index-6].date())),
            "test": (str(first_test.date()), str(sessions[last_test_index-5].date()))}
        prices = pd.concat(price_parts, axis=1)
        basic = store.read("daily_basic", columns=["ts_code", "total_mv"], filters=[("trade_date", "==", as_of)])
        size = basic.drop_duplicates("ts_code").set_index("ts_code").total_mv
        exposures, risk_meta = named_price_risk_exposures(prices, size)
        membership = industry_as_of(store.read("baseline_v2_industry_membership"), as_of)
        exposures["industry"] = exposures.index.map(membership.l1_name)
        exposures["listing_board"] = ["STAR" if s.startswith("68") else "CHINEXT" if s.startswith("30") else "MAIN" for s in exposures.index]
        exposures.to_parquet(OUT / "named-exposures.parquet")
        (OUT / "risk-coverage.json").write_text(json.dumps({**risk_meta,
            "as_of": as_of, "rows": len(exposures), "missing_fields": exposures.isna().sum().to_dict(),
            "missing_optional_factors": ["value", "quality", "growth", "liquidity"],
            "status": "EXPOSURES_ONLY_NOT_COVARIANCE_MODEL_NOT_CONSTRAINTS"}, indent=2), encoding="utf-8")
        status.update(status="FITTING_BASELINES", segments=segments, samples=len(common))
        save()
        all_results = []
        for basis, panel in (("raw", raw), ("adjusted", adjusted)):
            for scaling in ("none", "qlib_robust"):
                key = basis + "_" + scaling
                frame = panel
                if scaling != "none":
                    artifact = fit_training_processor(frame, *segments["train"])
                    frame = transform_features(artifact, frame)
                    (OUT / f"{key}-processor.pkl").write_bytes(pickle.dumps(artifact))
                dataset = QlibPanelDataset(frame, segments)
                train = dataset.prepare("train", ["feature", "label"])
                valid = dataset.prepare("valid", ["feature", "label"])
                test = dataset.prepare("test", ["feature", "label"])
                ridge = LinearModel(estimator="ridge", alpha=1., fit_intercept=True, include_valid=False)
                ridge.fit(dataset)
                booster = lgb.train({"objective": "regression", "verbosity": -1, "learning_rate": .03,
                    "num_leaves": 31, "num_threads": 4, "seed": 17},
                    lgb.Dataset(train["feature"], label=train["label"].iloc[:, 0]), num_boost_round=200,
                    valid_sets=[lgb.Dataset(valid["feature"], label=valid["label"].iloc[:, 0])],
                    callbacks=[lgb.early_stopping(20, verbose=False)])
                for name, model in (("ridge", ridge), ("lightgbm", booster)):
                    score = ridge.predict(dataset, "test") if name == "ridge" else pd.Series(booster.predict(test["feature"]), index=test.index)
                    prediction = pd.concat([score.rename("score"), test["label"].iloc[:, 0].rename("economic_label")], axis=1)
                    ic = prediction.groupby(level="datetime").apply(lambda x: x.score.corr(x.economic_label, method="spearman"), include_groups=False)
                    prediction.to_parquet(OUT / f"{key}-{name}-predictions.parquet")
                    (OUT / f"{key}-{name}.pkl").write_bytes(pickle.dumps(model))
                    all_results.append({"input": key, "model": name, "rank_ic": float(ic.mean()),
                        "test_days": len(ic), "monthly_ic": ic.groupby(ic.index.strftime("%Y-%m")).mean().to_dict(),
                        "lightgbm_best_round": booster.best_iteration if name == "lightgbm" else None})
                status["completed"].append(key)
                save()
        result = {"status": "EXPLORATORY_ABLATION_COMPLETED_NOT_RELEASED", "results": all_results,
            "segments": segments, "feature_count": 12, "lags": 1, "orders_enabled": False,
            "limitations": ["current candidate pool, not historical PIT training universe", "previously observed 2024-2025 test data",
                "vendor adjustment factors not fully event-verified", "not original GRU 60-lag architecture",
                "close-to-close labels, not executable next-open portfolio returns", "no cost/replay/profitability claim"]}
        (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        status["status"] = result["status"]
        save()
        print(json.dumps({"status": status["status"], "models": len(all_results)}), flush=True)
        if os.name == "nt":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except Exception as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
