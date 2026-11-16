import numpy as np
import pandas as pd
import pytest

from moneymore.research.ashare_mainline import (
    NativeDataset,
    assert_keys,
    fit_processor,
    learning_frame,
    market_context,
    prediction_diagnostics,
    purged_split,
    rank_target,
    transform,
    validate_financial_dates,
)


def test_split_purges_train_valid_and_test_endpoints():
    dates = pd.bdate_range("2024-01-01", "2025-04-30")
    panel = pd.DataFrame(
        {
            "date": dates.strftime("%Y%m%d"),
            "label_end_20": pd.Series(dates).shift(-20).dt.strftime("%Y%m%d"),
        }
    )
    masks = purged_split(panel, "2025Q1")
    assert panel.loc[masks["train"], "label_end_20"].max() < panel.loc[masks["valid"], "date"].min()
    assert panel.loc[masks["valid"], "label_end_20"].max() < panel.loc[masks["test"], "date"].min()
    assert not (masks["train"] & masks["valid"]).any()
    panel.loc[masks["test"], "label_end_20"] = None
    assert purged_split(panel, "2025Q1")["test"].equals(masks["test"])


def test_processor_uses_only_training_and_unknown_categories_are_explicit():
    train = pd.DataFrame(
        {
            "date": ["20240101"] * 3,
            "x": [1.0, 2.0, 3.0],
            "absent": [np.nan] * 3,
            "industry": ["A", "A", "B"],
        }
    )
    p = fit_processor(train, ["x", "absent"], categorical=True)
    test = pd.DataFrame(
        {"x": [np.nan, 1e20], "absent": [200.0, 300.0], "industry": ["FUTURE", None]}
    )
    result = transform(p, test)
    assert p["median"]["x"] == 2
    assert p["dropped_training_constant_or_missing"] == ["absent"]
    assert result.x.tolist() == [0.0, 8.0]
    assert result.missing__x.tolist() == [1.0, 0.0]
    assert result.industry_unknown.tolist() == [1.0, 1.0]


def test_financial_changes_require_both_observations_visible():
    frame = pd.DataFrame(
        {
            "date": ["20240102"],
            "roe": [2.0],
            "sales_yoy": [3.0],
            "roe_change": [1.0],
            "sales_change": [1.0],
            "roe_ann_date": ["20240101"],
            "q_sales_yoy_ann_date": ["20240101"],
            "previous_roe_ann_date": ["20230101"],
            "previous_q_sales_yoy_ann_date": ["20230101"],
        }
    )
    validate_financial_dates(frame)
    frame["previous_roe_ann_date"] = "20240102"
    with pytest.raises(ValueError, match="FINANCIAL_NOT_VISIBLE"):
        validate_financial_dates(frame)


def test_market_features_are_prefix_invariant_and_membership_specific():
    days = pd.bdate_range("2024-01-01", periods=90).strftime("%Y%m%d")
    bars = pd.DataFrame(
        [
            {"trade_date": d, "ts_code": s, "close": (1 + rate) ** i, "adj_factor": 1.0}
            for i, d in enumerate(days)
            for s, rate in [("A", 0.01), ("B", -0.01)]
        ]
    )
    members = pd.DataFrame({"date": days, "symbol": "A"})
    result, _ = market_context(bars, members)
    assert result.market_return1.dropna().sub(0.01).abs().max() < 1e-12
    changed = bars.copy()
    changed.loc[changed.trade_date > days[70], "close"] *= 10
    second, _ = market_context(changed, members)
    pd.testing.assert_frame_equal(result.loc[: days[70]], second.loc[: days[70]])
    # Unknown gap is not a zero-return observation.
    broken = bars.loc[~((bars.ts_code == "A") & (bars.trade_date == days[65]))]
    third, _ = market_context(broken, members)
    assert pd.isna(third.loc[days[65], "market_return1"])
    assert pd.isna(third.loc[days[70], "market_volatility20"])


def test_inference_adapter_is_label_free_and_ranks_are_cross_sectional():
    frame = pd.DataFrame({"date": ["20240101"] * 3, "label_20": [1.0, 2.0, np.nan]})
    target = rank_target(frame)
    assert target.iloc[0] == 0
    assert pd.isna(target.iloc[2])
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    train = learning_frame(x, target)
    data = NativeDataset({"train": train, "test": x})
    assert data.prepare("test").equals(x)
    assert data.prepare("test", col_set="feature").equals(x)
    assert data.prepare(["train"], col_set=["feature", "label"])[0].equals(train)


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="INVALID_PANEL_KEYS"):
        assert_keys(pd.DataFrame({"date": ["20240101"] * 2, "symbol": ["A"] * 2}))


def test_entire_missing_membership_day_breaks_market_window():
    days = pd.bdate_range("2024-01-01", periods=90).strftime("%Y%m%d")
    bars = pd.DataFrame(
        {"trade_date": days, "ts_code": "A", "close": np.arange(90) + 100.0, "adj_factor": 1.0}
    )
    members = pd.DataFrame({"date": days, "symbol": "A"})
    members = members.loc[members.date != days[65]]
    result, _ = market_context(bars, members)
    assert pd.isna(result.loc[days[65], "market_return1"])
    assert pd.isna(result.loc[days[70], "market_return20"])


def test_empty_future_fields_do_not_change_fitted_schema():
    train = pd.DataFrame(
        {"date": ["20240101", "20240102"], "x": [1.0, 2.0], "future": [np.nan, np.nan]}
    )
    p = fit_processor(train, ["x", "future"])
    a = transform(p, pd.DataFrame({"x": [3.0], "future": [np.nan]}))
    b = transform(p, pd.DataFrame({"x": [3.0], "future": [999.0]}))
    pd.testing.assert_frame_equal(a, b)


def test_missing_entire_price_day_does_not_compress_exchange_calendar():
    days = pd.bdate_range("2024-01-01", periods=90).strftime("%Y%m%d")
    bars = pd.DataFrame(
        {"trade_date": days, "ts_code": "A", "close": np.arange(90) + 100.0, "adj_factor": 1.0}
    )
    bars = bars.loc[bars.trade_date != days[65]]
    members = pd.DataFrame({"date": days, "symbol": "A"})
    result, _ = market_context(bars, members, calendar=days)
    assert pd.isna(result.loc[days[65], "market_return1"])
    assert pd.isna(result.loc[days[66], "market_return1"])
    assert pd.isna(result.loc[days[70], "market_return20"])


def test_immature_labels_keep_prediction_and_exclude_performance():
    frame = pd.DataFrame(
        {
            "date": ["20250102"] * 3,
            "symbol": ["600000.SH", "300001.SZ", "688001.SH"],
            "industry": ["A", "B", "B"],
            "score": [1.0, 2.0, 3.0],
            "label_20": [1.0, 2.0, 3.0],
            "label_end_20": ["20250401"] * 3,
        }
    )
    result = prediction_diagnostics(frame, "20250331").iloc[0]
    assert result.predicted_rows == 3 and result.evaluated_rows == 0 and pd.isna(result.rank_ic)


def test_no_silent_three_score_fallback():
    train = pd.DataFrame({"date": ["20240101"], "x": [np.nan]})
    with pytest.raises(ValueError, match="NO_TRAINING_FEATURES"):
        fit_processor(train, ["x"])


def test_native_qlib_fit_predict_and_reload(tmp_path):
    import pickle

    from threadpoolctl import threadpool_limits

    native = pytest.importorskip("qlib.contrib.model.double_ensemble")
    from moneymore.research.ashare_mainline import predict

    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(400, 12)), columns=[f"f{i}" for i in range(12)])
    y = x.f0 * 2 - x.f1 + rng.normal(size=400) * 0.1
    train, valid = (
        learning_frame(x.iloc[:300].reset_index(drop=True), y.iloc[:300]),
        learning_frame(x.iloc[300:].reset_index(drop=True), y.iloc[300:]),
    )
    np.random.seed(42)
    with threadpool_limits(limits=2):
        model = native.DEnsembleModel(
            num_models=2,
            epochs=3,
            decay=0.5,
            num_threads=2,
            num_leaves=7,
            min_data_in_leaf=10,
            verbosity=-1,
            seed=42,
        )
        model.fit(NativeDataset({"train": train, "valid": valid}))
        score = predict(model, "doubleensemble", x.iloc[300:])
        saved = tmp_path / "model.pkl"
        saved.write_bytes(pickle.dumps(model))
        second = predict(pickle.loads(saved.read_bytes()), "doubleensemble", x.iloc[300:])
    assert np.isfinite(score).all() and np.std(score) > 0
    np.testing.assert_array_equal(score, second)
