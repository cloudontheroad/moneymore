import numpy as np
import pandas as pd

from moneymore.research.baseline_v2_financials import common_period_features, required_period
from moneymore.research.baseline_v2_weights import (
    block_lower_bound,
    fit_qlib_linear,
    purged_split,
    validate_weights,
)


def test_purge_uses_label_end_not_feature_date_only():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-11-30", "2022-12-30", "2023-01-31"]),
            "label_end": ["20221228", "20230130", "20230228"],
            "label": [1.0, 2.0, 3.0],
        }
    )
    actual = purged_split(frame, "20200101", "20221231")
    assert actual.label.tolist() == [1.0]


def test_industry_dates_never_backfill_current_classification():
    from moneymore.research.baseline_v2_industry import industry_as_of

    f = pd.DataFrame(
        [
            {"ts_code": "A", "l1_code": "OLD", "in_date": "20180101", "out_date": "20200101"},
            {"ts_code": "A", "l1_code": "NEW", "in_date": "20210101", "out_date": None},
            {"ts_code": "B", "l1_code": "BAD", "in_date": "20180101", "out_date": "invalid"},
        ]
    )
    assert industry_as_of(f, "20191231").loc["A", "l1_code"] == "OLD"
    assert "A" not in industry_as_of(f, "20200630").index
    assert industry_as_of(f, "20210101").loc["A", "l1_code"] == "NEW"
    assert "B" not in industry_as_of(f, "20210101").index


def test_reporting_period_is_consistent_not_a_200_day_cliff():
    assert required_period("20260430") == "20250930"
    assert required_period("20260501") == "20260331"
    assert required_period("20260831") == "20260331"
    assert required_period("20260901") == "20260630"
    assert required_period("20261031") == "20260630"
    assert required_period("20261101") == "20260930"


def test_early_roe_report_cannot_mix_three_and_nine_month_measurements():
    rows = [
        {
            "ts_code": "A",
            "ann_date": "20251025",
            "end_date": "20250930",
            "roe": 12.0,
            "q_sales_yoy": 5.0,
        },
        {
            "ts_code": "A",
            "ann_date": "20260330",
            "end_date": "20251231",
            "roe": 20.0,
            "q_sales_yoy": 6.0,
        },
        {
            "ts_code": "A",
            "ann_date": "20260410",
            "end_date": "20260331",
            "roe": 4.0,
            "q_sales_yoy": 7.0,
        },
        {
            "ts_code": "B",
            "ann_date": "20251027",
            "end_date": "20250930",
            "roe": 13.0,
            "q_sales_yoy": 6.0,
        },
    ]
    actual = common_period_features(pd.DataFrame(rows), "20260429")
    assert set(actual.roe_end_date) == {"20250930"}
    assert actual.set_index("ts_code").loc["A", "roe"] == 12


def test_common_period_still_delays_announcement_one_day():
    f = pd.DataFrame(
        [
            {
                "ts_code": "A",
                "ann_date": "20260430",
                "end_date": "20260331",
                "roe": 4.0,
                "q_sales_yoy": 7.0,
            }
        ]
    )
    assert common_period_features(f, "20260430").empty
    assert common_period_features(f, "20260501").roe.iloc[0] == 4


def test_block_bootstrap_reproducible_and_does_not_turn_noise_into_pass():
    assert block_lower_bound([1] * 12, samples=100) == 1
    assert block_lower_bound([1, -1] * 12, samples=100) < 0
    assert block_lower_bound([1, 2]) is None
    assert block_lower_bound(np.arange(12), samples=100) == block_lower_bound(
        np.arange(12), samples=100
    )


def test_native_qlib_ridge_learns_without_mutating_input_frame():
    rng = np.random.default_rng(17)
    n = 100
    f = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-29"] * n),
            "symbol": [str(i) for i in range(n)],
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    f["label"] = 0.4 * f.a - 0.2 * f.b
    before = f.copy()
    model = fit_qlib_linear(f, ["a", "b"], 0.1)
    assert np.allclose(model.coef_, [0.4, -0.2], atol=0.01)
    pd.testing.assert_frame_equal(f, before)


def test_entire_validation_archives_evidence_and_never_releases_model(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    rng = np.random.default_rng(51)
    rows = []
    for year in (2022, 2023, 2024):
        for month in range(1, 7):
            date = pd.Timestamp(year, month, 15)
            for symbol in range(20):
                a, b = rng.normal(size=2)
                rows.append(
                    {
                        "date": date,
                        "symbol": str(symbol),
                        "a": a,
                        "b": b,
                        "label": 0.3 * a - 0.2 * b,
                        "forward_return": 0.03 * a - 0.02 * b,
                        "score": 0.5 * a + 0.5 * b,
                        "label_end": (date + pd.Timedelta(days=20)).strftime("%Y%m%d"),
                    }
                )
    directory = tmp_path / "state/baseline-v2-validation"
    directory.mkdir(parents=True)
    path = directory / "monthly-panel.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    coverage = [{"date": f"{y}0115", "coverage": 1.0} for y in (2022, 2023, 2024)]
    pd.DataFrame(coverage).to_parquet(directory / "coverage.parquet", index=False)
    manifest = {
        "model_policy": {"factors": {"a": 0.5, "b": 0.5}},
        "coverage": coverage,
        "panel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (directory / "panel-manifest.json").write_text(json.dumps(manifest))
    policy = {
        "start": "20220101",
        "train_end": "20221231",
        "valid_start": "20230101",
        "valid_end": "20231231",
        "test_start": "20240101",
        "test_end": "20241231",
        "ridge_alphas": [0.1, 1.0],
        "bootstrap_samples": 50,
        "seed": 7,
        "minimum_universe_coverage": 0.9,
        "minimum_test_months": 6,
    }
    first = validate_weights(tmp_path, policy)
    second = validate_weights(tmp_path, policy)
    assert first["evidence_path"] != second["evidence_path"]
    assert (Path(first["evidence_path"]) / "weight-validation.json").exists()
    assert first["status"] == "BLOCKED_NOT_FINAL_V2"
    assert first["production_policy_updated"] is False
    assert not (tmp_path / "state/factor-baseline-candidates").exists()
