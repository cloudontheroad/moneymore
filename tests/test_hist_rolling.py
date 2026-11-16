"""Annual refits must use only matured labels and train-only preprocessing."""
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_hist_rolling import (
    CONFIG,
    fit_robust,
    fold_schedule,
    learning_mask,
    normalize,
    rank_labels,
    raw_alpha360,
)


def test_fold_first_signal_precedes_first_trade_and_excludes_its_future_label():
    calendar = pd.DatetimeIndex(pd.to_datetime(["2016-09-30", "2019-09-30", "2019-10-08",
        "2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03", "2020-12-30", "2020-12-31", "2021-01-04"]))
    fold = fold_schedule(2020, calendar)
    assert fold["train"] == ["2016-10-01", "2019-09-30"]
    assert fold["valid"] == ["2019-10-01", "2019-12-31"]
    assert fold["test"] == ["2019-12-31", "2020-12-30"]
    assert fold["execution"] == ["2020-01-02", "2020-12-31"]


def test_split_purge_uses_realization_date_not_just_signal_date():
    dates = pd.to_datetime(["2019-09-27", "2019-09-30", "2019-12-30", "2019-12-31"])
    index = pd.MultiIndex.from_arrays([dates, ["A"]*4], names=["datetime", "instrument"])
    realization = np.array(["2019-09-30", "2019-10-08", "2019-12-31", "2020-01-02"], dtype="datetime64[ns]")
    train = learning_mask(index, np.ones(4), realization, "2016-10-01", "2019-09-30")
    valid = learning_mask(index, np.ones(4), realization, "2019-10-01", "2019-12-31")
    np.testing.assert_array_equal(train, [True, False, False, False])
    np.testing.assert_array_equal(valid, [False, False, True, False])


def test_missing_training_labels_excluded_but_not_inference_rows():
    index = pd.MultiIndex.from_arrays([pd.to_datetime(["2019-12-30"]*2), ["A", "B"]], names=["datetime", "instrument"])
    labels = np.array([.1, np.nan])
    realization = np.array(["2019-12-31"]*2, dtype="datetime64[ns]")
    assert learning_mask(index, labels, realization, "2019-10-01", "2019-12-31").tolist() == [True, False]
    dates = index.get_level_values("datetime")
    assert ((dates >= "2019-12-30") & (dates <= "2019-12-30")).sum() == 2


def test_train_only_normalizer_is_identical_to_native_robust_processor():
    from qlib.data.dataset.processor import RobustZScoreNorm
    x = np.array([[1., 2., np.nan], [3., 2., np.nan], [4., 6., np.nan]], dtype=np.float32)
    mean, std = fit_robust(x)
    idx = pd.MultiIndex.from_arrays([pd.date_range("2018-01-01", periods=3), ["A"]*3], names=["datetime", "instrument"])
    frame = pd.DataFrame(x, index=idx, columns=pd.MultiIndex.from_product([["feature"], ["A", "B", "VWAP"]]))
    reference = RobustZScoreNorm("2018-01-01", "2018-01-03", fields_group="feature")
    with pytest.warns(RuntimeWarning, match="All-NaN"):
        reference.fit(frame)
    np.testing.assert_array_equal(mean, reference.mean_train)
    np.testing.assert_array_equal(std, reference.std_train)
    # Arbitrarily extreme future data must not change fitted statistics.
    transformed = normalize(np.array([[1e9, -1e9, np.nan]], dtype=np.float32), mean, std)
    np.testing.assert_array_equal(transformed, [[3., -3., 0.]])


def test_cross_sectional_labels_do_not_rank_across_dates():
    idx = pd.MultiIndex.from_arrays([pd.to_datetime(["2018-01-01"]*2+["2018-01-02"]*2), list("ABAB")], names=["datetime", "instrument"])
    labels = rank_labels(np.array([10., 20., -100., -50.]), idx)
    np.testing.assert_allclose(labels, [0., 1.73, 0., 1.73])


def test_no_same_signal_date_between_adjacent_years():
    calendar = pd.bdate_range("2016-01-01", "2026-11-16")
    plans = [fold_schedule(y, calendar) for y in CONFIG["years"]]
    for previous, following in pairwise(plans):
        assert pd.Timestamp(previous["test"][1]) < pd.Timestamp(following["test"][0])
        assert pd.Timestamp(following["test"][0]) < pd.Timestamp(following["execution"][0])
    assert plans[-1]["execution"][1] == "2026-11-13"


def test_frozen_protocol_covers_seven_years_ten_seeds_and_no_production():
    assert len(CONFIG["years"])*len(CONFIG["seeds"]) == 70
    assert CONFIG["training_years"] == 3 and CONFIG["validation_months"] == 3
    assert CONFIG["maximum_training_workers"] == CONFIG["retry_attempts"] == 2
    assert CONFIG["production_enabled"] is CONFIG["real_orders_enabled"] is False


def test_vectorized_features_retain_offsets_lags_and_missing_vwap(tmp_path):
    dates = pd.bdate_range("2018-01-01", periods=100)
    (tmp_path/"calendars").mkdir()
    (tmp_path/"calendars/day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")))
    dest = tmp_path/"features/sh600000"
    dest.mkdir(parents=True)
    close = np.arange(1, 101, dtype=np.float32)
    for field, values in (("close", close), ("open", close*2), ("high", close*3), ("low", close*.5), ("volume", close*10)):
        np.r_[np.float32(0), values].astype("<f4").tofile(dest/f"{field}.day.bin")
    index = pd.MultiIndex.from_arrays([[dates[-1]], ["SH600000"]], names=["datetime", "instrument"])
    values = raw_alpha360(tmp_path, index)
    np.testing.assert_allclose(values[0, :60], close[40:]/100)
    assert values[0, 59] == 1 and values[0, 119] == 2 and values[0, 179] == 3
    assert values[0, 239] == .5 and values[0, 359] == 1
    assert np.isnan(values[:, 240:300]).all()
