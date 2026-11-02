import pickle
import numpy as np
import pandas as pd
import pytest
from moneymore.research.qlib_feature_preprocessing import fit_training_processor, transform_features


def frame():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2025-01-01", periods=10), ["A", "B"]], names=["datetime", "instrument"])
    return pd.DataFrame({("feature", "return_lag00"): np.arange(20)/100,
        ("feature", "volume_lag00"): np.arange(20)*100., ("label", "y"): np.arange(20)}, index=idx)


def test_future_values_do_not_fit_scaler_and_labels_unchanged():
    original = frame()
    altered = original.copy()
    altered.loc[altered.index.get_level_values("datetime") > pd.Timestamp("2025-01-08"), ("feature", "return_lag00")] = 1e9
    first = fit_training_processor(original, "2025-01-01", "2025-01-08")
    second = fit_training_processor(altered, "2025-01-01", "2025-01-08")
    np.testing.assert_array_equal(first["processor"].mean_train, second["processor"].mean_train)
    np.testing.assert_array_equal(first["processor"].std_train, second["processor"].std_train)
    transformed = transform_features(first, altered)
    assert transformed["feature"].abs().max().max() <= 3
    pd.testing.assert_frame_equal(transformed["label"], original["label"])
    pd.testing.assert_frame_equal(transform_features(pickle.loads(pickle.dumps(first)), original), transform_features(first, original))


def test_schema_mismatch_and_nonfinite_blocked():
    data = frame()
    artifact = fit_training_processor(data, "2025-01-01", "2025-01-08")
    with pytest.raises(ValueError):
        transform_features(artifact, data.drop(columns=[("feature", "volume_lag00")]))
    data.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        transform_features(artifact, data)
