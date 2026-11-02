"""Isolated versioned Qlib normalization. Never modifies an existing model."""
import hashlib
import inspect

import numpy as np
import pandas as pd


def fit_training_processor(frame, train_start, train_end):
    from qlib.data.dataset.processor import RobustZScoreNorm
    dates = frame.index.get_level_values("datetime")
    selected = frame.loc[(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))]
    if len(selected) == 0 or not np.isfinite(selected["feature"].to_numpy()).all():
        raise ValueError("training features empty or nonfinite")
    processor = RobustZScoreNorm(train_start, train_end, fields_group="feature", clip_outlier=True)
    processor.fit(frame)
    source = inspect.getsourcefile(RobustZScoreNorm)
    from pathlib import Path
    return {"version": "qlib_robust_zscore_feature_v1", "processor": processor,
        "schema": list(frame["feature"].columns), "fit_start": str(train_start), "fit_end": str(train_end),
        "fit_rows": len(selected), "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "labels_transformed": False, "production_enabled": False}


def transform_features(artifact, frame):
    if list(frame["feature"].columns) != artifact["schema"]:
        raise ValueError("feature schema/order mismatch")
    if not np.isfinite(frame["feature"].to_numpy()).all():
        raise ValueError("nonfinite inference features")
    original_labels = frame["label"].copy() if "label" in frame.columns.get_level_values(0) else None
    result = artifact["processor"](frame.copy(deep=True))
    if not np.isfinite(result["feature"].to_numpy()).all():
        raise ValueError("normalization produced nonfinite values")
    if original_labels is not None:
        pd.testing.assert_frame_equal(result["label"], original_labels)
    return result
