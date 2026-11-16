import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "mainline_n3_final",
        Path(__file__).resolve().parents[1] / "scripts/finalize_ashare_mainline_n3.py",
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_paired_identical_series_have_exact_zero_difference():
    values = np.random.default_rng(42).normal(0.0002, 0.01, 181)
    result = module().paired_intervals(values, values)
    assert result["net_return_difference_ci95"] == [0.0, 0.0]
    assert result["sharpe_difference_ci95"] == [0.0, 0.0]


def test_paired_intervals_use_identical_resampling_and_are_reproducible():
    values = np.random.default_rng(42).normal(0.0002, 0.01, 181)
    better = values + 0.001
    a = module().paired_intervals(better, values)
    b = module().paired_intervals(better, values)
    assert a == b
    assert a["net_return_difference_ci95"][0] > 0
    assert a["sharpe_difference_ci95"][0] > 0


def test_calibration_zero_signal_is_explicit_and_future_labels_rejected():
    records = [
        {
            "case": model,
            "month": f"2026{month:02d}",
            "cutoff_exclusive": f"2026{month:02d}01",
            "last_label_end": "20251231",
            "beta": 0.001,
            "standard_error_block20": 0.002,
            "beta_conservative": 0.0,
        }
        for model in ("a", "b", "c", "d")
        for month in range(1, 10)
    ]
    result = module().calibration_diagnostics(records)
    assert result["zero_expected_return_months"] == 36
    assert len(result["by_model"]["a"]["zero_months"]) == 9
    records[0]["last_label_end"] = "20260101"
    with pytest.raises(ValueError, match="CALIBRATION_DIAGNOSTIC_CONTRACT_FAILED"):
        module().calibration_diagnostics(records)
