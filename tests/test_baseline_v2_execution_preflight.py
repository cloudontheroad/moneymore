import pandas as pd
import pytest

from moneymore.research.baseline_v2_execution_preflight import paired_targets


def test_paired_targets_frozen_scores_not_future_return():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 12,
            "symbol": [str(i).zfill(2) for i in range(12)],
            "score": list(range(12)),
            "factor": list(range(12)),
            "forward_return": list(range(12)),
        }
    )
    report = {
        "diagnostic_coefficients_not_released": {"factor": -1.0},
        "protocol": {"test_start": "20240101", "test_end": "20251231"},
    }
    actual = paired_targets(frame, report)
    assert set(actual.loc[actual.variant.eq("fixed"), "symbol"]) == set(frame.symbol[2:])
    assert set(actual.loc[actual.variant.eq("learned"), "symbol"]) == set(frame.symbol[:10])
    assert actual.groupby("variant").target_weight.sum().eq(1).all()
    changed = frame.assign(forward_return=frame.forward_return * -100)
    pd.testing.assert_frame_equal(actual, paired_targets(changed, report))


def test_invalid_frozen_coefficients_rejected():
    with pytest.raises(ValueError, match="NONFINITE"):
        paired_targets(
            pd.DataFrame(), {"diagnostic_coefficients_not_released": {"a": float("nan")}}
        )
