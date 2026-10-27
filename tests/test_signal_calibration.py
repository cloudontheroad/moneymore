import numpy as np
import pandas as pd

from moneymore.research.signal_calibration import rank_signal, walk_forward_calibration


def test_rank_units_invariant():
    s = pd.Series([3., 1., 8.])
    np.testing.assert_allclose(rank_signal(s), rank_signal(s * 100 + 12))
    assert abs(rank_signal(s).mean()) < 1e-12


def test_maturity_purge_and_no_future_label_leakage():
    days = [f"{i:08d}" for i in range(40)]
    rows = [dict(date=days[i], maturity=days[i+5], horizon=5, prospective=True,
                 valid_label=True, x=np.array([-1., 1.]), y=np.array([-.02, .02])) for i in range(30)]
    folds = walk_forward_calibration(rows, days, minimum_train_dates=5)
    assert folds[10]["train_dates"] == 4
    assert folds[11]["train_dates"] == 5
    assert folds[11]["horizon_slope"] == .02
    rows[-1]["y"] *= 1000
    assert walk_forward_calibration(rows, days, minimum_train_dates=5)[11] == folds[11]


def test_negative_edge_not_promoted_or_inverted():
    days = [f"{i:08d}" for i in range(15)]
    rows = [dict(date=days[i], maturity=days[i+1], horizon=1, prospective=True,
                 valid_label=True, x=np.array([-1., 1.]), y=np.array([.02, -.02])) for i in range(12)]
    latest = walk_forward_calibration(rows, days, minimum_train_dates=3)[-1]
    assert latest["status"] == "NO_POSITIVE_CALIBRATED_EDGE"
    assert latest["daily_slope"] < 0
    assert not latest["orders_enabled"]
