import importlib.util
import sys
from pathlib import Path

import pandas as pd

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("corrected_stress", scripts / "stress_corrected_v2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.remove(str(scripts))


def test_increased_slippage_rechecks_both_limits_and_suspension():
    frame = pd.DataFrame({
        "open": [9.995, 10.005, 10.0], "up_limit": [10.0, 12.0, 12.0],
        "down_limit": [8.0, 10.0, 8.0], "can_buy": [True, True, False],
        "can_sell": [True, True, False],
        "verified_full_session_suspension": [False, False, True],
    })
    changed = module.stress_market(frame, "cost_10bps")
    assert changed.can_buy.tolist() == [False, True, False]
    assert changed.can_sell.tolist() == [True, False, False]
    assert frame.can_buy.tolist() == [True, True, False]


def test_native_risk_does_not_mutate_market_or_change_cost():
    frame = pd.DataFrame({"open": [10.0], "can_buy": [True]})
    changed = module.stress_market(frame, "native_risk")
    pd.testing.assert_frame_equal(frame, changed)
    assert changed is not frame
