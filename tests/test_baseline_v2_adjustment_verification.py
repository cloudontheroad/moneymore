import pandas as pd
import pytest

from moneymore.research.baseline_v2_adjustment_verification import verify_no_economic_adjustment


def rows():
    prev = {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5}
    current = {"open": 10.5, "high": 11.5, "low": 10.0, "close": 11.0, "pre_close": 10.5}
    adj_prev = {k: v * 3.7 for k, v in prev.items()}
    adj_current = {k: current[k] * 3.7 for k in prev}
    return [prev, current, prev.copy(), current.copy(), adj_prev, adj_current]


def test_all_independent_conditions_required():
    assert verify_no_economic_adjustment(*rows(), has_action=False)[0]
    assert not verify_no_economic_adjustment(*rows(), has_action=True)[0]


@pytest.mark.parametrize(
    "case,field,value",
    [(1, "pre_close", 10.4), (3, "close", 11.01), (5, "close", 41.0), (4, "high", float("nan"))],
)
def test_bad_reference_independent_price_or_factor_stays_blocked(case, field, value):
    inputs = rows()
    inputs[case][field] = value
    assert not verify_no_economic_adjustment(*inputs, has_action=False)[0]


def test_normalization_removes_only_verified_step_and_preserves_true_dividend():
    from moneymore.research.baseline_v2_adjustment_verification import normalize_verified_steps

    f = pd.DataFrame(
        {
            "ts_code": ["A"] * 4,
            "trade_date": ["1", "2", "3", "4"],
            "adj_factor": [2.0, 2.001, 2.001, 2.2011],
        }
    )
    proof = pd.DataFrame(
        [
            {
                "symbol": "A",
                "trade_date": "2",
                "original_adj_factor": 2.001,
                "original_factor_change": 2.001 / 2 - 1,
                "verified": True,
            }
        ]
    )
    original = f.copy(deep=True)
    actual, events = normalize_verified_steps(f, proof)
    assert actual.adj_factor.tolist() == pytest.approx([2, 2, 2, 2.2])
    assert actual.adj_factor.iloc[-1] / actual.adj_factor.iloc[-2] == pytest.approx(1.1)
    assert len(events) == 1
    pd.testing.assert_frame_equal(f, original)
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        normalize_verified_steps(f.assign(adj_factor=[2.0, 2.002, 2.002, 2.2022]), proof)
    with pytest.raises(ValueError, match="UNVERIFIED"):
        normalize_verified_steps(f, proof.assign(verified=False))
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        normalize_verified_steps(actual, proof)
