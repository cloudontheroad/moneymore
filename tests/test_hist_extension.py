"""Extension must not use future memberships or silently change old models."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from build_hist_extension import execution_calendar, original_array
from prepare_hist_extension import CONFIG, membership_from_snapshots


def test_membership_uses_only_latest_past_snapshot():
    prior = [f"{i:06d}.SH" for i in range(300)]
    after = [f"{i:06d}.SH" for i in range(100, 400)]
    weights = pd.DataFrame({"trade_date": ["20201231"]*300+["20210129"]*300,
                            "con_code": prior+after})
    result = membership_from_snapshots(weights, ["20210104", "20210128", "20210129", "20210201"])
    assert set(result.loc[result.trade_date.eq("20210128"), "ts_code"]) == set(prior)
    assert set(result.loc[result.trade_date.eq("20210129"), "ts_code"]) == set(after)
    assert (result.snapshot_date <= result.trade_date).all()
    assert result.groupby("trade_date").size().eq(300).all()


def test_membership_no_backward_fill_or_stale_indefinite_carry():
    weights = pd.DataFrame({"trade_date": ["20201231"]*300, "con_code": [str(i) for i in range(300)]})
    with pytest.raises(ValueError, match="No past"):
        membership_from_snapshots(weights, ["20201230"])
    with pytest.raises(ValueError, match="Stale"):
        membership_from_snapshots(weights, ["20210501"])


def test_membership_requires_full_constituent_snapshot():
    with pytest.raises(ValueError, match="300"):
        membership_from_snapshots(pd.DataFrame({"trade_date": ["20201231"], "con_code": ["A"]}), ["20210104"])


def test_binary_splice_retains_original_float_values_and_offset(tmp_path):
    dest = tmp_path/"features/sh600000"
    dest.mkdir(parents=True)
    values = np.array([2, 3.25, np.nan, 6.5, 99], dtype="<f4")
    values.tofile(dest/"close.day.bin")
    result = original_array(tmp_path, "SH600000", "close", 5)
    np.testing.assert_array_equal(result, [np.nan, np.nan, 3.25, np.nan, 6.5])
    assert np.isnan(original_array(tmp_path, "MISSING", "close", 5)).all()


def test_frozen_extension_protocol():
    assert CONFIG["extension_start"] == "20210101"
    assert CONFIG["requested_end"] == "20261113"
    assert CONFIG["fetch"]["max_attempts"] == 2
    assert CONFIG["production_enabled"] is CONFIG["real_orders_enabled"] is False
    assert "no retraining" in CONFIG["models"]
    assert "no future-label" in CONFIG["inference_membership"]


def test_last_observed_session_has_a_scheduled_interval_end():
    observed = pd.DatetimeIndex(pd.to_datetime(["2026-11-12", "2026-11-13"]))
    scheduled = pd.DatetimeIndex(pd.to_datetime(["2026-11-13", "2026-11-16", "2026-11-17"]))
    result = execution_calendar(observed, scheduled)
    assert result[:len(observed)].equals(observed)
    assert result[len(observed)] == pd.Timestamp("2026-11-16")
    with pytest.raises(ValueError, match="scheduled"):
        execution_calendar(observed, observed)


def test_merger_conversion_changes_security_not_cash_or_book_equity():
    from finalize_hist_extension import NEW, OLD, convert
    positions = {OLD: {"amount": 2000., "price": 4.}, NEW: {"amount": 50., "price": 5.}}
    result, event = convert(positions, .5, 2.)
    assert OLD in positions and OLD not in result
    assert result[NEW]["amount"] == 360
    assert event["new_actual_shares"] == 620
    assert event["cash_change"] == event["commission"] == 0
    assert sum(v["amount"]*v["price"] for v in result.values()) == 8250
    assert convert({NEW: positions[NEW]}, .5, 2.)[1] is None


def test_merger_fractional_allocation_is_conservative_and_bounded():
    from finalize_hist_extension import OLD, convert
    _, event = convert({OLD: {"amount": 1001., "price": 1.}}, 1., 1.)
    assert event["new_actual_shares"] == 620
    assert 0 < event["fractional_share_not_allocated"] < 1
    with pytest.raises(ValueError, match="fractional"):
        convert({OLD: {"amount": 1001.4, "price": 1.}}, 1., 1.)
