"""Cash-only is a valid position, not permission to ignore absent stock records."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from finalize_hist_rolling import audit_tail, compare_replay_tables


def sample():
    dates = pd.to_datetime(["2025-04-07", "2025-04-08", "2025-04-09"])
    bars = pd.DataFrame({"$close": [10., 10., 10.], "$open": [10., 10., 10.], "$volume": [1e5]*3},
                        index=pd.MultiIndex.from_product([["A"], dates], names=["instrument", "datetime"]))
    account = {"cash": 1000., "A": {"amount": 100.}}
    orders = pd.DataFrame([
        {"date": dates[1], "instrument": "A", "direction": 0, "value": 1000., "cost": 5.,
         "price_adjusted": 10., "filled_adjusted_shares": 100., "lane": "paper_top30_targets"},
        {"date": dates[2], "instrument": "A", "direction": 1, "value": 1000., "cost": 5.,
         "price_adjusted": 10., "filled_adjusted_shares": 100., "lane": "paper_top30_targets"}])
    daily = pd.DataFrame({"account": [2000., 1995., 1990.], "cash": [1000., 1995., 990.]}, index=dates)
    holdings = pd.DataFrame({"date": [dates[0], dates[2]], "instrument": ["A", "A"], "adjusted_shares": [100., 100.]})
    return daily, orders, holdings, account, None, bars


def test_liquidate_cash_day_and_reenter():
    result = audit_tail(*sample())
    assert result["passed"] and result["cash_only_days"] == 1 and result["fills"] == 2
    assert result["max_holding_error_adjusted_shares"] == 0


def test_all_cash_no_orders_or_holdings_columns_is_valid():
    daily, _, _, _, _, bars = sample()
    daily.loc[:, ["cash", "account"]] = 1000.
    assert audit_tail(daily, pd.DataFrame(), pd.DataFrame(), {"cash": 1000.}, None, bars)["cash_only_days"] == 3


def test_absent_nonempty_snapshot_still_fails():
    args = list(sample())
    args[2] = args[2].iloc[1:]
    with pytest.raises(AssertionError, match="Missing nonempty"):
        audit_tail(*args)


def test_cash_only_does_not_hide_wrong_equity():
    args = list(sample())
    args[0].iloc[1, 0] += 10.
    with pytest.raises(AssertionError, match="Cash-only equity"):
        audit_tail(*args)


def test_cash_only_does_not_hide_fake_cash():
    args = list(sample())
    args[0].iloc[1, :] += 10.
    with pytest.raises(AssertionError):
        audit_tail(*args)


def test_cash_only_does_not_hide_phantom_holding():
    args = list(sample())
    args[2] = pd.concat([args[2], pd.DataFrame([{"date": args[0].index[1], "instrument": "A", "adjusted_shares": 1.}])])
    with pytest.raises(AssertionError):
        audit_tail(*args)


def test_duplicate_holding_fails():
    args = list(sample())
    args[2] = pd.concat([args[2], args[2].iloc[:1]])
    with pytest.raises(AssertionError, match="Duplicate"):
        audit_tail(*args)


def test_nonempty_days_equal_original_audit():
    from finalize_hist_extension import audit_tail as old_audit
    args = list(sample())
    args[0] = args[0].iloc[:1]
    args[1] = args[1].iloc[:0]
    args[2] = args[2].iloc[:1]
    new, old = audit_tail(*args), old_audit(*args)
    for key, value in old.items():
        assert new[key] == value


def test_comparison_canonicalizes_order_not_holding_amount():
    held = sample()[2]
    assert compare_replay_tables(held, held.iloc[::-1], "holdings")["canonical_exact"]
    changed = held.copy()
    changed.iloc[0, 2] += .001
    with pytest.raises(AssertionError):
        compare_replay_tables(held, changed, "holdings")


def test_comparison_only_allows_absolute_float_roundoff():
    old = sample()[0]
    new = old.copy()
    new.iloc[0, 0] += 5e-8
    assert compare_replay_tables(old, new, "daily")["absolute_errors"]["account"] < 1e-6
    new.iloc[0, 0] += .001
    with pytest.raises(AssertionError):
        compare_replay_tables(old, new, "daily")


def test_lifetime_turnover_sum_uses_ulp_not_relative_tolerance():
    old = pd.DataFrame({"total_turnover": [1.6e11]})
    new = old.copy()
    new.iloc[0, 0] += .00003
    compare_replay_tables(old, new, "daily")
    new.iloc[0, 0] += .01
    with pytest.raises(AssertionError):
        compare_replay_tables(old, new, "daily")


def test_compensated_turnover_matches_fsum_and_handles_reset():
    import math
    from types import SimpleNamespace

    from repair_hist_rolling_turnover import compensated_turnover
    values = [1e11]+[.01]*100000
    state = SimpleNamespace(to=0.)
    naive = 0.
    for value in values:
        compensated_turnover(state, value)
        naive += value
    assert state.to == math.fsum(values)
    assert abs(naive-state.to) > .01
    state.to = 0.
    compensated_turnover(state, 10.)
    assert state.to == 10.
