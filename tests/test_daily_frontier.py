from unittest.mock import patch

import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.data.sync import SyncSummary, sync_daily_frontier


def seeded(tmp_path, adj_date="20261103"):
    store = ParquetStore(tmp_path)
    store.merge_curated("daily", [pd.DataFrame({"trade_date": ["20261103"]})], ["trade_date"])
    store.merge_curated("adj_factor", [pd.DataFrame({"trade_date": [adj_date]})], ["trade_date"])
    store.merge_curated("trade_calendar", [pd.DataFrame({
        "cal_date": ["20261103", "20261104", "20261105", "20261106"],
        "is_open": [1, 1, 1, 1],
    })], ["cal_date"])
    return store


def test_frontier_advances_without_strategy_completion(tmp_path):
    store = seeded(tmp_path)
    provider = object()
    with patch("moneymore.data.sync.sync_daily_history", return_value=SyncSummary(1, 10, 10)) as sync:
        assert sync_daily_frontier(provider, store, "20261104").open_days == 1
        sync.assert_called_once_with(provider, store, "20261104", "20261104", batch_days=5)


def test_cutoff_does_not_fetch_current_intraday_session(tmp_path):
    store = seeded(tmp_path)
    with patch("moneymore.data.sync.sync_daily_history") as sync:
        assert sync_daily_frontier(object(), store, "20261103").open_days == 0
        sync.assert_not_called()


def test_factor_lag_and_bounded_backlog(tmp_path):
    store = seeded(tmp_path, "20261102")
    provider = object()
    with patch("moneymore.data.sync.sync_daily_history", return_value=SyncSummary(2, 0, 0)) as sync:
        sync_daily_frontier(provider, store, "20261106", max_sessions=2)
        sync.assert_called_once_with(provider, store, "20261103", "20261104", batch_days=5)
