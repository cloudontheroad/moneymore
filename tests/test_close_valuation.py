import sqlite3

import pandas as pd
import pytest

from moneymore.close_valuation import recover_close_valuations, snapshot_as_of
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker


def fixture(tmp_path):
    db = tmp_path / "state/paper_orders.sqlite3"
    db.parent.mkdir()
    broker = PaperBroker(db)
    broker.initialize_account(1000, "example")
    store = ParquetStore(tmp_path / "data")
    store.merge_curated("trade_calendar", [pd.DataFrame({"cal_date": ["20261102", "20261103", "20261104"], "is_open": [1, 1, 1]})], ["cal_date"])
    store.merge_curated("qlib_candidate_account_daily", [pd.DataFrame([dict(account_id="example", trade_date="20261102", cash=1000., equity=1000., market_value=0.)])], ["account_id", "trade_date"])
    store.merge_curated("daily", [pd.DataFrame({"ts_code": ["000001.SZ", "000001.SZ"], "trade_date": ["20261103", "20261104"], "close": [10., 11.]})], ["ts_code", "trade_date"])
    return store, db


def test_asof_reverses_later_fill_and_fee(tmp_path):
    _, db = fixture(tmp_path)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE accounts SET cash=795 WHERE account_id='example'")
        c.execute("INSERT INTO positions(account_id,symbol,quantity,available_quantity,avg_cost) VALUES ('example','000001.SZ',20,20,10)")
        c.execute("INSERT INTO fills(account_id,idempotency_key,symbol,side,quantity,price,fee,trade_date) VALUES ('example','test','000001.SZ','BUY',20,10,5,'20261104')")
        assert snapshot_as_of(c, "example", "20261103") == (1000., {})
        assert snapshot_as_of(c, "example", "20261104") == (795., {"000001.SZ": 20})


def test_repair_is_idempotent_and_writes_no_trades(tmp_path):
    store, db = fixture(tmp_path)
    result = recover_close_valuations(tmp_path, "20261104", store)
    assert result["status"] == "COMPLETED" and len(result["updated"]) == 2
    history = store.read("qlib_candidate_account_daily").sort_values("trade_date")
    assert history.equity.tolist() == [1000., 1000., 1000.]
    assert history.iloc[0].trade_date == "20261102"
    assert recover_close_valuations(tmp_path, "20261104", store)["updated"] == []
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_unknown_missing_price_is_not_flat_filled(tmp_path):
    store, db = fixture(tmp_path)
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO positions(account_id,symbol,quantity,available_quantity,avg_cost) VALUES ('example','MISSING.SZ',20,20,10)")
    result = recover_close_valuations(tmp_path, "20261104", store)
    assert len(result["failures"]) == 2 and result["updated"] == []
    assert len(store.read("qlib_candidate_account_daily")) == 1
