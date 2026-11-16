import sqlite3
from types import SimpleNamespace
import pandas as pd
import pytest
from moneymore.research.replay_actions import account_replay_actions


def test_old_missing_payment_does_not_block_new_account(tmp_path):
    db = tmp_path / 'ledger.sqlite3'
    with sqlite3.connect(db) as c:
        c.executescript('CREATE TABLE positions(account_id,symbol); CREATE TABLE fills(account_id,trade_date); CREATE TABLE corporate_entitlements(account_id,symbol,record_date);')
        c.execute("INSERT INTO positions VALUES('a','600809.SH')")
        c.execute("INSERT INTO fills VALUES('a','20261105')")
    event = dict(ts_code='600809.SH',div_proc='实施',cash_div_tax=1.,stk_div=0.,stk_bo_rate=0.,stk_co_rate=0.,record_date='19950515',ex_date='19950516',pay_date=None,div_listdate=None,imp_ann_date='19950501',end_date='19941231')
    store = SimpleNamespace(read=lambda _:pd.DataFrame([event]))
    broker = SimpleNamespace(database=db)
    assert account_replay_actions(store,broker,'a','20261110').empty
    event.update(record_date='20261109',ex_date='20261110',imp_ann_date='20261101')
    with pytest.raises(ValueError,match='SETTLEMENT_DATE'):
        account_replay_actions(store,broker,'a','20261110')
    event.update(record_date='20261112',ex_date='20261113')
    assert account_replay_actions(store,broker,'a','20261110').empty
