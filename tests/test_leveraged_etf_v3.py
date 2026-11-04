from __future__ import annotations

from pathlib import Path

from test_leveraged_etf import StaticClient, _market

from moneymore.leveraged_etf_v3 import (
    V3_ACCOUNT_ID,
    V3_INITIAL_CASH,
    V3_STRATEGY_ID,
    LeveragedEtfV3Paper,
)


def test_v3_uses_a_fully_isolated_paper_account(tmp_path: Path) -> None:
    database = tmp_path / "v3.sqlite3"
    paper = LeveragedEtfV3Paper(database, tmp_path / "cache")
    snapshot = paper.run(StaticClient(_market()))

    assert snapshot["account_id"] == V3_ACCOUNT_ID
    assert snapshot["strategy_id"] == V3_STRATEGY_ID
    assert snapshot["initial_cash"] == V3_INITIAL_CASH
    assert snapshot["latest_signal"]["sqqq"] == 0.0
    assert snapshot["latest_signal"]["indicators"]["model_version"] == V3_STRATEGY_ID


def test_v3_and_v2_databases_never_overlap(tmp_path: Path) -> None:
    from moneymore.leveraged_etf import LeveragedEtfPaper

    market = StaticClient(_market())
    v2 = LeveragedEtfPaper(tmp_path / "v2.sqlite3", tmp_path / "cache-v2")
    v3 = LeveragedEtfV3Paper(tmp_path / "v3.sqlite3", tmp_path / "cache-v3")
    v2.run(market)
    v3.run(market)

    assert v2.database != v3.database
    assert v2.snapshot()["account_id"] != v3.snapshot()["account_id"]
