from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moneymore.leveraged_etf import ACCOUNT_ID, INITIAL_CASH, LeveragedEtfPaper


def _market(direction: float = 0.001) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=700).strftime("%Y-%m-%d")
    qqq_close = 400 * np.cumprod(np.full(len(dates), 1 + direction))
    result: dict[str, pd.DataFrame] = {}
    for symbol, close in {
        "QQQ": qqq_close,
        "TQQQ": 50 * np.cumprod(np.full(len(dates), 1 + direction * 3)),
        "SQQQ": 40 * np.cumprod(np.full(len(dates), 1 - direction * 3)),
        "^VIX": np.full(len(dates), 14.0),
        "^VIX3M": np.full(len(dates), 16.0),
        "^TNX": np.full(len(dates), 4.0),
    }.items():
        result[symbol] = pd.DataFrame(
            {
                "trade_date": dates,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.full(len(dates), 1_000_000),
            }
        )
    return result


class StaticClient:
    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self.data = data

    def history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return self.data[symbol].copy()


def test_bull_signal_queues_tqqq_in_isolated_account(tmp_path: Path) -> None:
    database = tmp_path / "us-etf.sqlite3"
    paper = LeveragedEtfPaper(database, tmp_path / "cache")

    snapshot = paper.run(StaticClient(_market()))

    assert snapshot["account_id"] == ACCOUNT_ID
    assert snapshot["initial_cash"] == INITIAL_CASH
    assert snapshot["equity"] == INITIAL_CASH
    assert snapshot["latest_signal"]["regime"] in {"BALANCED", "RISK_ON"}
    assert snapshot["latest_signal"]["tqqq"] > 0
    assert {row["symbol"] for row in snapshot["orders"]} == {"TQQQ"}
    assert all(row["status"] == "PENDING" for row in snapshot["orders"])


def test_same_session_is_idempotent(tmp_path: Path) -> None:
    paper = LeveragedEtfPaper(tmp_path / "us-etf.sqlite3", tmp_path / "cache")
    client = StaticClient(_market())

    first = paper.run(client)
    second = paper.run(client)

    assert len(second["orders"]) == len(first["orders"])
    assert second["cash"] == first["cash"]
    assert second["fills"] == []


def test_cycle_layer_holds_inside_rebalance_band() -> None:
    calm = _market()
    raw = LeveragedEtfPaper.allocation(calm)
    held = LeveragedEtfPaper.allocation(calm, previous_tqqq=raw.tqqq + 0.04)

    assert held.tqqq == pytest.approx(raw.tqqq + 0.04)
    assert held.sqqq == 0.0
    assert held.no_trade_band == 0.12


def test_outage_replays_every_missing_us_session(tmp_path: Path) -> None:
    paper = LeveragedEtfPaper(tmp_path / "us-etf.sqlite3", tmp_path / "cache")
    full = _market()
    partial = {symbol: frame.iloc[:-5].copy() for symbol, frame in full.items()}

    first = paper.run(StaticClient(partial))
    caught_up = paper.run(StaticClient(full))

    assert first["latest_signal"]["signal_date"] == str(partial["QQQ"].iloc[-1]["trade_date"])
    assert caught_up["latest_signal"]["signal_date"] == str(full["QQQ"].iloc[-1]["trade_date"])
    assert len(caught_up["history"]) == 6
    assert caught_up["fills"]
    assert caught_up["positions"]
    assert caught_up["benchmark_history"][0]["normalized_nav"] == pytest.approx(1.0)
    latest_date = str(full["TQQQ"].iloc[-1]["trade_date"])
    year_start = f"{latest_date[:4]}-01-01"
    first_year_price = float(
        full["TQQQ"].loc[full["TQQQ"]["trade_date"] >= year_start].iloc[0]["close"]
    )
    expected = float(full["TQQQ"].iloc[-1]["close"]) / first_year_price
    assert caught_up["benchmark_history"][-1]["normalized_nav"] == pytest.approx(expected)
    assert caught_up["backtest_history"][0]["trade_date"] == year_start
    assert caught_up["backtest_history"][0]["normalized_nav"] == pytest.approx(1.0)


def test_storage_is_not_the_moneymore_paper_broker_schema(tmp_path: Path) -> None:
    database = tmp_path / "us-etf.sqlite3"
    LeveragedEtfPaper(database, tmp_path / "cache")

    import sqlite3

    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "account" in tables
    assert "accounts" not in tables
    assert "rejections" not in tables


def test_display_replay_matches_next_open_ledger(tmp_path: Path) -> None:
    paper = LeveragedEtfPaper(tmp_path / "us-etf.sqlite3", tmp_path / "cache")
    market = _market()
    first = len(market["QQQ"]) - 8
    initial_market = {symbol: frame.iloc[:first + 1] for symbol, frame in market.items()}
    paper.run(StaticClient(initial_market))
    snapshot = paper.run(StaticClient(market))

    targets = paper._backtest_targets(market, start_index=first)
    replay = paper.replay(market, targets, list(range(first, first + 8)), "BASELINE")
    assert [row["equity"] for row in replay[1:]] == pytest.approx(
        [row["equity"] for row in snapshot["history"]]
    )
    assert sum(row["fill_count"] for row in replay[1:]) == len(snapshot["fills"])
    assert snapshot["version_segments"][0]["version"] == paper.strategy_id
    benchmark = snapshot["comparison_benchmark"]
    assert benchmark[0]["normalized_nav"] == 1.0
    assert sum(row.get("fill_count", 0) for row in benchmark) == 1
