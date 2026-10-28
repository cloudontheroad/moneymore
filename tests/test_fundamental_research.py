import json
import sqlite3

import pandas as pd

from moneymore.data.fundamental_research import (
    TABLE,
    build_financial_research_comparison,
    financial_features_as_of,
    financial_quality,
    sync_research_financials,
    visible_financials,
)
from moneymore.data.store import ParquetStore
from moneymore.research.score_evidence import archive_score_evidence


def frame(symbol="000001.SZ", ann="20260820", period="20260630", growth=10):
    return pd.DataFrame(
        [{"ts_code": symbol, "ann_date": ann, "end_date": period, "roe": 12, "q_sales_yoy": growth}]
    )


def test_visible_announcement_is_delayed():
    data = pd.concat([frame(), frame(ann="20261027", period="20260930")])
    assert visible_financials(data, "20261027").iloc[0].end_date == "20260630"
    assert visible_financials(data, "20261028").iloc[0].end_date == "20260930"


def test_missing_not_silently_zero_filled():
    report = financial_quality(frame(growth=None), ["000001.SZ"], "20261027")
    assert report["gate"] == "BLOCK"
    assert report["complete"] == 0


def test_stale_period_blocks_even_with_recent_announcement():
    report = financial_quality(frame(period="20251231"), ["000001.SZ"], "20261027")
    assert report["complete"] == 1
    assert report["fresh_complete"] == 0
    assert report["stale_symbols"] == ["000001.SZ"]


def test_empty_pool_cannot_pass():
    assert financial_quality(frame(), [], "20261027")["gate"] == "BLOCK"


def test_per_field_uses_previous_published_value_not_future_or_zero():
    data = pd.concat(
        [
            frame(),
            frame(ann="20261026", period="20260930", growth=None),
            frame(ann="20261027", period="20260930", growth=99),
        ]
    )
    selected = financial_features_as_of(data, "20261027").iloc[0]
    assert selected.q_sales_yoy == 10
    assert selected.q_sales_yoy_end_date == "20260630"
    assert selected.roe_end_date == "20260930"
    assert financial_quality(data, ["000001.SZ"], "20261027")["gate"] == "BLOCK"
    assert financial_quality(data, ["000001.SZ"], "20261027", per_field=True)["gate"] == "PASS"


def test_old_per_field_growth_cannot_hide_behind_fresh_roe():
    data = pd.concat(
        [frame(period="20251231"), frame(ann="20261026", period="20260930", growth=None)]
    )
    report = financial_quality(data, ["000001.SZ"], "20261027", per_field=True)
    assert report["complete"] == 1
    assert report["fresh_complete"] == 0


class Provider:
    def __init__(self):
        self.calls = []

    def financial_indicators(self, symbol, start, end):
        self.calls.append(symbol)
        return frame(symbol)


def test_isolated_resumable_and_weekly_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr("moneymore.data.fundamental_research.time.sleep", lambda _: None)
    store = ParquetStore(tmp_path / "data")
    store.merge_curated("fina_indicator", [frame(growth=999)], ["ts_code", "ann_date", "end_date"])
    before = (store.curated / "fina_indicator.parquet").read_bytes()
    provider = Provider()
    symbols = ["000001.SZ", "000002.SZ"]
    r = sync_research_financials(tmp_path, store, provider, symbols, "20261027", max_requests=1)
    assert r["complete"] == 1
    r = sync_research_financials(tmp_path, store, provider, symbols, "20261027", max_requests=1)
    assert r["complete"] == 2
    sync_research_financials(tmp_path, store, provider, symbols, "20261028")
    assert len(provider.calls) == 2
    sync_research_financials(tmp_path, store, provider, symbols, "20261103")
    assert len(provider.calls) == 4
    assert (store.curated / "fina_indicator.parquet").read_bytes() == before
    assert len(store.read(TABLE)) == 2


def test_single_writer_lock(tmp_path):
    directory = tmp_path / "state/fundamental-research-v2"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "sync-lock.sqlite3") as connection:
        connection.execute("BEGIN IMMEDIATE")
        result = sync_research_financials(
            tmp_path, ParquetStore(tmp_path / "data"), Provider(), ["000001.SZ"], "20261027"
        )
    assert result["status"] == "ALREADY_RUNNING"


def test_provider_error_stops_without_exposing_payload(tmp_path):
    class Denied:
        def financial_indicators(self, *args):
            raise RuntimeError("权限不足 credential=private-test-value")

    report = sync_research_financials(
        tmp_path, ParquetStore(tmp_path / "data"), Denied(), ["000001.SZ", "000002.SZ"], "20261027"
    )
    checkpoint = (tmp_path / "state/fundamental-research-v2/checkpoint.json").read_text(
        encoding="utf-8"
    )
    assert report["requests_this_run"] == 1
    assert report["status"] == "PAUSED_ERROR"
    assert "private-test-value" not in checkpoint
    assert not json.loads(checkpoint)["symbols"]
    again = sync_research_financials(
        tmp_path, ParquetStore(tmp_path / "data"), Denied(), ["000001.SZ"], "20261027"
    )
    assert again["status"] == "ERROR_COOLDOWN"
    assert again["requests"] == 0


def test_future_provider_rows_rejected(tmp_path):
    class Future:
        def financial_indicators(self, *args):
            return frame(ann="20261101")

    store = ParquetStore(tmp_path / "data")
    report = sync_research_financials(tmp_path, store, Future(), ["000001.SZ"], "20261027")
    assert report["complete"] == 0
    assert not (store.curated / (TABLE + ".parquet")).exists()


def test_future_announcement_is_quarantined_without_losing_visible_history(tmp_path):
    class Mixed:
        def financial_indicators(self, *args):
            return pd.concat([frame(), frame(ann="20261028", period="20260930")])
    store = ParquetStore(tmp_path / "data")
    report = sync_research_financials(tmp_path, store, Mixed(), ["000001.SZ"], "20261027")
    assert report["complete"] == 1
    assert set(store.read(TABLE).ann_date) == {"20260820"}
    assert set(store.read("fina_indicator_future_quarantine_v2").ann_date) == {"20261028"}


def test_candidate_data_gate_does_not_publish(tmp_path):
    report = build_financial_research_comparison(
        tmp_path, ParquetStore(tmp_path / "data"), ["000001.SZ"], "20261027"
    )
    assert report["status"] == "BLOCKED_DATA"
    assert not report["orders_enabled"]
    assert not (tmp_path / "data/processed/global_factor_recommendation.parquet").exists()


def test_score_archive_immutable_and_exposes_missing_controls(tmp_path):
    f = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("20261026"),
                "symbol": "000001.SZ",
                "score": 0.2,
                "active_weight": 0.5,
                "roe": None,
            }
        ]
    )
    policy = {"factors": {"roe": 1}}
    first = archive_score_evidence(tmp_path, {"processed_factors": f, "raw_factors": f}, f, policy)
    second = archive_score_evidence(tmp_path, {"processed_factors": f, "raw_factors": f}, f, policy)
    assert first["path"] != second["path"]
    assert first["files"] == second["files"]
    assert not first["size_control_present"]
    assert first["data_gate"] == "INCOMPLETE_FACTORS"
