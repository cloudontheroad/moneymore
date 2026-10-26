import os
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from moneymore import desktop_routes
from moneymore.desktop_execution import DesktopExecutionService
from moneymore.desktop_exports import DesktopExportStore

POSITIONS = "证券代码\t证券名称\t股票余额\t可用余额\n000001\t虚构证券\t100\t0\n"


def export_file(tmp_path, filename="positions-test.xls", text=POSITIONS):
    folder = tmp_path / "state/private/huatai-acceptance"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_bytes(text.encode("gbk"))
    return path


def ingest(store, **kw):
    return store.ingest(filename=kw.get("filename", "positions-test.xls"),
                        kind=kw.get("kind", "positions"), account_ref="test_local",
                        trade_date=kw.get("trade_date", "20261022"))


def test_persistent_idempotent_snapshot_is_not_broker_fill(tmp_path):
    export_file(tmp_path)
    store = DesktopExportStore(tmp_path)
    first = ingest(store)
    assert first == ingest(store)
    assert first["row_count"] == 1 and not first["ledger_updated"]
    snapshot = DesktopExportStore(tmp_path).snapshot()
    assert len(snapshot["snapshots"]) == 1
    row = snapshot["snapshots"][0]["rows"][0]
    assert row["code"] == "000001" and row["available_quantity"] == 0
    assert row["strategy_id"] is None
    assert not snapshot["live"] and not snapshot["automated_gui_collection"]
    assert not snapshot["snapshots"][0]["broker_receipts_verified"]


def test_old_file_is_stale_even_after_import_and_retry(tmp_path):
    path = export_file(tmp_path)
    os.utime(path, (1_600_000_000, 1_600_000_000))
    store = DesktopExportStore(tmp_path)
    ingest(store)
    before = store.snapshot()["snapshots"][0]
    ingest(store)
    after = store.snapshot()["snapshots"][0]
    assert after["imported_at"] == before["imported_at"]
    assert after["freshness"] == "STALE"
    assert after["source_mtime"].startswith("2020-")


@pytest.mark.parametrize("name", ["../positions.xls", "C:\\positions.xls", "history.xls", "positions.exe"])
def test_no_arbitrary_file_reads(tmp_path, name):
    store = DesktopExportStore(tmp_path)
    with pytest.raises(ValueError):
        ingest(store, filename=name)


def test_missing_malformed_wrong_kind_and_oversize_leave_no_snapshots(tmp_path):
    store = DesktopExportStore(tmp_path)
    with pytest.raises(ValueError):
        ingest(store)
    export_file(tmp_path, text="not a table")
    with pytest.raises(ValueError):
        ingest(store)
    export_file(tmp_path)
    with pytest.raises(ValueError):
        ingest(store, kind="trades")
    export_file(tmp_path, text="x" * 5_000_001)
    with pytest.raises(ValueError):
        ingest(store)
    assert store.snapshot()["snapshots"] == []


def test_immutable_snapshot_does_not_touch_paper_ledger(tmp_path):
    path = tmp_path / "state/paper_orders.sqlite3"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not even a valid sqlite database")
    before = path.read_bytes()
    export_file(tmp_path)
    ingest(DesktopExportStore(tmp_path))
    assert path.read_bytes() == before


def test_empty_export_is_observation_not_verified_empty_account(tmp_path):
    export_file(tmp_path, text=POSITIONS.splitlines()[0])
    store = DesktopExportStore(tmp_path)
    assert ingest(store)["row_count"] == 0
    item = store.snapshot()["snapshots"][0]
    assert not item["account_verified"] and not item["date_verified"]
    assert item["status"] == "PARSED_UNVERIFIED"


def test_export_import_route_needs_local_explicit_action(tmp_path, monkeypatch):
    export_file(tmp_path)
    service = DesktopExecutionService(tmp_path)
    monkeypatch.setattr(desktop_routes, "service", lambda: service)
    app = FastAPI()
    app.include_router(desktop_routes.router)
    client = TestClient(app)
    body = {"filename": "positions-test.xls", "kind": "positions",
            "account_ref": "test_local", "trade_date": "20261022"}
    assert client.post("/api/desktop-execution/exports/import", json=body).status_code == 403
    headers = {"Origin": "http://127.0.0.1:3000", "X-MoneyMore-Desktop": "prepare-v1"}
    result = client.post("/api/desktop-execution/exports/import", json=body, headers=headers)
    assert result.status_code == 200 and not result.json()["submitted"]
    assert client.get("/api/desktop-execution").json()["broker_exports"]["snapshots"][0]["row_count"] == 1
    assert client.post("/api/desktop-execution/submit", json=body, headers=headers).status_code == 404
    with sqlite3.connect(service.database) as db:
        assert db.execute("SELECT COUNT(*) FROM desktop_intents").fetchone()[0] == 0


def test_historical_and_unrelated_files_not_exposed_as_live_candidates(tmp_path):
    export_file(tmp_path)
    export_file(tmp_path, filename="history.xls")
    result = DesktopExportStore(tmp_path).candidates()
    assert [r["filename"] for r in result] == ["positions-test.xls"]
