import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from moneymore import desktop_routes
from moneymore.desktop_execution import DesktopExecutionService
from moneymore.desktop_readonly_worker import probe


def ledger(root: Path):
    path = root / "state/paper_orders.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE orders (account_id TEXT,idempotency_key TEXT,strategy_id TEXT,"
            "symbol TEXT,side TEXT,quantity INTEGER,signal_date TEXT,status TEXT)"
        )
        db.executemany(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)",
            [
                ("baseline", "one", "factor", "600036.SH", "BUY", 100, "20261021", "PENDING"),
                ("challenger", "two", "gru", "600036.SH", "BUY", 200, "20261021", "PENDING"),
            ],
        )
    return path


def test_prepare_isolated_idempotent_no_broker_fills(tmp_path):
    source = ledger(tmp_path)
    before = source.read_bytes()
    service = DesktopExecutionService(tmp_path)
    first = service.prepare("baseline")
    assert service.prepare("baseline") == first
    snapshot = service.snapshot()
    assert len(snapshot["intents"]) == 1
    assert snapshot["intents"][0]["quantity"] == 100
    assert snapshot["intents"][0]["broker_order_id"] is None
    assert snapshot["intents"][0]["confirmed_filled_quantity"] is None
    assert not snapshot["live_submission_supported"]
    assert source.read_bytes() == before
    with pytest.raises(ValueError):
        service.prepare("unknown")


def test_probe_only_fixed_worker_and_no_console(tmp_path, monkeypatch):
    service = DesktopExecutionService(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "INSPECTED"}))

    monkeypatch.setattr("moneymore.desktop_execution.subprocess.run", run)
    result = service.probe()
    assert result["status"] == "INSPECTED"
    assert calls[0][0][1:] == ["-X", "utf8", "-m", "moneymore.desktop_readonly_worker"]
    assert calls[0][1]["timeout"] == 25
    assert service.snapshot()["jobs"][0]["status"] == "INSPECTED"


def test_worker_missing_config_does_not_import_or_control_desktop(tmp_path):
    assert probe(tmp_path)["status"] == "BLOCKED"


@pytest.mark.parametrize("output", [None, b"not json", b"[]"])
def test_bad_worker_output_is_blocked_not_stuck_running(tmp_path, monkeypatch, output):
    service = DesktopExecutionService(tmp_path)
    monkeypatch.setattr(
        "moneymore.desktop_execution.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=output),
    )
    assert service.probe()["status"] == "BLOCKED"
    assert service.snapshot()["jobs"][0]["status"] == "BLOCKED"


def test_changed_source_invalidates_prepared_intent(tmp_path):
    source = ledger(tmp_path)
    service = DesktopExecutionService(tmp_path)
    service.prepare("baseline")
    with sqlite3.connect(source) as db:
        db.execute("UPDATE orders SET quantity=300 WHERE account_id='baseline'")
    assert service.snapshot()["intents"][0]["status"] == "SOURCE_STALE"
    service.prepare("baseline")
    assert len(service.snapshot()["intents"]) == 2


def test_web_mutations_require_local_origin_and_explicit_header(tmp_path, monkeypatch):
    ledger(tmp_path)
    service = DesktopExecutionService(tmp_path)
    monkeypatch.setattr(desktop_routes, "service", lambda: service)
    app = FastAPI()
    app.include_router(desktop_routes.router)
    client = TestClient(app)
    assert client.get("/api/desktop-execution").status_code == 200
    payload = {"source_account": "baseline"}
    assert client.post("/api/desktop-execution/prepare", json=payload).status_code == 403
    assert (
        client.post(
            "/api/desktop-execution/prepare",
            json=payload,
            headers={"Origin": "https://evil.example", "X-MoneyMore-Desktop": "prepare-v1"},
        ).status_code
        == 403
    )
    result = client.post(
        "/api/desktop-execution/prepare",
        json=payload,
        headers={"Origin": "http://127.0.0.1:3000", "X-MoneyMore-Desktop": "prepare-v1"},
    )
    assert result.status_code == 200 and result.json()["submitted"] is False
    assert client.post("/api/desktop-execution/submit", json={}).status_code == 404


def test_worker_has_no_foreground_or_order_primitives():
    import ast
    import inspect

    from moneymore import desktop_readonly_worker

    tree = ast.parse(inspect.getsource(desktop_readonly_worker))
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not calls & {
        "click",
        "click_input",
        "set_focus",
        "type_keys",
        "set_edit_text",
        "invoke",
        "submit",
        "press_key",
        "SetForegroundWindow",
    }
