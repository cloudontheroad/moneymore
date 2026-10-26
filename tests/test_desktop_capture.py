import ast
import hashlib
import inspect
import json
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from moneymore import desktop_routes
from moneymore.desktop_capture import DesktopCaptureService
from moneymore.desktop_capture_worker import PAGES, CaptureBlocked, capture
from moneymore.desktop_execution import DesktopExecutionService

JOB = "a" * 32
TABLES = {
    "positions": "证券代码\t证券名称\t股票余额\t可用余额\n000001\t测试\t100\t0\n",
    "orders": "证券代码\t证券名称\t委托编号\t操作\t委托数量\t成交数量\t委托价格\t委托时间\n",
    "trades": "证券代码\t证券名称\t委托编号\t成交编号\t操作\t成交数量\t成交价格\t成交时间\n",
}


class FakeClock:
    def __init__(self):
        self.now = 0

    def clock(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeAdapter:
    def __init__(self, fail_page=None):
        self.calls = []
        self.fail_page = fail_page

    def attach(self):
        self.calls.append("attach")

    def export(self, page, destination, check):
        check()
        self.calls.append(page)
        if page == self.fail_page:
            raise CaptureBlocked("FOREGROUND_CHANGED")
        kind = {label: kind for kind, label in PAGES}[page]
        destination.write_bytes(TABLES[kind].encode("gbk"))

    def finish(self, check):
        check()
        self.calls.append("finish")


def test_pure_capture_only_three_readonly_pages_and_fresh_unique_files(tmp_path):
    clock = FakeClock()
    adapter = FakeAdapter()
    result = capture(tmp_path, JOB, adapter, clock=clock.clock, sleep=clock.sleep)
    assert adapter.calls == ["attach", "资金股票", "当日委托", "当日成交", "finish"]
    assert [r["rows"] for r in result["exports"]] == [1, 0, 0]
    assert result["status"] == "CAPTURED_UNVERIFIED" and not result["submitted"]
    assert len(list((tmp_path / "state/private/huatai-acceptance").glob("*.xls"))) == 3
    with pytest.raises(CaptureBlocked, match="FILE_EXISTS"):
        capture(tmp_path, JOB, FakeAdapter(), clock=clock.clock, sleep=clock.sleep)


def test_user_interference_aborts_without_retry_or_fake_empty_results(tmp_path):
    clock = FakeClock()
    adapter = FakeAdapter("当日委托")
    with pytest.raises(CaptureBlocked, match="FOREGROUND_CHANGED"):
        capture(tmp_path, JOB, adapter, clock=clock.clock, sleep=clock.sleep)
    assert adapter.calls == ["attach", "资金股票", "当日委托"]


def test_stop_expired_lease_and_wrong_page_block_before_input(tmp_path):
    stop = tmp_path / f"state/private/desktop-capture/stop-{JOB}"
    stop.parent.mkdir(parents=True)
    stop.touch()
    adapter = FakeAdapter()
    with pytest.raises(CaptureBlocked, match="STOP_REQUESTED"):
        capture(tmp_path, JOB, adapter)
    assert adapter.calls == []
    with pytest.raises(CaptureBlocked, match="INVALID_JOB"):
        capture(tmp_path, "../bad", adapter)


def configured_service(tmp_path, monkeypatch):
    root = tmp_path / "state/private"
    root.mkdir(parents=True)
    (root / "desktop-client.json").write_text('{"executable":"unused-test-path"}', encoding="utf-8")
    monkeypatch.setattr("moneymore.desktop_capture.threading.Thread", lambda **kw: SimpleNamespace(start=lambda: None))
    return DesktopCaptureService(tmp_path)


def test_explicit_start_is_single_flight_and_stop_does_not_submit(tmp_path, monkeypatch):
    service = configured_service(tmp_path, monkeypatch)
    first = service.start()
    assert first["foreground_authorized_once"] and not first["submitted"]
    with pytest.raises(ValueError):
        service.start()
    assert service.stop(first["job_id"])["status"] == "STOP_REQUESTED"
    assert (service.private / f"stop-{first['job_id']}").exists()
    assert service.snapshot()["scheduled"] is False


def test_restart_recovers_abandoned_jobs_without_rerunning_gui(tmp_path, monkeypatch):
    service = configured_service(tmp_path, monkeypatch)
    job = service.start()
    with service._connect() as db:
        db.execute("UPDATE desktop_capture_jobs SET owner_pid=-1 WHERE id=?", (job["job_id"],))
    recovered = DesktopCaptureService(tmp_path)
    assert recovered.snapshot()["jobs"][0]["status"] == "INTERRUPTED"
    assert (recovered.private / f"stop-{job['job_id']}").exists()


def fake_worker_output(tmp_path, job_id):
    date = datetime.now(UTC).astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    folder = tmp_path / "state/private/huatai-acceptance"
    folder.mkdir(parents=True, exist_ok=True)
    exports = []
    for kind, text in TABLES.items():
        name = f"{kind}-{date}-{job_id}.xls"
        data = text.encode("gbk")
        (folder / name).write_bytes(data)
        exports.append({"kind": kind, "filename": name, "sha256": hashlib.sha256(data).hexdigest()})
    return {"status": "CAPTURED_UNVERIFIED", "job_id": job_id, "trade_date": date, "exports": exports}


def test_fixed_worker_success_imports_three_snapshots_without_paper_writes(tmp_path, monkeypatch):
    service = configured_service(tmp_path, monkeypatch)
    job = service.start()
    output = fake_worker_output(tmp_path, job["job_id"])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(output))

    monkeypatch.setattr("moneymore.desktop_capture.subprocess.run", run)
    service._run(job["job_id"])
    report = service.snapshot()["jobs"][0]
    assert report["status"] == "COMPLETED_UNVERIFIED"
    assert [r["row_count"] for r in report["result"]["imports"]] == [1, 0, 0]
    assert calls[0][0][1:] == ["-X", "utf8", "-m", "moneymore.desktop_capture_worker", job["job_id"]]
    assert calls[0][1]["timeout"] == 85
    assert not (tmp_path / "state/paper_orders.sqlite3").exists()


@pytest.mark.parametrize("mode", ["timeout", "malformed", "mismatch", "blocked"])
def test_failed_worker_never_creates_success_or_imports_empty_fallback(tmp_path, monkeypatch, mode):
    service = configured_service(tmp_path, monkeypatch)
    job = service.start()

    def run(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired("fixed worker", 85)
        if mode == "malformed":
            return SimpleNamespace(returncode=0, stdout="not json")
        if mode == "blocked":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "BLOCKED", "reason_code": "FOREGROUND_CHANGED"}))
        output = fake_worker_output(tmp_path, job["job_id"])
        output["exports"][0]["filename"] = "../outside.xls"
        return SimpleNamespace(returncode=0, stdout=json.dumps(output))

    monkeypatch.setattr("moneymore.desktop_capture.subprocess.run", run)
    service._run(job["job_id"])
    assert service.snapshot()["jobs"][0]["status"] == "BLOCKED"
    from moneymore.desktop_exports import DesktopExportStore
    assert DesktopExportStore(tmp_path).snapshot()["snapshots"] == []


def test_route_requires_per_request_foreground_grant(tmp_path, monkeypatch):
    configured_service(tmp_path, monkeypatch)
    service = DesktopExecutionService(tmp_path)
    monkeypatch.setattr(desktop_routes, "service", lambda: service)
    app = FastAPI()
    app.include_router(desktop_routes.router)
    client = TestClient(app)
    headers = {"Origin": "http://127.0.0.1:3000", "X-MoneyMore-Desktop": "prepare-v1"}
    body = {"allow_foreground_once": True, "read_only_only": True}
    assert client.post("/api/desktop-execution/capture", json=body).status_code == 403
    assert client.post("/api/desktop-execution/capture", json={**body, "allow_foreground_once": False}, headers=headers).status_code == 422
    assert client.post("/api/desktop-execution/capture", json=body, headers=headers).json()["submitted"] is False


def test_new_capture_of_identical_contents_is_fresh_observation_not_mutation(tmp_path):
    from moneymore.desktop_exports import DesktopExportStore
    folder = tmp_path / "state/private/huatai-acceptance"
    folder.mkdir(parents=True)
    (folder / "positions-test.xls").write_bytes(TABLES["positions"].encode("gbk"))
    store = DesktopExportStore(tmp_path)
    args = {"filename": "positions-test.xls", "kind": "positions", "account_ref": "test", "trade_date": "20261022"}
    first = store.ingest(**args, collection_job_id="a" * 32)
    assert store.ingest(**args, collection_job_id="a" * 32) == first
    assert store.ingest(**args, collection_job_id="b" * 32)["snapshot_id"] != first["snapshot_id"]
    assert len(store.snapshot()["snapshots"]) == 2


def test_worker_input_surface_is_readonly_and_has_no_blind_enter_or_coordinates():
    from moneymore import desktop_capture_worker
    tree = ast.parse(inspect.getsource(desktop_capture_worker))
    assert {label for _, label in PAGES} == {"资金股票", "当日委托", "当日成交"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"click_input", "click", "press", "start", "launch", "send_chars"}
            if node.func.attr == "type_keys":
                assert isinstance(node.args[0], ast.Constant)
                assert node.args[0].value in {"{F5}", "^s", "{END}{SPACE}{BACKSPACE}{TAB}"}


def test_account_fingerprint_uses_observed_uia_id_without_opening_selector():
    from moneymore.desktop_capture_worker import HuataiForegroundAdapter
    adapter = HuataiForegroundAdapter(None)
    adapter.main = SimpleNamespace(handle=123)
    selector = SimpleNamespace(element_info=SimpleNamespace(automation_id="1004"),
                               selected_text=lambda: "test-account")
    ui = SimpleNamespace(descendants=lambda **kw: [selector])
    adapter.Desktop = lambda **kw: SimpleNamespace(window=lambda **kw: SimpleNamespace(wrapper_object=lambda: ui))
    assert adapter._session_hash() == hashlib.sha256(b"test-account").hexdigest()
    selector.selected_text = lambda: None
    with pytest.raises(CaptureBlocked, match="MASKED"):
        adapter._session_hash()


@pytest.mark.parametrize("class_name,title,width,height,expected", [
    ("Internet Explorer_Hidden", "", 0, 0, True),
    ("Internet Explorer_Hidden", "", 300, 200, False),
    ("Internet Explorer_Hidden", "unexpected prompt", 0, 0, False),
    ("#32770", "", 0, 0, False),
])
def test_only_zero_area_empty_title_broker_helper_is_inert(class_name, title, width, height, expected):
    from moneymore.desktop_capture_worker import inert_broker_helper
    window = SimpleNamespace(class_name=lambda: class_name, window_text=lambda: title,
                             rectangle=lambda: SimpleNamespace(width=lambda: width, height=lambda: height))
    assert inert_broker_helper(window) is expected
