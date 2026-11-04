import json
import pytest
from moneymore.execution.isolation import isolated_execution_call
from moneymore.execution.paper import PaperBroker
from moneymore.intraday_execution import prepare_intraday_branch_orders
from moneymore.multi_sector_daily import MULTI_SECTOR_ACCOUNT


def test_one_failed_group_does_not_prevent_next_group():
    calls = []
    def failed():
        raise ValueError("bad independent target")
    results = [isolated_execution_call("bad", failed),
               isolated_execution_call("good", lambda: calls.append("executed"))]
    assert calls == ["executed"]
    assert results[0]["status"] == "EXECUTION_ERROR"
    assert results[1]["status"] == "COMPLETED"


@pytest.mark.parametrize("status,planning_only,valid", [
    ("PLAN_READY", True, True), ("PLAN_READY", False, False),
    ("FAILED", True, False), ("COMPLETED", False, True),
])
def test_branch_accepts_only_valid_plan_state(tmp_path, status, planning_only, valid):
    broker = PaperBroker(tmp_path / "paper.sqlite3")
    broker.initialize_account(100000, MULTI_SECTOR_ACCOUNT)
    folder = tmp_path / "multi-sector-shadow"
    folder.mkdir()
    (folder / "20261104.json").write_text(json.dumps({
        "status": status, "planning_only": planning_only,
        "target_weights": {"600036.SH": .1},
    }))
    if valid:
        assert prepare_intraday_branch_orders(broker, "20261105") == 0
    else:
        with pytest.raises(RuntimeError, match="invalid target report"):
            prepare_intraday_branch_orders(broker, "20261105")


@pytest.mark.parametrize("error,expected", [
    ("RuntimeError: independent strategy failures: old_model: HELD_RISK_MISSING", ["20261102", "20261103", "20261104"]),
    ("DataQualityError: daily is empty", ["20261102"]),
])
def test_recovery_only_isolates_strategy_failures(tmp_path, monkeypatch, error, expected):
    from moneymore.web import TaskService
    import moneymore.web as web
    import moneymore.close_valuation as valuation
    service = TaskService(tmp_path / "service.sqlite3")
    monkeypatch.setattr(web, "DATA", tmp_path / "data")
    monkeypatch.setattr(web, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(valuation, "recover_close_valuations", lambda *a: {"failures": []})
    monkeypatch.setattr(service, "_run_step", lambda *a, **k: (True, None))
    monkeypatch.setattr(service, "_recovery_dates", lambda *a: ["20261102", "20261103", "20261104"])
    visited = []
    def execute(run, day, historical_recovery=False):
        visited.append(day)
        service._finish(run, "FAILED" if day == "20261102" else "COMPLETED", error=error if day == "20261102" else None)
    monkeypatch.setattr(service, "_execute", execute)
    parent = service._create_run("recovery_pipeline", "20261104", "TEST")
    service._execute_recovery(parent, "20261104")
    assert visited == expected
    assert next(r for r in service.runs(100) if r["id"] == parent)["status"] == "FAILED"
