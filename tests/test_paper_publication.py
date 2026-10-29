import json
import sqlite3
from pathlib import Path

import pytest
from test_paper_release_jobs import make_setup

import moneymore.research.paper_publication as publication_module
from moneymore.execution.paper import PaperBroker
from moneymore.research.baseline_v2_release import digest
from moneymore.research.paper_publication import PaperPublication
from moneymore.research.paper_release_jobs import PaperReleaseJobs


@pytest.fixture
def ready(tmp_path):
    _, _, request, now = make_setup(tmp_path)
    directory = Path(request["release_directory"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = Path(publication_module.__file__).resolve()
    manifest["input_hashes"][str(source)] = digest(source)
    manifest_path.write_text(json.dumps(manifest))
    contract_path = directory / "provisional-paper-contract.json"
    contract = json.loads(contract_path.read_text())
    contract.update(
        publication_version="SYNTHETIC_V2",
        account_id="baseline_v2:SYNTHETIC_V2",
        initial_cash=100000,
        manifest_sha256=digest(manifest_path),
    )
    contract_path.write_text(json.dumps(contract))
    for name in manifest["protocol"]["required_stages"][:-1]:
        (directory / (name + ".json")).write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "manifest_sha256": digest(manifest_path),
                    "artifact_hashes": {str(contract_path): digest(contract_path)},
                }
            )
        )
    request.update(account_id=contract["account_id"], contract_sha256=digest(contract_path))
    return PaperPublication(tmp_path), request, now


def test_preparation_creates_account_and_disabled_tasks_once_not_published(ready):
    publisher, request, now = ready
    result = publisher.prepare(request, now=now)
    assert result["state"] == "PREPARED" and result["published"] is False
    assert publisher.prepare(request, now=now) == result
    with sqlite3.connect(result["database"]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*),SUM(enabled) FROM v2_publication_outbox"
        ).fetchone() == (3, 0)


def test_failed_gate_does_not_even_create_publication_database(ready):
    publisher, request, now = ready
    path = Path(request["release_directory"]) / "robustness.json"
    report = json.loads(path.read_text())
    report["status"] = "FAIL_REAL_STATISTICAL_GATE"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="HARD_GATES_NOT_PASSED"):
        publisher.prepare(request, now=now)
    assert not (
        publisher.root / "state/baseline-v2-release/paper-epochs/SYNTHETIC_V2.sqlite3"
    ).exists()


def test_account_and_outbox_failure_roll_back_together(ready):
    publisher, request, now = ready
    _, database = publisher.preflight(request)
    PaperBroker(database)
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE v2_publication_outbox(account_id TEXT,phase TEXT,enabled INTEGER)"
        )
        conn.execute(
            "CREATE TRIGGER synthetic_failure BEFORE INSERT ON v2_publication_outbox BEGIN SELECT RAISE(ABORT,'SYNTHETIC_DISK_STEP_FAILURE'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        publisher.prepare(request, now=now)
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM v2_publications").fetchone()[0] == 0


def test_rollback_cancels_pending_preserves_history_and_blocks_jobs(ready):
    publisher, request, now = ready
    result = publisher.prepare(request, now=now)
    account = request["account_id"]
    from moneymore.config import BacktestConfig
    from moneymore.execution.paper import ExecutionBar
    from moneymore.execution.risk import OrderIntent, RiskResult
    from moneymore.models import Side

    broker = PaperBroker(result["database"])
    broker.submit(
        RiskResult(
            True,
            OrderIntent("filled", "SYNTHETIC", "AAA", Side.BUY, 100, "20261027", "SYNTHETIC"),
            None,
        ),
        account,
    )
    config = BacktestConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    broker.execute_pending(ExecutionBar("AAA", "20261028", 10, 10), config, account)
    history = broker.fills(account_id=account)
    assert len(history) == 1
    with sqlite3.connect(result["database"]) as conn:
        conn.execute(
            "INSERT INTO orders VALUES ('pending','now','PENDING','SYNTHETIC','AAA','BUY',100,'20261027','SYNTHETIC',?)",
            (account,),
        )
        before = conn.execute("SELECT initial_cash,cash FROM accounts").fetchone()
        previous_attempts = conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone()[0]
    # Pause works even after evidence is invalidated; safety must not need a PASS.
    (Path(request["release_directory"]) / "robustness.json").write_text("{}")
    receipt = publisher.rollback("SYNTHETIC_V2", reason="SYNTHETIC_OPERATOR_ROLLBACK")
    assert receipt["cancelled_pending"] == 1 and receipt["history_deleted"] is False
    with sqlite3.connect(result["database"]) as conn:
        assert conn.execute("SELECT initial_cash,cash FROM accounts").fetchone() == before
        assert (
            conn.execute("SELECT status FROM orders WHERE idempotency_key='pending'").fetchone()[0]
            == "CANCELLED"
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM execution_attempts").fetchone()[0]
            == previous_attempts + 1
        )
    assert broker.fills(account_id=account) == history
    assert broker.reconcile(account).matched
    with pytest.raises(ValueError, match="PUBLICATION_PAUSED"):
        PaperReleaseJobs(publisher.root, result["database"]).run(
            request, lambda _: pytest.fail("must stay paused"), now=now
        )
    assert publisher.rollback("SYNTHETIC_V2", reason="repeat")["cancelled_pending"] == 0
