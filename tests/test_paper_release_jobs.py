import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import moneymore.research.paper_release_jobs as job_module
import moneymore.research.paper_risk_epoch as epoch_module
from moneymore.execution.paper import PaperBroker
from moneymore.research.baseline_v2_release import digest
from moneymore.research.paper_release_jobs import PaperReleaseJobs


def make_setup(tmp_path):
    directory = tmp_path / "state/baseline-v2-release/runs/SYNTHETIC_TEST_ONLY"
    directory.mkdir(parents=True)
    database = tmp_path / "state/baseline-v2-release/paper-epochs/SYNTHETIC.sqlite3"
    broker = PaperBroker(database)
    account = "baseline_v2:synthetic"
    broker.initialize_account(100000, account)
    stages = [
        "attribution",
        "full_union_economics",
        "financial_vintages",
        "robustness",
        "execution_stress",
        "parity",
        "forward_observation",
    ]
    manifest = {
        "protocol": {
            "required_stages": stages,
            "real_orders_enabled": False,
            "forward_review_days": 60,
        },
        "input_hashes": {
            str(Path(m.__file__).resolve()): digest(m.__file__) for m in (job_module, epoch_module)
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sha = digest(directory / "manifest.json")
    contract = directory / "provisional-paper-contract.json"
    contract.write_text(
        json.dumps(
            {
                "account_id": account,
                "manifest_sha256": sha,
                "paper_activation_authorized": True,
                "orders_enabled": False,
                "real_orders_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    for name in stages[:-1]:
        (directory / (name + ".json")).write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "manifest_sha256": sha,
                    "artifact_hashes": {str(contract): digest(contract)},
                }
            ),
            encoding="utf-8",
        )
    evidence = tmp_path / "SYNTHETIC-INPUT.json"
    evidence.write_text('{"test_only":true}')
    request = {
        "account_id": account,
        "release_directory": str(directory),
        "contract_sha256": digest(contract),
        "trade_date": "20261028",
        "phase": "OPEN_EXECUTION",
        "input_artifact": str(evidence),
        "input_sha256": digest(evidence),
    }
    return PaperReleaseJobs(tmp_path, database), broker, request, datetime(2026, 10, 29, tzinfo=UTC)


@pytest.fixture
def setup(tmp_path):
    return make_setup(tmp_path)


def test_done_job_is_idempotent_and_historical_recovery_is_not_prospective(setup):
    jobs, _, request, now = setup
    first = jobs.run(request, lambda key: {"key": key}, now=now)
    assert jobs.run(request, lambda _: pytest.fail("must not execute again"), now=now) == first
    assert first["historical_recovery"] is True
    assert first["counts_as_prospective_observation"] is False


def test_changed_payload_and_tampered_input_block(setup):
    jobs, _, request, now = setup
    jobs.run(request, lambda _: {}, now=now)
    with pytest.raises(ValueError, match="PAYLOAD_CHANGED"):
        jobs.run({**request, "unexpected_new_plan": True}, lambda _: {}, now=now)
    Path(request["input_artifact"]).write_text("changed")
    with pytest.raises(ValueError, match="INPUT_EVIDENCE_CHANGED"):
        jobs.run(request, lambda _: {}, now=now)


def test_actual_failed_release_never_invokes_callback(setup):
    jobs, _, request, now = setup
    path = Path(request["release_directory"]) / "robustness.json"
    value = json.loads(path.read_text())
    value["status"] = "FAIL_SIGNAL_ROBUSTNESS"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="HARD_GATES_NOT_PASSED"):
        jobs.run(request, lambda _: pytest.fail("failed gates must block"), now=now)


def test_nested_worker_cannot_overlap_even_after_expiry(setup):
    jobs, _, request, now = setup

    def callback(_):
        with pytest.raises(ValueError, match="ALREADY_RUNNING"):
            jobs.run(request, lambda _: {}, now=now.replace(hour=2))
        return {}

    jobs.run(request, callback, now=now, lease_seconds=1)


def test_callback_failure_retains_record_and_retries_same_operation_key(setup):
    jobs, _, request, now = setup
    keys = []

    def fail(key):
        keys.append(key)
        raise RuntimeError("SYNTHETIC_FAILURE")

    with pytest.raises(RuntimeError):
        jobs.run(request, fail, now=now)
    jobs.run(request, lambda key: keys.append(key), now=now)
    assert keys[0] == keys[1]
    with sqlite3.connect(jobs.database) as conn:
        assert conn.execute("SELECT status,attempts FROM v2_jobs").fetchone() == ("DONE", 2)


def test_process_dies_after_fill_recovery_does_not_duplicate_fill(setup, tmp_path):
    jobs, broker, request, now = setup
    request_path = tmp_path / "synthetic-request.json"
    request_path.write_text(json.dumps(request))
    code = """
import json,os,sys
from datetime import UTC,datetime
from moneymore.research.paper_release_jobs import PaperReleaseJobs
from moneymore.execution.paper import ExecutionBar
from moneymore.execution.risk import OrderIntent,RiskResult
from moneymore.models import Side
from moneymore.config import BacktestConfig
request=json.load(open(sys.argv[3]))
jobs=PaperReleaseJobs(sys.argv[1],sys.argv[2])
def execute(key):
    intent=OrderIntent(key,'SYNTHETIC','AAA',Side.BUY,100,'20261027','SYNTHETIC')
    jobs.epochs.broker.submit(RiskResult(True,intent,None),request['account_id'])
    jobs.epochs.broker.execute_pending(ExecutionBar('AAA','20261028',10,10),BacktestConfig.from_yaml(sys.argv[4]),request['account_id'])
    os._exit(71)
jobs.run(request,execute,now=datetime(2026,10,29,tzinfo=UTC))
"""
    config_path = Path(__file__).resolve().parents[1] / "configs/default.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path),
            str(jobs.database),
            str(request_path),
            str(config_path),
        ],
        timeout=30,
        capture_output=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 71, result.stderr.decode(errors="replace")
    assert len(broker.fills(account_id=request["account_id"])) == 1
    # OS lock is released immediately on process death, before lease expiry.
    from moneymore.config import BacktestConfig
    from moneymore.execution.paper import ExecutionBar
    from moneymore.execution.risk import OrderIntent, RiskResult
    from moneymore.models import Side

    def recover(key):
        intent = OrderIntent(key, "SYNTHETIC", "AAA", Side.BUY, 100, "20261027", "SYNTHETIC")
        assert broker.submit(RiskResult(True, intent, None), request["account_id"]) == "DUPLICATE"
        broker.execute_pending(
            ExecutionBar("AAA", "20261028", 10, 10),
            BacktestConfig.from_yaml(config_path),
            request["account_id"],
        )
        return {"recovered": True}

    PaperReleaseJobs(tmp_path, jobs.database).run(request, recover, now=now)
    assert len(broker.fills(account_id=request["account_id"])) == 1
    assert broker.reconcile(request["account_id"]).matched


def test_failed_prior_phase_cannot_be_skipped(setup):
    jobs, _, request, now = setup

    def fail(_):
        raise RuntimeError("SYNTHETIC_INTERRUPTION")

    with pytest.raises(RuntimeError):
        jobs.run(request, fail, now=now)
    with pytest.raises(ValueError, match="MUST_BE_RECOVERED_FIRST"):
        jobs.run({**request, "phase": "NEXT_SESSION_PLAN"}, lambda _: {}, now=now)


def test_reconciliation_failure_prevents_job_completion(setup):
    jobs, _, request, now = setup

    def corrupt(_):
        with sqlite3.connect(jobs.database) as conn:
            conn.execute(
                "UPDATE accounts SET cash=cash-1 WHERE account_id=?", (request["account_id"],)
            )
        return {}

    with pytest.raises(ValueError, match="RECONCILIATION_FAILED"):
        jobs.run(request, corrupt, now=now)
    with sqlite3.connect(jobs.database) as conn:
        assert conn.execute("SELECT status FROM v2_jobs").fetchone()[0] == "FAILED"


def test_future_session_rejected_without_callback(setup):
    jobs, _, request, now = setup
    with pytest.raises(ValueError, match="FUTURE_SESSION"):
        jobs.run({**request, "trade_date": "20261030"}, lambda _: pytest.fail("future"), now=now)
