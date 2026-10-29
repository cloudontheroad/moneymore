"""Durable leases for isolated V2 paper jobs. Not a scheduler or release approval.

Callbacks MUST use the stable operation key with PaperBroker idempotency. A
process can die after a fill commits and before this journal commits: recovery
is at-least-once, never a claim of exactly-once arbitrary external side effects.
No existing production task invokes this module.
"""

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from .baseline_v2_release import digest, read_json
from .paper_risk_epoch import PaperRiskEpochs


@contextmanager
def exclusive_worker(database):
    """OS-owned lock releases on process death; expiry cannot permit overlap."""
    path = database.with_suffix(".worker.lock")
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("PAPER_JOB_ALREADY_RUNNING") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PaperReleaseJobs:
    def __init__(self, root, database):
        self.epochs = PaperRiskEpochs(root, database)
        self.database = self.epochs.database
        with sqlite3.connect(self.database) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_jobs (
                operation_key TEXT PRIMARY KEY, payload TEXT NOT NULL,
                status TEXT NOT NULL, lease_token TEXT, lease_until TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, result TEXT, error_type TEXT)""")

    def validate(self, request):
        if not request["account_id"].startswith("baseline_v2:"):
            raise ValueError("ONLY_ISOLATED_V2_ACCOUNT_ALLOWED")
        with sqlite3.connect(self.database) as conn:
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_publications'"
            ).fetchone():
                state = conn.execute(
                    "SELECT state FROM v2_publications WHERE account_id=?", (request["account_id"],)
                ).fetchone()
                if not state or state[0] == "PAUSED":
                    raise ValueError("PUBLICATION_PAUSED_OR_NOT_REGISTERED")
        self.epochs.validate_contract(request)
        manifest = read_json(Path(request["release_directory"]) / "manifest.json")
        if str(Path(__file__).resolve()) not in manifest["input_hashes"]:
            raise ValueError("JOB_IMPLEMENTATION_NOT_BOUND_TO_RELEASE")
        if (
            digest(Path(request["release_directory"]) / "provisional-paper-contract.json")
            != request["contract_sha256"]
        ):
            raise ValueError("JOB_CONTRACT_CHANGED")

    def run(self, request, callback, *, now=None, lease_seconds=120):
        with exclusive_worker(self.database):
            return self._run(request, callback, now=now, lease_seconds=lease_seconds)

    def _run(self, request, callback, *, now=None, lease_seconds=120):
        """Run/recover one immutable operation. Same account must use one worker.

        The OS lock, not a lease timeout, prevents overlap. On process death it
        releases immediately; the same operation may resume without waiting for
        expiry. The recorded lease is diagnostic, not permission to run twice.
        """
        stamp = now or datetime.now(UTC)
        if stamp.tzinfo is None or not 1 <= lease_seconds <= 600:
            raise ValueError("VALID_JOB_CLOCK_AND_BOUNDED_LEASE_REQUIRED")
        self.validate(request)
        if request["phase"] not in {"OPEN_EXECUTION", "CLOSE_RECONCILIATION", "NEXT_SESSION_PLAN"}:
            raise ValueError("UNKNOWN_PAPER_JOB_PHASE")
        date = (
            datetime.strptime(request["trade_date"], "%Y%m%d")
            .replace(tzinfo=UTC)
            .strftime("%Y%m%d")
        )
        if date != request["trade_date"]:
            raise ValueError("CANONICAL_SESSION_DATE_REQUIRED")
        # A historical operation can be recovered, but must never be called a
        # new prospective observation or a live delayed order.
        if date > stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"):
            raise ValueError("FUTURE_SESSION_JOB_REJECTED")
        evidence = Path(request["input_artifact"])
        if digest(evidence) != request["input_sha256"]:
            raise ValueError("JOB_INPUT_EVIDENCE_CHANGED")
        payload = json.dumps(request, sort_keys=True)
        key = hashlib.sha256((request["account_id"] + date + request["phase"]).encode()).hexdigest()
        token = uuid4().hex
        with sqlite3.connect(self.database, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM v2_jobs WHERE operation_key=?", (key,)
            ).fetchone()
            if existing and existing["payload"] != payload:
                raise ValueError("JOB_PAYLOAD_CHANGED_FOR_SAME_OPERATION")
            if existing and existing["status"] == "DONE":
                return json.loads(existing["result"])
            if conn.execute(
                "SELECT 1 FROM v2_jobs WHERE status IN ('RUNNING','FAILED') AND operation_key<>?",
                (key,),
            ).fetchone():
                raise ValueError("PREVIOUS_INTERRUPTED_JOB_MUST_BE_RECOVERED_FIRST")
            conn.execute(
                """INSERT INTO v2_jobs(operation_key,payload,status,attempts)
                VALUES (?,?,'PENDING',0) ON CONFLICT(operation_key) DO NOTHING""",
                (key, payload),
            )
            conn.execute(
                """UPDATE v2_jobs SET status='RUNNING',lease_token=?,lease_until=?,
                attempts=attempts+1,error_type=NULL WHERE operation_key=?""",
                (token, (stamp + timedelta(seconds=lease_seconds)).isoformat(), key),
            )
        try:
            result = callback(key)
            if not self.epochs.broker.reconcile(request["account_id"]).matched:
                raise ValueError("PAPER_JOB_RECONCILIATION_FAILED")
            self.validate(request)
            if digest(evidence) != request["input_sha256"]:
                raise ValueError("JOB_INPUT_CHANGED_DURING_EXECUTION")
            receipt = {
                "operation_key": key,
                "payload": result,
                "historical_recovery": date
                < stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"),
                "counts_as_prospective_observation": False,
                "real_orders_enabled": False,
            }
            encoded = json.dumps(receipt, allow_nan=False, sort_keys=True)
            with sqlite3.connect(self.database, timeout=10) as conn:
                changed = conn.execute(
                    """UPDATE v2_jobs SET status='DONE',result=?,lease_until=NULL
                    WHERE operation_key=? AND status='RUNNING' AND lease_token=?""",
                    (encoded, key, token),
                ).rowcount
                if changed != 1:
                    raise ValueError("PAPER_JOB_LEASE_LOST")
            return receipt
        except Exception as error:
            with sqlite3.connect(self.database, timeout=10) as conn:
                conn.execute(
                    """UPDATE v2_jobs SET status='FAILED',error_type=?,lease_until=NULL
                    WHERE operation_key=? AND lease_token=?""",
                    (type(error).__name__, key, token),
                )
            raise
