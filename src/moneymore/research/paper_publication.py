"""Transactional staging/rollback for a new isolated paper publication.

PREPARED is deliberately not ACTIVE. Outbox records are disabled until a real
scheduler/first-cycle acceptance adapter is implemented and validated. This
module alone must never claim that the observation account has been published.
"""

import math
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ..execution.paper import PaperBroker
from .paper_release_jobs import exclusive_worker
from .paper_risk_epoch import validate_observation_contract

PHASES = ("OPEN_EXECUTION", "CLOSE_RECONCILIATION", "NEXT_SESSION_PLAN")


class PaperPublication:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def preflight(self, request):
        manifest, contract = validate_observation_contract(self.root, request)
        if str(Path(__file__).resolve()) not in manifest["input_hashes"]:
            raise ValueError("PUBLICATION_IMPLEMENTATION_NOT_FROZEN")
        version = contract.get("publication_version", "")
        cash = contract.get("initial_cash")
        if (
            not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", version)
            or request["account_id"] != "baseline_v2:" + version
            or type(cash) not in (int, float)
            or not math.isfinite(cash)
            or cash <= 0
        ):
            raise ValueError("INVALID_ISOLATED_PUBLICATION_CONTRACT")
        database = (
            self.root / "state/baseline-v2-release/paper-epochs" / (version + ".sqlite3")
        ).resolve()
        if not database.is_relative_to(self.root / "state/baseline-v2-release/paper-epochs"):
            raise ValueError("PUBLICATION_DATABASE_OUTSIDE_ISOLATED_ROOT")
        return contract, database

    def prepare(self, request, *, now=None):
        # All evidence checks precede even database/account creation.
        contract, database = self.preflight(request)
        PaperBroker(database)
        stamp = (now or datetime.now(UTC)).isoformat()
        account = request["account_id"]
        with exclusive_worker(database), sqlite3.connect(database, timeout=10) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_publications(
                account_id TEXT PRIMARY KEY,contract_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,created_at TEXT NOT NULL,reason TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_publication_outbox(
                account_id TEXT NOT NULL,phase TEXT NOT NULL,enabled INTEGER NOT NULL,
                PRIMARY KEY(account_id,phase))""")
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT contract_sha256,state FROM v2_publications WHERE account_id=?", (account,)
            ).fetchone()
            if existing:
                if existing[0] != request["contract_sha256"]:
                    raise ValueError("PUBLICATION_VERSION_ALREADY_BOUND_TO_OTHER_CONTRACT")
                return {"state": existing[1], "database": str(database), "published": False}
            if conn.execute("SELECT 1 FROM accounts").fetchone():
                raise ValueError("PUBLICATION_DATABASE_NOT_EMPTY")
            cash = float(contract["initial_cash"])
            conn.execute(
                "INSERT INTO accounts(account_id,initial_cash,cash,updated_at) VALUES (?,?,?,?)",
                (account, cash, cash, stamp),
            )
            conn.execute(
                "INSERT INTO v2_publications VALUES (?,?,'PREPARED',?,NULL)",
                (account, request["contract_sha256"], stamp),
            )
            for phase in PHASES:
                conn.execute("INSERT INTO v2_publication_outbox VALUES (?,?,0)", (account, phase))
            # No scheduler is called before commit. Disabled outbox is durable
            # intent, NOT successful registration or real observation activation.
        return {"state": "PREPARED", "database": str(database), "published": False}

    def rollback(self, version, *, reason):
        """May pause even if source evidence is now stale; never erase history."""
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", version) or not reason.strip():
            raise ValueError("EXPLICIT_VALID_ROLLBACK_REQUIRED")
        database = (
            self.root / "state/baseline-v2-release/paper-epochs" / (version + ".sqlite3")
        ).resolve()
        allowed = self.root / "state/baseline-v2-release/paper-epochs"
        if not database.is_relative_to(allowed) or not database.is_file():
            raise ValueError("ISOLATED_PUBLICATION_DATABASE_REQUIRED")
        account = "baseline_v2:" + version
        with exclusive_worker(database), sqlite3.connect(database, timeout=10) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if not conn.execute(
                "SELECT 1 FROM v2_publications WHERE account_id=?", (account,)
            ).fetchone():
                raise ValueError("PUBLICATION_NOT_REGISTERED")
            if conn.execute("SELECT 1 FROM accounts WHERE account_id<>?", (account,)).fetchone():
                raise ValueError("UNEXPECTED_OTHER_ACCOUNT_IN_PUBLICATION_DATABASE")
            conn.execute(
                "UPDATE v2_publications SET state='PAUSED',reason=? WHERE account_id=?",
                (reason, account),
            )
            conn.execute(
                "UPDATE v2_publication_outbox SET enabled=0 WHERE account_id=?", (account,)
            )
            conn.execute(
                """INSERT INTO execution_attempts(idempotency_key,trade_date,outcome,reason_code)
                SELECT idempotency_key,'','CANCELLED','V2_PUBLICATION_ROLLBACK'
                FROM orders WHERE account_id=? AND status='PENDING'""",
                (account,),
            )
            cancelled = conn.execute(
                "UPDATE orders SET status='CANCELLED' WHERE account_id=? AND status='PENDING'",
                (account,),
            ).rowcount
            conn.execute(
                """INSERT INTO account_risk_states(account_id,effective_state,proposed_state,reason_code,recovery_required,updated_at)
                VALUES (?,'SUSPENDED','SUSPENDED','V2_PUBLICATION_ROLLBACK',1,?)
                ON CONFLICT(account_id) DO UPDATE SET effective_state='SUSPENDED',proposed_state='SUSPENDED',
                reason_code='V2_PUBLICATION_ROLLBACK',recovery_required=1,updated_at=excluded.updated_at""",
                (account, datetime.now(UTC).isoformat()),
            )
        return {
            "state": "PAUSED",
            "cancelled_pending": cancelled,
            "history_deleted": False,
            "real_orders_enabled": False,
        }
