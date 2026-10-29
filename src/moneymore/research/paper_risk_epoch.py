"""Explicit new-risk-budget transactions for isolated future V2 paper accounts.

Not a timing strategy or automatic recovery. No web route or scheduler invokes
this module. Historical equity/high-water marks are never rewritten.
"""

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ..execution.paper import PaperBroker
from ..execution.risk_state import evaluate_account_risk_state
from .baseline_v2_platform_risk import PlatformRiskReductionPlans
from .baseline_v2_release import digest, read_json, release_decision, verify_manifest


def validate_observation_contract(root, request):
    """Read-only gate shared by publication, job recovery and risk approvals."""
    directory = Path(request["release_directory"]).resolve()
    if not directory.is_relative_to(Path(root).resolve() / "state/baseline-v2-release/runs"):
        raise ValueError("RECOVERY_CONTRACT_OUTSIDE_RELEASE_ROOT")
    manifest = read_json(directory / "manifest.json")
    verify_manifest(manifest)
    if str(Path(__file__).resolve()) not in manifest["input_hashes"]:
        raise ValueError("RECOVERY_IMPLEMENTATION_NOT_BOUND_TO_RELEASE")
    required = {
        "attribution",
        "full_union_economics",
        "financial_vintages",
        "robustness",
        "execution_stress",
        "parity",
        "forward_observation",
    }
    controlled = (required - {"financial_vintages"}) | {"financial_controls"}
    actual = set(manifest["protocol"]["required_stages"])
    vendor_protocol = (
        manifest["protocol"].get("version") == "baseline_v2_release_vendor_controls_v3_20261029"
    )
    if actual != (controlled if vendor_protocol else required):
        raise ValueError("RECOVERY_RELEASE_HARD_GATES_CHANGED")
    if vendor_protocol:
        policy = manifest["protocol"].get("financial_data_policy", {})
        if (
            policy.get("original_statement_exhaustive_attestation_required") is not False
            or policy.get("historical_point_in_time_attestation_claimed") is not False
            or policy.get("snapshot_required") is not True
        ):
            raise ValueError("VENDOR_FINANCIAL_POLICY_NOT_EXPLICIT")
    required = actual
    manifest_hash = digest(directory / "manifest.json")
    reports = {}
    for name in required - {"forward_observation"}:
        stage = read_json(directory / (name + ".json"))
        if stage.get("manifest_sha256") != manifest_hash or any(
            digest(p) != checksum for p, checksum in stage.get("artifact_hashes", {}).items()
        ):
            raise ValueError("RECOVERY_STAGE_EVIDENCE_CHANGED")
        reports[name] = stage
    if vendor_protocol and (
        reports["financial_controls"].get("acceptance_scope")
        != "VENDOR_DATA_CONTROLS_NOT_ORIGINAL_FINANCIAL_PIT"
        or reports["financial_controls"].get("historical_point_in_time_attested") is not False
    ):
        raise ValueError("VENDOR_CONTROL_STAGE_SCOPE_NOT_EXPLICIT")
    if not release_decision(manifest["protocol"], reports)["observation_activation_authorized"]:
        raise ValueError("RECOVERY_OBSERVATION_HARD_GATES_NOT_PASSED")
    contract = read_json(directory / "provisional-paper-contract.json")
    if (
        digest(directory / "provisional-paper-contract.json") != request["contract_sha256"]
        or contract.get("manifest_sha256") != manifest_hash
        or contract.get("account_id") != request["account_id"]
        or contract.get("paper_activation_authorized") is not True
        or contract.get("real_orders_enabled") is not False
        or contract.get("orders_enabled") is not False
    ):
        raise ValueError("RECOVERY_FROZEN_ACCOUNT_CONTRACT_INVALID")
    return manifest, contract


class PaperRiskEpochs:
    def __init__(self, root, database):
        self.root = Path(root).resolve()
        self.database = Path(database).resolve()
        allowed = (Path(root) / "state/baseline-v2-release/paper-epochs").resolve()
        if not self.database.is_relative_to(allowed) or not self.database.is_file():
            raise ValueError("ONLY_EXISTING_ISOLATED_V2_PAPER_DATABASE_ALLOWED")
        self.broker = PaperBroker(self.database)
        with sqlite3.connect(self.database) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_risk_epochs (
                approval_id TEXT PRIMARY KEY, account_id TEXT NOT NULL,
                epoch INTEGER NOT NULL, request_sha256 TEXT NOT NULL,
                epoch_peak REAL NOT NULL, receipt TEXT NOT NULL,
                UNIQUE(account_id, epoch))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_lifetime_peaks (
                account_id TEXT PRIMARY KEY,peak REAL NOT NULL)""")

    def lifetime_mark(self, account_id, equity, *, previous_peak=0):
        """Persist lifetime high-water mark; restart/epoch approval cannot reset it."""
        if (
            not account_id.startswith("baseline_v2:")
            or not math.isfinite(equity)
            or equity < 0
            or not math.isfinite(previous_peak)
            or previous_peak < 0
        ):
            raise ValueError("INVALID_ISOLATED_LIFETIME_MARK")
        with sqlite3.connect(self.database) as conn:
            conn.execute("BEGIN IMMEDIATE")
            account = conn.execute(
                "SELECT initial_cash FROM accounts WHERE account_id=?", (account_id,)
            ).fetchone()
            if account is None:
                raise ValueError("LIFETIME_ACCOUNT_MISSING")
            previous = conn.execute(
                "SELECT peak FROM v2_lifetime_peaks WHERE account_id=?", (account_id,)
            ).fetchone()
            peak = max(
                float(account[0]), equity, previous_peak, float(previous[0]) if previous else 0
            )
            conn.execute(
                "INSERT INTO v2_lifetime_peaks VALUES (?,?) ON CONFLICT(account_id) DO UPDATE SET peak=excluded.peak",
                (account_id, peak),
            )
        return {"peak": peak, "drawdown": equity / peak - 1}

    def validate_contract(self, request):
        validate_observation_contract(self.root, request)

    def approve(self, request, *, now=None):
        """Caller must be a trusted local approval flow, never a trading model.

        The future P7 integration must verify its signed/frozen release contract
        before supplying this request. This method additionally fails closed on
        all account, operational, expiry, idempotency and concurrency checks.
        """
        stamp = now or datetime.now(UTC)
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        account_id = request["account_id"]
        if (
            not account_id.startswith("baseline_v2:")
            or not request.get("approval_id", "").strip()
            or request.get("confirm_new_risk_budget") is not True
            or request.get("real_orders_enabled") is not False
            or request.get("observation_contract_authorized") is not True
            or len(request.get("contract_sha256", "")) != 64
            or not request.get("operator", "").strip()
            or not request.get("reason", "").strip()
        ):
            raise ValueError("EXPLICIT_AUTHORIZED_PAPER_RISK_BUDGET_REQUIRED")
        with sqlite3.connect(self.database) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM v2_risk_epochs WHERE approval_id=?", (request["approval_id"],)
            ).fetchone()
            if existing:
                if existing["request_sha256"] != fingerprint:
                    raise ValueError("APPROVAL_ID_REUSED_WITH_DIFFERENT_REQUEST")
                return json.loads(existing["receipt"])
            self.validate_contract(request)
            observed = datetime.fromisoformat(request["observed_at"])
            if observed.tzinfo is None or not 0 <= (stamp - observed).total_seconds() <= 120:
                raise ValueError("STALE_OR_FUTURE_RECOVERY_EVIDENCE")
            if (
                request.get("data_health") != "PASS"
                or not self.broker.reconcile(account_id).matched
            ):
                raise ValueError("RECOVERY_DATA_OR_RECONCILIATION_BLOCKED")
            account = conn.execute(
                "SELECT * FROM accounts WHERE account_id=?", (account_id,)
            ).fetchone()
            risk = conn.execute(
                "SELECT * FROM account_risk_states WHERE account_id=?", (account_id,)
            ).fetchone()
            if (
                not account
                or not risk
                or risk["effective_state"] != "SELL_ONLY"
                or risk["reason_code"] != "DRAWDOWN_LIMIT"
            ):
                raise ValueError("ONLY_DRAWDOWN_STOP_MAY_START_NEW_RISK_BUDGET")
            if risk["updated_at"] != request["expected_risk_updated_at"]:
                raise ValueError("RISK_STATE_CHANGED_SINCE_APPROVAL_PREVIEW")
            checks = [
                ("SELECT COUNT(*) FROM positions WHERE account_id=? AND quantity!=0",),
                ("SELECT COUNT(*) FROM orders WHERE account_id=? AND status='PENDING'",),
                (
                    "SELECT COUNT(*) FROM corporate_entitlements WHERE account_id=? AND ((cash_per_share>0 AND cash_status!='SETTLED') OR (stock_ratio>0 AND stock_status!='SETTLED'))",
                ),
            ]
            if any(conn.execute(sql, (account_id,)).fetchone()[0] for (sql,) in checks):
                raise ValueError("MUST_BE_SETTLED_CASH_ONLY_WITH_NO_PENDING_ORDERS")
            cash = float(account["cash"])
            lifetime_peak = float(request["lifetime_peak_before_recovery"])
            if (
                not math.isfinite(cash)
                or cash <= 0
                or abs(cash - float(request["expected_cash"])) > 0.01
                or not math.isfinite(lifetime_peak)
                or lifetime_peak < cash
            ):
                raise ValueError("INVALID_OR_CHANGED_RECOVERY_ACCOUNT_VALUES")
            epoch = conn.execute(
                "SELECT COALESCE(MAX(epoch),0) FROM v2_risk_epochs WHERE account_id=?",
                (account_id,),
            ).fetchone()[0]
            if epoch != request["expected_epoch"]:
                raise ValueError("RISK_EPOCH_CHANGED_SINCE_APPROVAL_PREVIEW")
            receipt = {
                "approval_id": request["approval_id"],
                "account_id": account_id,
                "epoch": epoch + 1,
                "epoch_initial_equity": cash,
                "lifetime_peak_before_recovery": lifetime_peak,
                "lifetime_drawdown_preserved": cash / lifetime_peak - 1,
                "contract_sha256": request["contract_sha256"],
                "operator": request["operator"],
                "reason": request["reason"],
                "approved_at": stamp.isoformat(),
                "history_rewritten": False,
                "automatic_recovery": False,
                "real_orders_enabled": False,
            }
            conn.execute(
                "INSERT INTO v2_risk_epochs VALUES (?,?,?,?,?,?)",
                (
                    request["approval_id"],
                    account_id,
                    epoch + 1,
                    fingerprint,
                    cash,
                    json.dumps(receipt),
                ),
            )
            conn.execute(
                "UPDATE account_risk_states SET effective_state='NORMAL', proposed_state='NORMAL', reason_code='EXPLICIT_NEW_PAPER_RISK_EPOCH', recovery_required=0, updated_at=? WHERE account_id=?",
                (stamp.isoformat(), account_id),
            )
            conn.execute(
                """INSERT INTO risk_state_transitions(account_id, trade_date, from_state, to_state,
                         proposed_state, reason_code, transition_type, operator, created_at)
                         VALUES (?, '', 'SELL_ONLY', 'NORMAL', 'NORMAL', 'EXPLICIT_NEW_PAPER_RISK_EPOCH',
                                 'MANUAL_NEW_RISK_EPOCH', ?, ?)""",
                (account_id, request["operator"], stamp.isoformat()),
            )
            return receipt

    def drawdown(self, account_id, equity):
        """Only the new isolated runtime may use this epoch peak for risk checks.

        Account lifetime PnL/drawdown must continue to use the untouched equity
        history, not this value. Without an approved epoch, return None.
        """
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("INVALID_EPOCH_EQUITY")
        with sqlite3.connect(self.database) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT approval_id, epoch_peak FROM v2_risk_epochs WHERE account_id=? ORDER BY epoch DESC LIMIT 1",
                (account_id,),
            ).fetchone()
            if row is None:
                return None
            peak = max(float(row[1]), equity)
            conn.execute(
                "UPDATE v2_risk_epochs SET epoch_peak=? WHERE approval_id=?", (peak, row[0])
            )
            return equity / peak - 1


class EpochAwareRiskPlans(PlatformRiskReductionPlans):
    """V2-only planner consumes explicit approvals; never generates one.

    Without an epoch, native lifetime-peak risk and sticky states are unchanged.
    Not wired to production accounts.
    """

    def __init__(self, *args, epoch_store, **kwargs):
        super().__init__(*args, **kwargs)
        if epoch_store.database != self.risk_database.resolve():
            raise ValueError("EPOCH_AND_PAPER_LEDGER_MUST_SHARE_ISOLATED_DATABASE")
        self.epoch_store = epoch_store

    def closing(self, date, snapshot):
        self.epoch_store.lifetime_mark(
            self.account_id, float(snapshot["equity"]), previous_peak=self.peak
        )
        self.epoch_store.drawdown(self.account_id, float(snapshot["equity"]))
        return super().closing(date, snapshot)

    def evaluate_opening_risk(self, date, snapshot):
        equity = float(snapshot["equity"])
        lifetime = self.epoch_store.lifetime_mark(self.account_id, equity, previous_peak=self.peak)
        self.peak = max(self.peak, lifetime["peak"])
        lifetime_drawdown = lifetime["drawdown"]
        epoch_drawdown = self.epoch_store.drawdown(self.account_id, equity)
        decision = evaluate_account_risk_state(
            self.config,
            equity,
            float(snapshot["cash"]),
            float(snapshot["market_value"]) / equity,
            lifetime_drawdown if epoch_drawdown is None else epoch_drawdown,
            "BLOCKED" if self.input_hold else "PASS",
            self.epoch_store.broker.reconcile(self.account_id).matched,
        )
        state = self.epoch_store.broker.update_risk_state(
            self.account_id, decision.proposed_state, decision.reason_code, date
        )
        return {
            **state,
            "lifetime_drawdown_preserved": lifetime_drawdown,
            "explicit_epoch_drawdown": epoch_drawdown,
            "automatic_recovery": False,
        }
