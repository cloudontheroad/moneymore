from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import BacktestConfig
from ..costs import execution_price, transaction_fee
from ..models import Side
from ..signals import SignalDecision
from .risk import OrderIntent, RiskResult


@dataclass(frozen=True)
class ExecutionBar:
    symbol: str
    trade_date: str
    open: float
    close: float
    can_buy: bool = True
    can_sell: bool = True


@dataclass(frozen=True)
class ReconciliationResult:
    matched: bool
    expected_cash: float
    actual_cash: float
    expected_quantity: int
    actual_quantity: int
    cash_difference: float
    quantity_difference: int
    filled_order_count: int
    fill_count: int
    missing_fill_count: int
    orphan_fill_count: int


class PaperBroker:
    """Audit-only order intake. It never connects to a real broker."""

    _schema_lock = threading.Lock()
    _initialized_databases: set[Path] = set()

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def submit(self, result: RiskResult, account_id: str = "default") -> str:
        if self._candidate_frozen(account_id):
            return "REJECTED_CANDIDATE_FROZEN"
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.database) as connection:
            if not result.accepted or result.intent is None:
                connection.execute(
                    "INSERT INTO rejections(created_at, code, payload) VALUES (?, ?, ?)",
                    (now, result.rejection_code, json.dumps(asdict(result))),
                )
                if result.rejection_code == "NO_REBALANCE_REQUIRED":
                    return "NO_ACTION"
                return "REJECTED"
            intent = result.intent
            try:
                connection.execute(
                    """
                    INSERT INTO orders(
                        idempotency_key, created_at, status, strategy_id, symbol,
                        side, quantity, signal_date, reason_code, account_id
                    ) VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.idempotency_key,
                        now,
                        intent.strategy_id,
                        intent.symbol,
                        intent.side.value,
                        intent.quantity,
                        intent.signal_date,
                        intent.reason_code,
                        account_id,
                    ),
                )
            except sqlite3.IntegrityError:
                existing = connection.execute(
                    """
                    SELECT status, account_id FROM orders
                    WHERE idempotency_key = ?
                    """,
                    (intent.idempotency_key,),
                ).fetchone()
                if existing == ("CANCELLED", account_id):
                    connection.execute(
                        """
                        UPDATE orders
                        SET status = 'PENDING', created_at = ?, strategy_id = ?,
                            symbol = ?, side = ?, quantity = ?, signal_date = ?,
                            reason_code = ?, account_id = ?
                        WHERE idempotency_key = ?
                        """,
                        (
                            now,
                            intent.strategy_id,
                            intent.symbol,
                            intent.side.value,
                            intent.quantity,
                            intent.signal_date,
                            intent.reason_code,
                            account_id,
                            intent.idempotency_key,
                        ),
                    )
                    return "REOPENED"
                self._record_duplicate(connection, intent, now)
                return "DUPLICATE"
        return "PENDING"

    def initialize_account(
        self, initial_cash: float, account_id: str = "default"
    ) -> str:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        with sqlite3.connect(self.database, timeout=30) as connection:
            if connection.execute(
                "SELECT 1 FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone():
                return "EXISTS"
            try:
                connection.execute(
                    """
                    INSERT INTO accounts(account_id, initial_cash, cash, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (account_id, initial_cash, initial_cash, datetime.now(UTC).isoformat()),
                )
            except sqlite3.IntegrityError:
                return "EXISTS"
        return "INITIALIZED"

    def portfolio(
        self, mark_price: float, symbol: str, account_id: str = "default"
    ) -> dict[str, float | int]:
        with sqlite3.connect(self.database) as connection:
            account = connection.execute(
                "SELECT cash FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
            if account is None:
                raise ValueError(f"paper account does not exist: {account_id}")
            position = connection.execute(
                """
                SELECT quantity, available_quantity, avg_cost
                FROM positions WHERE account_id = ? AND symbol = ?
                """,
                (account_id, symbol),
            ).fetchone() or (0, 0, 0.0)
        cash = float(account[0])
        quantity = int(position[0])
        return {
            "cash": cash,
            "equity": cash + quantity * mark_price,
            "position_quantity": quantity,
            "available_quantity": int(position[1]),
            "avg_cost": float(position[2]),
        }

    def account_snapshot(
        self, mark_prices: dict[str, float], account_id: str = "default"
    ) -> dict[str, object]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            account = connection.execute(
                "SELECT cash FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
            if account is None:
                raise ValueError(f"paper account does not exist: {account_id}")
            positions = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT symbol, quantity, available_quantity, avg_cost
                    FROM positions WHERE account_id = ? AND quantity != 0
                    ORDER BY symbol
                    """,
                    (account_id,),
                )
            ]
        market_value = sum(
            int(position["quantity"]) * float(mark_prices.get(position["symbol"], 0))
            for position in positions
        )
        cash = float(account["cash"])
        return {
            "cash": cash,
            "market_value": market_value,
            "equity": cash + market_value,
            "positions": positions,
        }

    def settle_available_positions(self, account_id: str, trade_date: str) -> None:
        """Release only prior-session purchases before a paper planning snapshot."""
        with sqlite3.connect(self.database) as connection:
            self._settle_positions(connection, account_id, trade_date)

    def execute_pending(
        self,
        bar: ExecutionBar,
        config: BacktestConfig,
        account_id: str = "default",
    ) -> list[dict[str, object]]:
        if self._candidate_frozen(account_id):
            self.cancel_pending(account_id, "CANDIDATE_LIFECYCLE_FROZEN")
            return []
        outcomes: list[dict[str, object]] = []
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            account = connection.execute(
                "SELECT cash FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
            if account is None:
                raise ValueError(f"paper account does not exist: {account_id}")
            self._settle_positions(connection, account_id, bar.trade_date)
            orders = connection.execute(
                """
                SELECT * FROM orders
                WHERE account_id = ? AND symbol = ? AND status = 'PENDING'
                  AND signal_date < ?
                ORDER BY created_at, idempotency_key
                """,
                (account_id, bar.symbol, bar.trade_date),
            ).fetchall()
            for order in orders:
                side = Side(order["side"])
                if side is Side.BUY and not bar.can_buy:
                    outcomes.append(
                        self._attempt(
                            connection, order, bar.trade_date, "DEFERRED", "LIMIT_UP"
                        )
                    )
                    continue
                if side is Side.SELL and not bar.can_sell:
                    outcomes.append(
                        self._attempt(
                            connection, order, bar.trade_date, "DEFERRED", "LIMIT_DOWN"
                        )
                    )
                    continue
                outcome = self._fill_order(
                    connection, order, bar, config, account_id
                )
                outcomes.append(outcome)
        return outcomes

    def reconcile(self, account_id: str = "default") -> ReconciliationResult:
        with sqlite3.connect(self.database) as connection:
            account = connection.execute(
                "SELECT initial_cash, cash FROM accounts WHERE account_id = ?",
                (account_id,),
            ).fetchone()
            if account is None:
                raise ValueError(f"paper account does not exist: {account_id}")
            fills = connection.execute(
                """
                SELECT side, quantity, price, fee FROM fills
                WHERE account_id = ? ORDER BY id
                """,
                (account_id,),
            ).fetchall()
            actual_quantity = connection.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM positions WHERE account_id = ?",
                (account_id,),
            ).fetchone()[0]
            filled_order_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM orders
                    WHERE account_id = ? AND status = 'FILLED'
                    """,
                    (account_id,),
                ).fetchone()[0]
            )
            fill_count = len(fills)
            missing_fill_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM orders o
                    LEFT JOIN fills f ON f.idempotency_key = o.idempotency_key
                    WHERE o.account_id = ? AND o.status = 'FILLED' AND f.id IS NULL
                    """,
                    (account_id,),
                ).fetchone()[0]
            )
            orphan_fill_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM fills f
                    LEFT JOIN orders o ON o.idempotency_key = f.idempotency_key
                    WHERE f.account_id = ? AND (
                        o.idempotency_key IS NULL OR o.status != 'FILLED'
                    )
                    """,
                    (account_id,),
                ).fetchone()[0]
            )
            corporate_actions = connection.execute(
                """
                SELECT action_type, cash_amount, share_quantity
                FROM corporate_action_ledger WHERE account_id = ?
                ORDER BY id
                """,
                (account_id,),
            ).fetchall()
        expected_cash = float(account[0])
        expected_quantity = 0
        for side, quantity, price, fee in fills:
            notional = int(quantity) * float(price)
            if side == Side.BUY.value:
                expected_cash -= notional + float(fee)
                expected_quantity += int(quantity)
            else:
                expected_cash += notional - float(fee)
                expected_quantity -= int(quantity)
        for action_type, cash_amount, share_quantity in corporate_actions:
            if action_type == "CASH_DIVIDEND":
                expected_cash += float(cash_amount)
            if action_type == "STOCK_DIVIDEND":
                expected_quantity += int(share_quantity)
        cash_difference = float(account[1]) - expected_cash
        quantity_difference = int(actual_quantity) - expected_quantity
        return ReconciliationResult(
            matched=(
                abs(cash_difference) < 0.01
                and quantity_difference == 0
                and filled_order_count == fill_count
                and missing_fill_count == 0
                and orphan_fill_count == 0
            ),
            expected_cash=expected_cash,
            actual_cash=float(account[1]),
            expected_quantity=expected_quantity,
            actual_quantity=int(actual_quantity),
            cash_difference=cash_difference,
            quantity_difference=quantity_difference,
            filled_order_count=filled_order_count,
            fill_count=fill_count,
            missing_fill_count=missing_fill_count,
            orphan_fill_count=orphan_fill_count,
        )

    def record_decision(self, decision: SignalDecision) -> str:
        with sqlite3.connect(self.database) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO decisions(
                        strategy_id, symbol, as_of_date, recorded_at, payload
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        decision.strategy_id,
                        decision.symbol,
                        decision.as_of_date,
                        datetime.now(UTC).isoformat(),
                        json.dumps(decision.to_dict(), ensure_ascii=False),
                    ),
                )
            except sqlite3.IntegrityError:
                return "DUPLICATE"
        return "RECORDED"

    def orders(self) -> list[dict[str, object]]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute("SELECT * FROM orders")]

    def fills(
        self, account_id: str, trade_date: str | None = None
    ) -> list[dict[str, object]]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            query = "SELECT * FROM fills WHERE account_id = ?"
            parameters: list[object] = [account_id]
            if trade_date is not None:
                query += " AND trade_date = ?"
                parameters.append(trade_date)
            query += " ORDER BY id"
            return [dict(row) for row in connection.execute(query, parameters)]

    def execution_attempts(
        self, account_id: str
    ) -> list[dict[str, object]]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT a.* FROM execution_attempts a
                    JOIN orders o ON o.idempotency_key = a.idempotency_key
                    WHERE o.account_id = ?
                    ORDER BY a.id
                    """,
                    (account_id,),
                )
            ]

    def cancel_pending(
        self,
        account_id: str,
        reason_code: str,
        signal_date: str | None = None,
        side: str | None = None,
    ) -> int:
        with sqlite3.connect(self.database) as connection:
            query = (
                "SELECT idempotency_key FROM orders "
                "WHERE account_id = ? AND status = 'PENDING'"
            )
            parameters: list[object] = [account_id]
            if signal_date is not None:
                query += " AND signal_date = ?"
                parameters.append(signal_date)
            if side is not None:
                query += " AND side = ?"
                parameters.append(side)
            keys = [row[0] for row in connection.execute(query, parameters)]
            for key in keys:
                connection.execute(
                    "UPDATE orders SET status = 'CANCELLED' WHERE idempotency_key = ?",
                    (key,),
                )
                connection.execute(
                    """
                    INSERT INTO execution_attempts(
                        idempotency_key, trade_date, outcome, reason_code
                    ) VALUES (?, ?, 'CANCELLED', ?)
                    """,
                    (key, signal_date or "", reason_code),
                )
        return len(keys)

    def cancel_pending_symbol(
        self,
        account_id: str,
        symbol: str,
        signal_date: str,
        reason_code: str,
    ) -> int:
        """Cancel stale same-day intents after an execution-rule update."""
        with sqlite3.connect(self.database) as connection:
            keys = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT idempotency_key FROM orders
                    WHERE account_id = ? AND symbol = ? AND signal_date = ?
                      AND status = 'PENDING'
                    """,
                    (account_id, symbol, signal_date),
                )
            ]
            for key in keys:
                connection.execute(
                    "UPDATE orders SET status = 'CANCELLED' WHERE idempotency_key = ?",
                    (key,),
                )
                connection.execute(
                    """
                    INSERT INTO execution_attempts(
                        idempotency_key, trade_date, outcome, reason_code
                    ) VALUES (?, ?, 'CANCELLED', ?)
                    """,
                    (key, signal_date, reason_code),
                )
        return len(keys)

    def restore_cancelled(
        self,
        account_id: str,
        signal_date: str,
        reason_code: str,
    ) -> int:
        """Restore a dated order batch after its blocking incident is resolved."""
        if self._candidate_frozen(account_id):
            return 0
        with sqlite3.connect(self.database) as connection:
            keys = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT idempotency_key FROM orders
                    WHERE account_id = ? AND signal_date = ?
                      AND status = 'CANCELLED'
                    """,
                    (account_id, signal_date),
                )
            ]
            for key in keys:
                connection.execute(
                    """
                    UPDATE orders SET status = 'PENDING'
                    WHERE idempotency_key = ?
                    """,
                    (key,),
                )
                connection.execute(
                    """
                    INSERT INTO execution_attempts(
                        idempotency_key, trade_date, outcome, reason_code
                    ) VALUES (?, ?, 'RESTORED', ?)
                    """,
                    (key, signal_date, reason_code),
                )
        return len(keys)

    def _candidate_frozen(self, account_id: str) -> bool:
        if account_id == "qlib_gru_shadow":
            deployment = self.database.parent / "qlib-governance" / "deployment.json"
            if deployment.exists():
                try:
                    return json.loads(deployment.read_text(encoding="utf-8")).get("lifecycle") == "RETIRED"
                except (OSError, ValueError):
                    return True
        if not account_id.startswith("qlib_candidate_"):
            return False
        from ..candidate_lifecycle import trading_allowed

        return not trading_allowed(
            self.database.parent.parent, account_id.removeprefix("qlib_candidate_"), ""
        )

    def risk_state(self, account_id: str) -> dict[str, object]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM account_risk_states WHERE account_id = ?",
                (account_id,),
            ).fetchone()
            if row is None:
                now = datetime.now(UTC).isoformat()
                connection.execute(
                    """
                    INSERT INTO account_risk_states(
                        account_id, effective_state, proposed_state,
                        reason_code, recovery_required, updated_at
                    ) VALUES (?, 'NORMAL', 'NORMAL', 'INITIALIZED', 0, ?)
                    """,
                    (account_id, now),
                )
                return {
                    "account_id": account_id,
                    "effective_state": "NORMAL",
                    "proposed_state": "NORMAL",
                    "reason_code": "INITIALIZED",
                    "recovery_required": False,
                    "updated_at": now,
                }
            result = dict(row)
            result["recovery_required"] = bool(result["recovery_required"])
            return result

    def update_risk_state(
        self,
        account_id: str,
        proposed_state: str,
        reason_code: str,
        trade_date: str,
    ) -> dict[str, object]:
        levels = {"NORMAL": 0, "REDUCE_ONLY": 1, "SELL_ONLY": 2, "SUSPENDED": 3}
        if proposed_state not in levels:
            raise ValueError(f"invalid risk state: {proposed_state}")
        current = self.risk_state(account_id)
        effective = str(current["effective_state"])
        recovery_required = levels[proposed_state] < levels[effective]
        next_effective = (
            proposed_state
            if levels[proposed_state] >= levels[effective]
            else effective
        )
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE account_risk_states SET
                    effective_state = ?, proposed_state = ?, reason_code = ?,
                    recovery_required = ?, updated_at = ?
                WHERE account_id = ?
                """,
                (
                    next_effective,
                    proposed_state,
                    reason_code,
                    int(recovery_required),
                    now,
                    account_id,
                ),
            )
            if next_effective != effective or proposed_state != current["proposed_state"]:
                connection.execute(
                    """
                    INSERT INTO risk_state_transitions(
                        account_id, trade_date, from_state, to_state,
                        proposed_state, reason_code, transition_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'AUTOMATIC', ?)
                    """,
                    (
                        account_id,
                        trade_date,
                        effective,
                        next_effective,
                        proposed_state,
                        reason_code,
                        now,
                    ),
                )
        return self.risk_state(account_id)

    def approve_risk_recovery(
        self, account_id: str, operator: str = "LOCAL_USER"
    ) -> dict[str, object]:
        current = self.risk_state(account_id)
        if not current["recovery_required"]:
            return current
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                UPDATE account_risk_states SET
                    effective_state = proposed_state,
                    recovery_required = 0, updated_at = ?
                WHERE account_id = ?
                """,
                (now, account_id),
            )
            connection.execute(
                """
                INSERT INTO risk_state_transitions(
                    account_id, trade_date, from_state, to_state,
                    proposed_state, reason_code, transition_type,
                    operator, created_at
                ) VALUES (?, '', ?, ?, ?, ?, 'MANUAL_RECOVERY', ?, ?)
                """,
                (
                    account_id,
                    current["effective_state"],
                    current["proposed_state"],
                    current["proposed_state"],
                    current["reason_code"],
                    operator,
                    now,
                ),
            )
        return self.risk_state(account_id)

    def risk_transitions(self, account_id: str) -> list[dict[str, object]]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM risk_state_transitions
                    WHERE account_id = ? ORDER BY id DESC
                    """,
                    (account_id,),
                )
            ]

    def register_corporate_entitlement(
        self,
        account_id: str,
        symbol: str,
        action_key: str,
        record_date: str,
        cash_per_share: float,
        stock_ratio: float,
        cash_pay_date: str | None,
        stock_list_date: str | None,
    ) -> str:
        with sqlite3.connect(self.database) as connection:
            position = connection.execute(
                """
                SELECT quantity FROM positions
                WHERE account_id = ? AND symbol = ?
                """,
                (account_id, symbol),
            ).fetchone()
            quantity = int(position[0]) if position else 0
            if quantity <= 0:
                return "NO_POSITION"
            try:
                connection.execute(
                    """
                    INSERT INTO corporate_entitlements(
                        account_id, action_key, symbol, record_date,
                        entitled_quantity, cash_per_share, stock_ratio,
                        cash_pay_date, stock_list_date, cash_status, stock_status,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        account_id,
                        action_key,
                        symbol,
                        record_date,
                        quantity,
                        cash_per_share,
                        stock_ratio,
                        cash_pay_date,
                        stock_list_date,
                        "PENDING" if cash_per_share > 0 else "NOT_APPLICABLE",
                        "PENDING" if stock_ratio > 0 else "NOT_APPLICABLE",
                        datetime.now(UTC).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                return "DUPLICATE"
        return "REGISTERED"

    def settle_corporate_actions(
        self, account_id: str, trade_date: str
    ) -> list[dict[str, object]]:
        settled: list[dict[str, object]] = []
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT * FROM corporate_entitlements
                WHERE account_id = ? AND (
                    (cash_status = 'PENDING' AND cash_pay_date <= ?)
                    OR (stock_status = 'PENDING' AND stock_list_date <= ?)
                )
                ORDER BY symbol, action_key
                """,
                (account_id, trade_date, trade_date),
            ).fetchall()
            for row in rows:
                if row["cash_status"] == "PENDING" and row["cash_pay_date"] <= trade_date:
                    amount = int(row["entitled_quantity"]) * float(
                        row["cash_per_share"]
                    )
                    connection.execute(
                        "UPDATE accounts SET cash = cash + ?, updated_at = ? "
                        "WHERE account_id = ?",
                        (amount, datetime.now(UTC).isoformat(), account_id),
                    )
                    connection.execute(
                        """
                        UPDATE corporate_entitlements SET cash_status = 'SETTLED'
                        WHERE account_id = ? AND action_key = ?
                        """,
                        (account_id, row["action_key"]),
                    )
                    self._record_corporate_ledger(
                        connection, account_id, row, trade_date,
                        "CASH_DIVIDEND", amount, 0,
                    )
                    settled.append(
                        {
                            "symbol": row["symbol"],
                            "action_type": "CASH_DIVIDEND",
                            "cash_amount": amount,
                            "share_quantity": 0,
                        }
                    )
                if (
                    row["stock_status"] == "PENDING"
                    and row["stock_list_date"] <= trade_date
                ):
                    shares = int(
                        int(row["entitled_quantity"]) * float(row["stock_ratio"])
                    )
                    if shares > 0:
                        position = connection.execute(
                            """
                            SELECT quantity, available_quantity, avg_cost
                            FROM positions WHERE account_id = ? AND symbol = ?
                            """,
                            (account_id, row["symbol"]),
                        ).fetchone() or (0, 0, 0.0)
                        held, available, avg_cost = (
                            int(position[0]), int(position[1]), float(position[2])
                        )
                        new_quantity = held + shares
                        new_avg = held * avg_cost / new_quantity if new_quantity else 0
                        connection.execute(
                            """
                            INSERT INTO positions(
                                account_id, symbol, quantity, available_quantity,
                                avg_cost, last_buy_date
                            ) VALUES (?, ?, ?, ?, ?, NULL)
                            ON CONFLICT(account_id, symbol) DO UPDATE SET
                                quantity=excluded.quantity,
                                available_quantity=excluded.available_quantity,
                                avg_cost=excluded.avg_cost
                            """,
                            (
                                account_id,
                                row["symbol"],
                                new_quantity,
                                available + shares,
                                new_avg,
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE corporate_entitlements SET stock_status = 'SETTLED'
                        WHERE account_id = ? AND action_key = ?
                        """,
                        (account_id, row["action_key"]),
                    )
                    self._record_corporate_ledger(
                        connection, account_id, row, trade_date,
                        "STOCK_DIVIDEND", 0.0, shares,
                    )
                    settled.append(
                        {
                            "symbol": row["symbol"],
                            "action_type": "STOCK_DIVIDEND",
                            "cash_amount": 0.0,
                            "share_quantity": shares,
                        }
                    )
        return settled

    def corporate_actions(self, account_id: str) -> list[dict[str, object]]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT * FROM corporate_action_ledger
                    WHERE account_id = ? ORDER BY trade_date DESC, id DESC
                    """,
                    (account_id,),
                )
            ]

    def _initialize(self) -> None:
        database = self.database.resolve()
        with self._schema_lock:
            if database in self._initialized_databases:
                return
            with sqlite3.connect(self.database, timeout=30) as connection:
                connection.execute("PRAGMA busy_timeout = 30000")
                connection.execute("PRAGMA journal_mode = WAL")
                self._initialize_schema(connection)
            self._initialized_databases.add(database)

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    idempotency_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    signal_date TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    account_id TEXT NOT NULL DEFAULT 'default'
                );
                CREATE TABLE IF NOT EXISTS rejections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    code TEXT,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(strategy_id, symbol, as_of_date)
                );
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    initial_cash REAL NOT NULL,
                    cash REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    available_quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    last_buy_date TEXT,
                    PRIMARY KEY(account_id, symbol)
                );
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    trade_date TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS execution_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idempotency_key TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reason_code TEXT
                );
                CREATE TABLE IF NOT EXISTS corporate_entitlements (
                    account_id TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    record_date TEXT NOT NULL,
                    entitled_quantity INTEGER NOT NULL,
                    cash_per_share REAL NOT NULL,
                    stock_ratio REAL NOT NULL,
                    cash_pay_date TEXT,
                    stock_list_date TEXT,
                    cash_status TEXT NOT NULL,
                    stock_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(account_id, action_key)
                );
                CREATE TABLE IF NOT EXISTS corporate_action_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    entitled_quantity INTEGER NOT NULL,
                    cash_amount REAL NOT NULL,
                    share_quantity INTEGER NOT NULL,
                    UNIQUE(account_id, action_key, action_type)
                );
                CREATE TABLE IF NOT EXISTS account_risk_states (
                    account_id TEXT PRIMARY KEY,
                    effective_state TEXT NOT NULL,
                    proposed_state TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    recovery_required INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS risk_state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    proposed_state TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    transition_type TEXT NOT NULL,
                    operator TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(orders)")
            }
            if "account_id" not in columns:
                connection.execute(
                    "ALTER TABLE orders ADD COLUMN account_id TEXT NOT NULL DEFAULT 'default'"
                )

    @staticmethod
    def _record_corporate_ledger(
        connection: sqlite3.Connection,
        account_id: str,
        row: sqlite3.Row,
        trade_date: str,
        action_type: str,
        cash_amount: float,
        share_quantity: int,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO corporate_action_ledger(
                account_id, action_key, symbol, trade_date, action_type,
                entitled_quantity, cash_amount, share_quantity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id,
                row["action_key"],
                row["symbol"],
                trade_date,
                action_type,
                row["entitled_quantity"],
                cash_amount,
                share_quantity,
            ),
        )

    @staticmethod
    def _record_duplicate(
        connection: sqlite3.Connection, intent: OrderIntent, now: str
    ) -> None:
        connection.execute(
            "INSERT INTO rejections(created_at, code, payload) VALUES (?, ?, ?)",
            (now, "DUPLICATE_ORDER", json.dumps(asdict(intent))),
        )

    @staticmethod
    def _settle_positions(
        connection: sqlite3.Connection, account_id: str, trade_date: str
    ) -> None:
        connection.execute(
            """
            UPDATE positions SET available_quantity = quantity
            WHERE account_id = ? AND last_buy_date IS NOT NULL
              AND last_buy_date < ?
            """,
            (account_id, trade_date),
        )

    @staticmethod
    def _attempt(
        connection: sqlite3.Connection,
        order: sqlite3.Row,
        trade_date: str,
        outcome: str,
        reason: str | None,
    ) -> dict[str, object]:
        connection.execute(
            """
            INSERT INTO execution_attempts(
                idempotency_key, trade_date, outcome, reason_code
            ) VALUES (?, ?, ?, ?)
            """,
            (order["idempotency_key"], trade_date, outcome, reason),
        )
        return {
            "idempotency_key": order["idempotency_key"],
            "outcome": outcome,
            "reason_code": reason,
        }

    def _fill_order(
        self,
        connection: sqlite3.Connection,
        order: sqlite3.Row,
        bar: ExecutionBar,
        config: BacktestConfig,
        account_id: str,
    ) -> dict[str, object]:
        side = Side(order["side"])
        quantity = int(order["quantity"])
        price = execution_price(bar.open, side, config.slippage_bps)
        notional = quantity * price
        fee = transaction_fee(notional, side, config)
        cash = float(
            connection.execute(
                "SELECT cash FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()[0]
        )
        position = connection.execute(
            """
            SELECT quantity, available_quantity, avg_cost, last_buy_date
            FROM positions WHERE account_id = ? AND symbol = ?
            """,
            (account_id, bar.symbol),
        ).fetchone() or (0, 0, 0.0, None)
        held, available, avg_cost = int(position[0]), int(position[1]), float(position[2])
        prior_last_buy_date = position[3]
        if side is Side.BUY and notional + fee > cash:
            connection.execute(
                "UPDATE orders SET status = 'REJECTED' WHERE idempotency_key = ?",
                (order["idempotency_key"],),
            )
            return self._attempt(
                connection, order, bar.trade_date, "REJECTED", "INSUFFICIENT_CASH"
            )
        if side is Side.SELL and quantity > held:
            connection.execute(
                "UPDATE orders SET status = 'REJECTED' WHERE idempotency_key = ?",
                (order["idempotency_key"],),
            )
            return self._attempt(
                connection, order, bar.trade_date, "REJECTED", "INSUFFICIENT_POSITION"
            )
        if side is Side.SELL and quantity > available:
            return self._attempt(
                connection, order, bar.trade_date, "DEFERRED", "T_PLUS_ONE"
            )
        if side is Side.BUY:
            new_quantity = held + quantity
            new_avg = (held * avg_cost + notional + fee) / new_quantity
            new_available = available
            new_cash = cash - notional - fee
            last_buy_date = bar.trade_date
        else:
            new_quantity = held - quantity
            new_avg = avg_cost if new_quantity else 0.0
            new_available = available - quantity
            new_cash = cash + notional - fee
            last_buy_date = None if new_quantity == 0 else prior_last_buy_date
        connection.execute(
            """
            INSERT INTO positions(
                account_id, symbol, quantity, available_quantity, avg_cost, last_buy_date
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id, symbol) DO UPDATE SET
                quantity=excluded.quantity,
                available_quantity=excluded.available_quantity,
                avg_cost=excluded.avg_cost,
                last_buy_date=excluded.last_buy_date
            """,
            (
                account_id,
                bar.symbol,
                new_quantity,
                new_available,
                new_avg,
                last_buy_date,
            ),
        )
        connection.execute(
            "UPDATE accounts SET cash = ?, updated_at = ? WHERE account_id = ?",
            (new_cash, datetime.now(UTC).isoformat(), account_id),
        )
        connection.execute(
            """
            INSERT INTO fills(
                account_id, idempotency_key, symbol, side, quantity, price, fee, trade_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id,
                order["idempotency_key"],
                bar.symbol,
                side.value,
                quantity,
                price,
                fee,
                bar.trade_date,
            ),
        )
        connection.execute(
            "UPDATE orders SET status = 'FILLED' WHERE idempotency_key = ?",
            (order["idempotency_key"],),
        )
        return self._attempt(
            connection, order, bar.trade_date, "FILLED", None
        ) | {"price": price, "quantity": quantity, "fee": fee}
