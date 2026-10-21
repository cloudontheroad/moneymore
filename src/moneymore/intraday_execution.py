from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .baseline_exposure_daily import BASELINE_EXPOSURE_ACCOUNT
from .config import BacktestConfig
from .corporate_actions import process_corporate_actions
from .costs import execution_price, transaction_fee
from .data.realtime import create_realtime_provider
from .data.store import ParquetStore
from .execution.paper import ExecutionBar, PaperBroker
from .models import Side
from .multi_sector_daily import MULTI_SECTOR_ACCOUNT

SHANGHAI = ZoneInfo("Asia/Shanghai")
POLICY_ID = "adaptive_vwap_sniper_v1"
INTRADAY_ACCOUNT = "multi_sector_intraday_shadow"
EXPOSURE_INTRADAY_ACCOUNT = "multi_sector_pysystemtrade_intraday_shadow"


def _ensure_comparison_account_and_orders(
    broker: PaperBroker,
    trade_date: str,
    *,
    source_account: str = MULTI_SECTOR_ACCOUNT,
    target_account: str = INTRADAY_ACCOUNT,
) -> int:
    """Share the parent's target holdings, not its path-dependent order deltas."""
    report_subdir = (
        "baseline-pysystemtrade-shadow"
        if source_account == BASELINE_EXPOSURE_ACCOUNT else "multi-sector-shadow"
    )
    report_dir = broker.database.parent / report_subdir
    reports = sorted(path for path in report_dir.glob("*.json") if path.stem < trade_date)
    if not reports:
        raise RuntimeError(f"no prior target report for {source_account} on {trade_date}")
    plan_date = reports[-1].stem
    plan = json.loads(reports[-1].read_text(encoding="utf-8"))
    if plan.get("status") != "COMPLETED" or not isinstance(plan.get("target_weights"), dict):
        raise RuntimeError(f"invalid target report: {reports[-1]}")
    with sqlite3.connect(broker.database, timeout=30) as connection:
        source = connection.execute(
            "SELECT initial_cash, cash, updated_at FROM accounts WHERE account_id = ?",
            (source_account,),
        ).fetchone()
        if source is None:
            raise RuntimeError("baseline paper account is not initialized")
        exists = connection.execute(
            "SELECT 1 FROM accounts WHERE account_id = ?", (target_account,)
        ).fetchone()
        if exists is None:
            connection.execute(
                "INSERT INTO accounts(account_id, initial_cash, cash, updated_at) VALUES (?, ?, ?, ?)",
                (target_account, *source),
            )
            connection.execute(
                """
                INSERT INTO positions(
                    account_id, symbol, quantity, available_quantity, avg_cost, last_buy_date
                )
                SELECT ?, symbol, quantity, available_quantity, avg_cost, last_buy_date
                FROM positions WHERE account_id = ?
                """,
                (target_account, source_account),
            )
        # Source positions already include today's fills. Pending or rejected
        # source orders still express the shared plan's terminal target.
        target_quantities = {
            str(symbol): int(quantity)
            for symbol, quantity in connection.execute(
                "SELECT symbol, quantity FROM positions WHERE account_id = ?",
                (source_account,),
            )
        }
        source_quantities = dict(target_quantities)
        for symbol, side, quantity in connection.execute(
            """SELECT symbol, side, quantity FROM orders
               WHERE account_id = ? AND signal_date = ?
                 AND status IN ('PENDING', 'REJECTED')""",
            (source_account, plan_date),
        ):
            target_quantities[str(symbol)] = target_quantities.get(str(symbol), 0) + (
                int(quantity) if side == "BUY" else -int(quantity)
            )
        # A report with no source order is still a plan: recover any branch
        # drift from an earlier rejected or missed order.
        target_quantities.update({
            str(symbol): target_quantities.get(str(symbol), 0)
            for symbol in plan["target_weights"]
        })
        branch_quantities = {
            str(symbol): int(quantity)
            for symbol, quantity in connection.execute(
                "SELECT symbol, quantity FROM positions WHERE account_id = ?",
                (target_account,),
            )
        }
        pretrade_source = dict(source_quantities)
        for symbol, side, quantity in connection.execute(
            """SELECT f.symbol, f.side, f.quantity FROM fills f
               JOIN orders o ON o.idempotency_key = f.idempotency_key
               WHERE f.account_id = ? AND f.trade_date = ? AND o.signal_date = ?""",
            (source_account, trade_date, plan_date),
        ):
            pretrade_source[str(symbol)] = pretrade_source.get(str(symbol), 0) - (
                int(quantity) if side == "BUY" else -int(quantity)
            )
        # Undo today's branch fills when identifying a debt inherited from
        # earlier sessions. Same-day ordinary orders must retain VWAP timing.
        pretrade_branch = dict(branch_quantities)
        for symbol, side, quantity in connection.execute(
            "SELECT symbol, side, quantity FROM fills WHERE account_id = ? AND trade_date = ?",
            (target_account, trade_date),
        ):
            pretrade_branch[str(symbol)] = pretrade_branch.get(str(symbol), 0) - (
                int(quantity) if side == "BUY" else -int(quantity)
            )
        # Never let a missed old order execute against a newer plan.
        stale_keys = [row[0] for row in connection.execute(
            """SELECT idempotency_key FROM orders
               WHERE account_id = ? AND status = 'PENDING' AND signal_date < ?""",
            (target_account, plan_date),
        )]
        for stale_key in stale_keys:
            connection.execute(
                "UPDATE orders SET status = 'CANCELLED' WHERE idempotency_key = ?",
                (stale_key,),
            )
            connection.execute(
                """INSERT INTO execution_attempts(
                       idempotency_key, trade_date, outcome, reason_code
                   ) VALUES (?, ?, 'CANCELLED', 'MISSED_SESSION_SUPERSEDED')""",
                (stale_key, trade_date),
            )
        mirrored = 0
        for symbol in sorted(set(target_quantities) | set(branch_quantities)):
            difference = target_quantities.get(symbol, 0) - branch_quantities.get(symbol, 0)
            if difference == 0:
                continue
            side = "BUY" if difference > 0 else "SELL"
            quantity = abs(difference)
            inherited_gap = pretrade_source.get(symbol, 0) - pretrade_branch.get(symbol, 0)
            reason_code = (
                "POSITION_GAP_COMPENSATION"
                if inherited_gap and (inherited_gap > 0) == (difference > 0)
                else "SHARED_TARGET_REBALANCE"
            )
            key = f"{target_account}:target:{plan_date}:{symbol}:{side}:{quantity}"
            cursor = connection.execute(
                """
                INSERT INTO orders(
                    idempotency_key, created_at, status, strategy_id, symbol,
                    side, quantity, signal_date, reason_code, account_id
                ) VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    status = 'PENDING', created_at = excluded.created_at,
                    reason_code = excluded.reason_code
                WHERE orders.status = 'CANCELLED' AND EXISTS (
                    SELECT 1 FROM execution_attempts a
                    WHERE a.idempotency_key = orders.idempotency_key
                      AND a.reason_code = 'HISTORICAL_REPLAY_SUPERSEDED'
                )
                """,
                (
                    key,
                    datetime.now(UTC).isoformat(),
                    f"{POLICY_ID}:{source_account}",
                    symbol,
                    side,
                    quantity,
                    plan_date,
                    reason_code,
                    target_account,
                ),
            )
            mirrored += int(cursor.rowcount > 0)
    return mirrored


def prepare_intraday_branch_orders(broker: PaperBroker, trade_date: str) -> int:
    """Mirror parent orders before open execution; recover same-day races safely."""
    return _ensure_comparison_account_and_orders(broker, trade_date)


def prepare_exposure_intraday_branch_orders(broker: PaperBroker, trade_date: str) -> int:
    return _ensure_comparison_account_and_orders(
        broker, trade_date,
        source_account=BASELINE_EXPOSURE_ACCOUNT,
        target_account=EXPOSURE_INTRADAY_ACCOUNT,
    )


@dataclass(frozen=True)
class IntradayDecision:
    symbol: str
    side: str
    action: str
    reason: str
    price: float
    vwap: float
    spread_bps: float
    urgency: float


def decide_intraday_execution(
    *,
    symbol: str,
    side: str,
    price: float,
    open_price: float,
    vwap: float,
    bid1: float,
    ask1: float,
    now_time: time,
    compensation: bool = False,
) -> IntradayDecision:
    start = time(9, 35)
    deadline = time(14, 50)
    if price <= 0 or open_price <= 0 or vwap <= 0:
        return IntradayDecision(symbol, side, "WAIT", "INVALID_QUOTE", price, vwap, 0, 0)
    spread_bps = (ask1 - bid1) / price * 10_000 if ask1 > 0 and bid1 > 0 and ask1 >= bid1 else 0
    if now_time < start:
        return IntradayDecision(
            symbol, side, "WAIT", "OPEN_NOISE_WINDOW", price, vwap, spread_bps, 0
        )
    if compensation:
        if spread_bps > 30 and now_time < time(9, 45):
            return IntradayDecision(
                symbol, side, "WAIT", "COMPENSATION_SPREAD_GUARD", price, vwap, spread_bps, 1
            )
        return IntradayDecision(
            symbol, side, "EXECUTE", "POSITION_GAP_COMPENSATION", price, vwap, spread_bps, 1
        )
    minutes = max(0, (now_time.hour * 60 + now_time.minute) - (9 * 60 + 35))
    urgency = min(1.0, minutes / 315)
    if now_time >= deadline:
        return IntradayDecision(
            symbol, side, "EXECUTE", "DEADLINE_FALLBACK", price, vwap, spread_bps, 1
        )
    if spread_bps > 30:
        return IntradayDecision(
            symbol, side, "WAIT", "SPREAD_TOO_WIDE", price, vwap, spread_bps, urgency
        )
    if side == "BUY":
        favorable = price <= vwap * (1.0005 + urgency * 0.0015) and price <= open_price * 1.005
    else:
        favorable = price >= vwap * (0.9995 - urgency * 0.0015) and price >= open_price * 0.995
    return IntradayDecision(
        symbol,
        side,
        "EXECUTE" if favorable else "WAIT",
        "VWAP_SNIPER_TRIGGER" if favorable else "WAIT_FOR_PRICE_IMPROVEMENT",
        price,
        vwap,
        spread_bps,
        urgency,
    )


def affordable_buy_quantity(
    requested: int, cash: float, price: float, config: BacktestConfig
) -> int:
    """Largest whole-lot buy whose slipped price and fees fit real cash."""
    if price <= 0 or cash <= 0:
        return 0
    unit_price = execution_price(price, Side.BUY, config.slippage_bps)
    lots = min(requested // config.lot_size, int(cash / unit_price) // config.lot_size)
    while lots > 0:
        quantity = lots * config.lot_size
        notional = quantity * unit_price
        if notional + transaction_fee(notional, Side.BUY, config) <= cash:
            return quantity
        lots -= 1
    return 0


def run_baseline_intraday_once(
    *,
    root: Path,
    now: datetime | None = None,
    source_account: str = MULTI_SECTOR_ACCOUNT,
    account_id: str = INTRADAY_ACCOUNT,
    snapshot_table: str = "baseline_intraday_snapshots",
    tick_table: str = "baseline_intraday_account_ticks",
    daily_table: str = "baseline_intraday_account_daily",
    audit_subdirectory: str = "intraday-execution",
) -> dict[str, object]:
    current = (now or datetime.now(SHANGHAI)).astimezone(SHANGHAI)
    trade_date = current.strftime("%Y%m%d")
    broker = PaperBroker(root / "state" / "paper_orders.sqlite3")
    mirrored = _ensure_comparison_account_and_orders(
        broker, trade_date, source_account=source_account, target_account=account_id
    )
    pending = [
        row
        for row in broker.orders()
        if row.get("account_id") == account_id
        and row.get("status") == "PENDING"
        and str(row.get("signal_date", "")) < trade_date
    ]
    payload: dict[str, object] = {
        "policy_id": POLICY_ID,
        "trade_date": trade_date,
        "observed_at": current.isoformat(),
        "account_id": account_id,
        "mirrored_orders": mirrored,
        "pending_orders": len(pending),
        "decisions": [],
        "executions": [],
        "status": "TRACKING" if not pending else "RUNNING",
    }
    empty_account = broker.account_snapshot({}, account_id)
    held_symbols = {str(row["symbol"]) for row in empty_account.get("positions", [])}
    symbols = sorted(held_symbols | {str(row["symbol"]) for row in pending})
    if not symbols:
        payload["account"] = empty_account
        return payload
    provider = create_realtime_provider(root=root)
    quotes = provider.intraday_quotes(symbols)
    if quotes.empty:
        payload["status"] = "REALTIME_QUOTES_EMPTY"
        return payload
    store = ParquetStore(root / "data")
    quotes["trade_date"] = trade_date
    quotes["observed_at"] = current.isoformat()
    store.merge_curated(
        snapshot_table,
        [quotes],
        ["trade_date", "observed_at", "symbol"],
    )
    config = BacktestConfig.from_yaml(root / "configs" / "default.yaml")
    quote_map = {str(row["symbol"]): row for row in quotes.to_dict("records")}
    marks = {symbol: float(row["last"]) for symbol, row in quote_map.items()}
    account = broker.account_snapshot(marks, account_id)
    positions = {str(row["symbol"]): row for row in account.get("positions", [])}
    decisions: list[dict[str, object]] = []
    executions: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    for order in sorted(pending, key=lambda row: (row.get("side") != "SELL", row["symbol"])):
        quote = quote_map.get(str(order["symbol"]))
        if not quote:
            continue
        if "execution_ready" in quote and not quote["execution_ready"]:
            blocked.append({"symbol": order["symbol"], "reason": quote.get("quality_reasons"),
                            "source": quote.get("source")})
            continue
        decision = decide_intraday_execution(
            symbol=str(order["symbol"]),
            side=str(order["side"]),
            price=float(quote["last"]),
            open_price=float(quote["open"]),
            vwap=float(quote["vwap"]),
            bid1=float(quote["bid1"]),
            ask1=float(quote["ask1"]),
            now_time=current.time(),
            compensation=order.get("reason_code") == "POSITION_GAP_COMPENSATION",
        )
        detail = asdict(decision) | {
            "idempotency_key": order["idempotency_key"],
            "quantity": int(order["quantity"]),
            "account_cash": float(account["cash"]),
            "held_quantity": int(positions.get(decision.symbol, {}).get("quantity", 0)),
            "available_quantity": int(
                positions.get(decision.symbol, {}).get("available_quantity", 0)
            ),
            "average_cost": float(positions.get(decision.symbol, {}).get("avg_cost", 0)),
        }
        decisions.append(detail)
        if decision.action != "EXECUTE":
            continue
        if order["side"] == "BUY":
            current_account = broker.account_snapshot(marks, account_id)
            affordable = affordable_buy_quantity(
                int(order["quantity"]), float(current_account["cash"]),
                decision.price, config,
            )
            if affordable == 0:
                detail["action"] = "WAIT"
                detail["reason"] = "COMPENSATION_CASH_BLOCKED"
                blocked.append({"symbol": decision.symbol, "missing_quantity": int(order["quantity"]), "reason": "INSUFFICIENT_CASH"})
                continue
            if affordable < int(order["quantity"]):
                # Fill the affordable part now; the remaining target gap is
                # regenerated on the next tick from actual holdings.
                adjusted_key = f"{order['idempotency_key']}:affordable:{affordable}"
                with sqlite3.connect(broker.database, timeout=30) as connection:
                    connection.execute(
                        "UPDATE orders SET status = 'CANCELLED' WHERE idempotency_key = ? AND status = 'PENDING'",
                        (order["idempotency_key"],),
                    )
                    connection.execute(
                        """INSERT OR IGNORE INTO orders(
                           idempotency_key, created_at, status, strategy_id, symbol,
                           side, quantity, signal_date, reason_code, account_id
                           ) VALUES (?, ?, 'PENDING', ?, ?, 'BUY', ?, ?, ?, ?)""",
                        (adjusted_key, datetime.now(UTC).isoformat(), order["strategy_id"],
                         decision.symbol, affordable, order["signal_date"],
                         order["reason_code"], account_id),
                    )
                detail["executed_quantity"] = affordable
                blocked.append({"symbol": decision.symbol, "missing_quantity": int(order["quantity"]) - affordable, "reason": "INSUFFICIENT_CASH"})
        can_buy = not quote.get("up_limit") or decision.price < float(quote["up_limit"])
        can_sell = not quote.get("down_limit") or decision.price > float(quote["down_limit"])
        executions.extend(
            broker.execute_pending(
                ExecutionBar(
                    symbol=decision.symbol,
                    trade_date=trade_date,
                    open=decision.price,
                    close=decision.price,
                    can_buy=can_buy,
                    can_sell=can_sell,
                ),
                config,
                account_id,
            )
        )
    process_corporate_actions(store, broker, account_id, trade_date, held_symbols)
    marks = {symbol: float(row["last"]) for symbol, row in quote_map.items()}
    account = broker.account_snapshot(marks, account_id)
    equity = float(account["equity"])
    market_value = float(account["market_value"])
    account_row = pd.DataFrame(
        [
            {
                "account_id": account_id,
                "trade_date": trade_date,
                "observed_at": current.isoformat(),
                "equity": equity,
                "cash": float(account["cash"]),
                "market_value": market_value,
                "gross_exposure": market_value / equity if equity else 0.0,
            }
        ]
    )
    store.merge_curated(
        tick_table,
        [account_row],
        ["account_id", "trade_date", "observed_at"],
    )
    store.merge_curated(
        daily_table,
        [account_row],
        ["account_id", "trade_date"],
    )
    payload["decisions"] = decisions
    payload["executions"] = executions
    payload["blocked_target_gaps"] = blocked
    payload["account"] = account
    payload["status"] = (
        "PARTIAL_COMPENSATION" if blocked else (
            "COMPLETED" if executions else ("WAITING_FOR_TRIGGER" if pending else "TRACKING")
        )
    )
    audit_dir = root / "state" / audit_subdirectory
    audit_dir.mkdir(parents=True, exist_ok=True)
    target = audit_dir / f"{trade_date}.json"
    history = []
    if target.exists():
        history = json.loads(target.read_text(encoding="utf-8")).get("observations", [])
    history.append(payload)
    target.write_text(
        json.dumps(
            {"trade_date": trade_date, "policy_id": POLICY_ID, "observations": history[-600:]},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload
