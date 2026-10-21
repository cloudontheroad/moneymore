from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .config import BacktestConfig
from .data.research import load_total_return_stock_bars
from .data.store import ParquetStore
from .execution.paper import ExecutionBar, PaperBroker


def _pending_orders(
    broker: PaperBroker, account_ids: list[str], trade_date: str
) -> tuple[list[str], list[dict[str, object]]]:
    accounts = sorted(set(account_ids))
    pending = [
        row
        for row in broker.orders()
        if str(row.get("account_id")) in accounts
        and str(row.get("status")) == "PENDING"
        and str(row.get("signal_date", "")) < trade_date
    ]
    return accounts, pending


def _execute_with_bars(
    *,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
    accounts: list[str],
    pending: list[dict[str, object]],
    bars: dict[str, ExecutionBar],
    missing: list[str],
    audit_dir: str | Path,
    price_source: str,
) -> dict[str, object]:
    executions: list[dict[str, object]] = []
    account_results: list[dict[str, object]] = []
    for account_id in accounts:
        account_pending = [row for row in pending if str(row.get("account_id")) == account_id]
        pending_sells = {
            str(row["symbol"]) for row in account_pending if str(row.get("side")) == "SELL"
        }
        account_symbols = {str(row["symbol"]) for row in account_pending}
        execution_order = sorted(
            account_symbols & set(bars),
            key=lambda symbol: (symbol not in pending_sells, symbol),
        )
        account_executions: list[dict[str, object]] = []
        for symbol in execution_order:
            account_executions.extend(broker.execute_pending(bars[symbol], config, account_id))
        executions.extend(account_executions)
        account_results.append(
            {
                "account_id": account_id,
                "pending_before": len(account_pending),
                "executions": len(account_executions),
                "reconciliation": asdict(broker.reconcile(account_id)),
            }
        )

    payload: dict[str, object] = {
        "trade_date": trade_date,
        "status": "COMPLETED" if not missing else "PARTIAL_MISSING_BARS",
        "price_source": price_source,
        "pending_orders": len(pending),
        "execution_events": len(executions),
        "missing_symbols": missing,
        "accounts": account_results,
        "executions": executions,
        "created_at": datetime.now(UTC).isoformat(),
    }
    target_dir = Path(audit_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{trade_date}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return payload


def execute_accounts_at_open(
    *,
    store: ParquetStore,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
    account_ids: list[str],
    audit_dir: str | Path,
) -> dict[str, object]:
    """Execute prior-session pending orders before expensive daily research."""
    accounts, pending = _pending_orders(broker, account_ids, trade_date)
    symbols = sorted({str(row["symbol"]) for row in pending})
    bars: dict[str, ExecutionBar] = {}
    missing: list[str] = []
    for symbol in symbols:
        history = load_total_return_stock_bars(store, symbol)
        current = history.loc[
            history["date"].astype(str).str.replace("-", "").str[:8] == trade_date
        ]
        if current.empty:
            missing.append(symbol)
            continue
        row = current.iloc[-1]
        bars[symbol] = ExecutionBar(
            symbol=symbol,
            trade_date=trade_date,
            open=float(row["raw_open"]),
            close=float(row["raw_close"]),
            can_buy=bool(row.get("can_buy", True)),
            can_sell=bool(row.get("can_sell", True)),
        )

    return _execute_with_bars(
        broker=broker,
        config=config,
        trade_date=trade_date,
        accounts=accounts,
        pending=pending,
        bars=bars,
        missing=missing,
        audit_dir=audit_dir,
        price_source="DAILY_OPEN",
    )


def execute_accounts_at_qmt_open(
    *,
    provider: object,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
    account_ids: list[str],
    audit_dir: str | Path,
) -> dict[str, object]:
    """Execute yesterday's plans from QMT's immutable current-session open."""
    accounts, pending = _pending_orders(broker, account_ids, trade_date)
    symbols = sorted({str(row["symbol"]) for row in pending})
    if not symbols:
        return _execute_with_bars(
            broker=broker,
            config=config,
            trade_date=trade_date,
            accounts=accounts,
            pending=pending,
            bars={},
            missing=[],
            audit_dir=audit_dir,
            price_source=f"{getattr(provider, 'name', 'qmt').upper()}_SESSION_OPEN",
        )
    quotes = provider.intraday_quotes(symbols)
    bars: dict[str, ExecutionBar] = {}
    if quotes is not None and not quotes.empty:
        for row in quotes.to_dict("records"):
            if "execution_ready" in row and not row["execution_ready"]:
                continue
            symbol = str(row["symbol"])
            open_price = float(row.get("open", 0) or 0)
            if open_price <= 0:
                continue
            up_limit = float(row.get("up_limit", 0) or 0)
            down_limit = float(row.get("down_limit", 0) or 0)
            bars[symbol] = ExecutionBar(
                symbol=symbol,
                trade_date=trade_date,
                open=open_price,
                close=float(row.get("last", open_price) or open_price),
                can_buy=not up_limit or open_price < up_limit,
                can_sell=not down_limit or open_price > down_limit,
            )
    missing = sorted(set(symbols) - set(bars))
    return _execute_with_bars(
        broker=broker,
        config=config,
        trade_date=trade_date,
        accounts=accounts,
        pending=pending,
        bars=bars,
        missing=missing,
        audit_dir=audit_dir,
        price_source=f"{getattr(provider, 'name', 'qmt').upper()}_SESSION_OPEN",
    )
