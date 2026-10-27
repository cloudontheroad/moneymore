from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .execution.paper import ExecutionBar, PaperBroker
from .execution.risk import PortfolioSnapshot, create_order_intent
from .signals import SignalDecision


@dataclass(frozen=True)
class ExecutionReplayResult:
    strategy_id: str
    account_id: str
    equity: pd.DataFrame
    orders: list[dict[str, object]]
    fills: list[dict[str, object]]
    attempts: list[dict[str, object]]
    reconciliation: dict[str, object]
    summary: dict[str, object]


def run_execution_replay(
    *,
    bars: pd.DataFrame,
    targets: pd.DataFrame,
    config: BacktestConfig,
    database: str | Path,
    strategy_id: str,
    account_id: str,
    corporate_actions: pd.DataFrame | None = None,
    closing_target_provider: Callable[[str, dict], pd.DataFrame] | None = None,
    accrue_action_receivables: bool = False,
    opening_plan_provider: Callable[[str, dict, pd.DataFrame], dict] | None = None,
) -> ExecutionReplayResult:
    """Replay historical targets through the production PaperBroker.

    Required bar fields are date, symbol, open and close. Optional can_buy and
    can_sell fields model suspension and price-limit constraints. Targets are
    sparse rebalance instructions and require date, symbol and target_weight.
    """
    market = _prepare_bars(bars)
    instructions = _prepare_targets(targets)
    actions = _prepare_actions(corporate_actions)
    if accrue_action_receivables and not actions.empty and "ex_date" not in actions:
        raise ValueError("ex_date required for economic action replay")
    database = Path(database)
    if database.exists():
        raise FileExistsError(f"replay database already exists: {database}")
    broker = PaperBroker(database)
    broker.initialize_account(config.initial_cash, account_id)
    target_state: dict[str, float] = {}
    equity_rows: list[dict[str, object]] = []
    previous_equity = config.initial_cash
    opening_audits: list[dict] = []
    if opening_plan_provider is not None and closing_target_provider is None:
        raise ValueError("opening planner requires a closing signal provider")

    for trade_date, day in market.groupby("date", sort=True):
        date_key = pd.Timestamp(trade_date).strftime("%Y%m%d")
        day = day.set_index("symbol")
        marks = {str(symbol): float(row["close"]) for symbol, row in day.iterrows()}

        broker.settle_corporate_actions(account_id, date_key)
        execution_order: dict[str, int] = {}
        opening_plan = None
        open_marks = {str(s): float(r["open"]) for s, r in day.iterrows()}
        if opening_plan_provider is not None:
            # Never expose today's close/high/low to the opening planner.
            opening = broker.account_snapshot(open_marks, account_id)
            if any(
                int(p["quantity"]) > 0 and p["symbol"] not in open_marks
                for p in opening["positions"]
            ):
                raise ValueError("opening valuation missing a held symbol")
            if accrue_action_receivables:
                from .research.replay_actions import economic_snapshot

                opening = economic_snapshot(
                    broker, opening, open_marks, account_id, date_key, actions
                )
            broker.cancel_pending(account_id, "REPLAY_REPLAN_AT_OPEN", signal_date=None)
            opening_plan = opening_plan_provider(
                date_key, deepcopy(opening), day[["open", "can_buy", "can_sell"]].copy()
            )
            audit = {"trade_date": date_key, "plan": deepcopy(opening_plan)}
            opening_audits.append(audit)
            if opening_plan.get("status") == "PASS_REFERENCE_QUOTES_ONLY":
                from .research.integer_execution import paper_intents_from_integer_plan

                signal_date = str(opening_plan.get("signal_date", ""))
                if len(signal_date) != 8 or not signal_date.isdigit() or signal_date >= date_key:
                    raise ValueError("opening plan must reference a prior closing signal")
                for index, risk in enumerate(
                    paper_intents_from_integer_plan(opening_plan, strategy_id, signal_date)
                ):
                    execution_order[str(opening_plan["orders"][index]["symbol"])] = index
                    broker.submit(risk, account_id)
        pending_orders = [
            row
            for row in broker.orders()
            if row["account_id"] == account_id and row["status"] == "PENDING"
        ]
        pending_sells = {str(row["symbol"]) for row in pending_orders if row["side"] == "SELL"}
        pending_symbols = {str(row["symbol"]) for row in pending_orders}
        execution_symbols = sorted(
            set(day.index.astype(str)) & pending_symbols,
            key=lambda s: (s not in pending_sells, execution_order.get(s, 0), s),
        )
        for symbol in execution_symbols:
            row = day.loc[symbol]
            broker.execute_pending(
                ExecutionBar(
                    symbol=symbol,
                    trade_date=date_key,
                    open=float(row["open"]),
                    close=float(row["close"]),
                    can_buy=bool(row["can_buy"]),
                    can_sell=bool(row["can_sell"]),
                ),
                config,
                account_id,
            )

        if opening_plan is not None and opening_plan.get("status") == "PASS_REFERENCE_QUOTES_ONLY":
            actual = broker.account_snapshot(open_marks, account_id)
            actual_quantities = {str(p["symbol"]): int(p["quantity"]) for p in actual["positions"]}
            expected = opening_plan["target_quantities"]
            quantities_match = all(
                actual_quantities.get(s, 0) == int(expected.get(s, 0))
                for s in set(actual_quantities) | set(expected)
            )
            cash_error = float(actual["cash"]) - float(opening_plan["estimated_cash"])
            opening_audits[-1]["execution_check"] = {
                "status": "MATCHED"
                if quantities_match and abs(cash_error) <= 0.01
                else "EXECUTION_MISMATCH",
                "quantities_match": quantities_match,
                "cash_error": cash_error,
                "actual_quantities": actual_quantities,
            }
            if not quantities_match or abs(cash_error) > 0.01:
                raise ValueError("integer opening plan execution mismatch; research replay stopped")
            proof = opening_plan.get("execution_style_proof")
            if proof is not None:
                z = pd.DataFrame.from_dict(proof["standardized"], orient="index")
                values = pd.Series(
                    {s: q * open_marks[s] for s, q in actual_quantities.items() if q > 0}
                )
                if set(values.index) - set(z.index) or values.sum() <= 0:
                    raise ValueError("actual opening holdings lack frozen style evidence")
                active = z.reindex(values.index).T.dot(values / values.sum())
                bounds = pd.Series(proof["limits"]).reindex(active.index)
                excess = float((active.abs() - bounds).max())
                opening_audits[-1]["execution_check"].update(
                    actual_style_exposures=active.to_dict(),
                    maximum_style_excess=excess,
                    style_basis="frozen_prior_close_factors_actual_open_share_values",
                )
                if not np.isfinite(active).all() or not np.isfinite(bounds).all() or excess > 1e-5:
                    raise ValueError(
                        "actual opening style boundary failed; research replay stopped"
                    )

        if not actions.empty:
            today_actions = actions.loc[actions["record_date"] == pd.Timestamp(trade_date)]
            for action in today_actions.to_dict("records"):
                broker.register_corporate_entitlement(
                    account_id,
                    str(action["symbol"]),
                    str(action["action_key"]),
                    date_key,
                    float(action["cash_per_share"]),
                    float(action["stock_ratio"]),
                    _date_key(action["cash_pay_date"]),
                    _date_key(action["stock_list_date"]),
                )

        snapshot = broker.account_snapshot(marks, account_id)
        if accrue_action_receivables:
            from .research.replay_actions import economic_snapshot

            snapshot = economic_snapshot(broker, snapshot, marks, account_id, date_key, actions)
        positions = {
            str(row["symbol"]): row
            for row in snapshot["positions"]  # type: ignore[index]
        }
        today_targets = instructions.loc[instructions["date"] == pd.Timestamp(trade_date)]
        if closing_target_provider is not None:
            if not instructions.empty:
                raise ValueError("dynamic and static targets cannot be mixed")
            # Only closing information and this replay's own actual holdings.
            # Never seed each day from a production account's historical positions.
            generated = closing_target_provider(date_key, deepcopy(snapshot))
            today_targets = _prepare_targets(generated)
            if not today_targets.empty and set(today_targets.date) != {pd.Timestamp(trade_date)}:
                raise ValueError("dynamic targets must match current closing date")
            if not today_targets.empty:
                broker.cancel_pending(account_id, "REPLAY_SUPERSEDED_AT_CLOSE", signal_date=None)
        if not today_targets.empty and opening_plan_provider is None:
            target_state.update(
                {
                    str(row["symbol"]): float(row["target_weight"])
                    for row in today_targets.to_dict("records")
                }
            )
            for symbol in sorted(set(target_state) | set(positions)):
                if symbol not in marks:
                    continue
                position = positions.get(symbol, {})
                if target_state.get(symbol, 0.0) == 0 and int(position.get("quantity", 0)) == 0:
                    continue
                decision = SignalDecision(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    as_of_date=date_key,
                    target_weight=max(
                        0.0,
                        float(target_state.get(symbol, 0.0))
                        - snapshot.get("unsettled_stock_rights", {}).get(symbol, 0)
                        * marks[symbol]
                        / float(snapshot["equity"]),
                    ),
                    action="HOLD" if target_state.get(symbol, 0.0) > 0 else "EXIT",
                    reason_code="HISTORICAL_REPLAY_TARGET",
                    close=marks[symbol],
                    fast_ma=None,
                    slow_ma=None,
                    history_bars=0,
                )
                risk = create_order_intent(
                    decision,
                    PortfolioSnapshot(
                        cash=float(snapshot["cash"]),
                        equity=float(snapshot["equity"]),
                        position_quantity=int(position.get("quantity", 0)),
                        reference_price=marks[symbol],
                    ),
                    lot_size=config.lot_size,
                    max_symbol_weight=config.max_position_weight,
                )
                if (
                    closing_target_provider is not None
                    and risk.rejection_code == "NO_REBALANCE_REQUIRED"
                ):
                    continue
                broker.record_decision(decision)
                broker.submit(risk, account_id)

        closing = broker.account_snapshot(marks, account_id)
        if accrue_action_receivables:
            closing = economic_snapshot(broker, closing, marks, account_id, date_key, actions)
        equity = float(closing["equity"])
        market_value = float(closing["market_value"])
        equity_rows.append(
            {
                "trade_date": date_key,
                "equity": equity,
                "cash": float(closing["cash"]),
                "market_value": market_value,
                "receivable_cash": float(closing.get("receivable_cash", 0.0)),
                "unsettled_stock_value": float(closing.get("unsettled_stock_value", 0.0)),
                "gross_exposure": market_value / equity if equity else 0.0,
                "daily_return": equity / previous_equity - 1.0,
            }
        )
        previous_equity = equity

    equity_frame = pd.DataFrame(equity_rows)
    if not equity_frame.empty:
        equity_frame["normalized_nav"] = equity_frame["equity"] / float(
            equity_frame["equity"].iloc[0]
        )
        equity_frame["drawdown"] = (
            equity_frame["normalized_nav"] / equity_frame["normalized_nav"].cummax() - 1.0
        )
    fills = broker.fills(account_id)
    orders = [row for row in broker.orders() if str(row["account_id"]) == account_id]
    attempts = broker.execution_attempts(account_id=account_id)
    reconciliation = asdict(broker.reconcile(account_id))
    summary = _summary(equity_frame, orders, fills, attempts, reconciliation)
    if opening_plan_provider is not None:
        summary["opening_plan_audits"] = opening_audits
    return ExecutionReplayResult(
        strategy_id,
        account_id,
        equity_frame,
        orders,
        fills,
        attempts,
        reconciliation,
        summary,
    )


def _prepare_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing bar columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["symbol"] = result["symbol"].astype(str)
    result["open"] = pd.to_numeric(result["open"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    for column in ("can_buy", "can_sell"):
        if column not in result:
            result[column] = True
        result[column] = result[column].astype("boolean").fillna(True).astype(bool)
    return result.dropna(subset=["date", "open", "close"]).sort_values(["date", "symbol"])


def _prepare_targets(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "target_weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing target columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["symbol"] = result["symbol"].astype(str)
    result["target_weight"] = pd.to_numeric(result["target_weight"], errors="coerce")
    if (result["target_weight"].dropna() < 0).any():
        raise ValueError("target weights must be non-negative")
    daily_gross = result.groupby("date")["target_weight"].sum()
    if (daily_gross > 1.0 + 1e-9).any():
        raise ValueError("daily target gross exposure cannot exceed 1")
    return result.dropna(subset=["date", "target_weight"]).sort_values(["date", "symbol"])


def _prepare_actions(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "action_key",
                "record_date",
                "cash_per_share",
                "stock_ratio",
                "cash_pay_date",
                "stock_list_date",
            ]
        )
    required = {
        "symbol",
        "action_key",
        "record_date",
        "cash_per_share",
        "stock_ratio",
        "cash_pay_date",
        "stock_list_date",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing corporate action columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("record_date", "cash_pay_date", "stock_list_date", "ex_date"):
        if column not in result:
            continue
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def _date_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y%m%d")


def _summary(
    equity: pd.DataFrame,
    orders: list[dict[str, object]],
    fills: list[dict[str, object]],
    attempts: list[dict[str, object]],
    reconciliation: dict[str, Any],
) -> dict[str, object]:
    if equity.empty:
        return {"observation_days": 0, "reconciled": reconciliation["matched"]}
    returns = equity["equity"].pct_change().dropna()
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else None
    return {
        "start_date": str(equity.iloc[0]["trade_date"]),
        "end_date": str(equity.iloc[-1]["trade_date"]),
        "observation_days": len(equity),
        "total_return": float(equity.iloc[-1]["normalized_nav"] - 1.0),
        "annualized_volatility": volatility,
        "sharpe": (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
            if len(returns) > 1 and returns.std(ddof=1) > 0
            else None
        ),
        "max_drawdown": float(equity["drawdown"].min()),
        "average_gross_exposure": float(equity["gross_exposure"].mean()),
        "order_count": len(orders),
        "fill_count": len(fills),
        "deferred_count": sum(str(row.get("outcome")) == "DEFERRED" for row in attempts),
        "transaction_cost": float(sum(float(row["fee"]) for row in fills)),
        "reconciled": bool(reconciliation["matched"]),
    }
