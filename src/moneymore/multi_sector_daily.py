from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import BacktestConfig
from .corporate_actions import process_corporate_actions
from .data.health import run_data_health_checks
from .data.research import load_total_return_stock_bars
from .data.store import ParquetStore
from .execution.paper import ExecutionBar, PaperBroker
from .execution.risk import PortfolioSnapshot, create_order_intent
from .execution.risk_state import (
    constrained_target_weight,
    evaluate_account_risk_state,
)
from .model_registry import ModelRegistry
from .monthly_acceptance import evaluate_monthly_cycle, freeze_passed_report
from .signals import SignalDecision, write_signal_artifact
from .strategy_universe import active_strategy_universe, execution_strategy_id, universe_summary

ROOT = Path(__file__).resolve().parents[2]
MULTI_SECTOR_ACCOUNT = "multi_sector_shadow"
MULTI_SECTOR_STRATEGY = "multi_sector_dynamic_v1"


@dataclass(frozen=True)
class MultiSectorDailyResult:
    trade_date: str
    status: str
    target_weights: dict[str, float]
    theoretical_target_weights: dict[str, float]
    target_adjustments: list[dict[str, object]]
    symbol_sectors: dict[str, str]
    executions: list[dict[str, object]]
    orders: list[dict[str, object]]
    portfolio: dict[str, object]
    attribution: list[dict[str, object]]
    return_attribution: list[dict[str, object]]
    risk_attribution: list[dict[str, object]]
    execution_attribution: list[dict[str, object]]
    attribution_reconciliation: dict[str, object]
    deviations: list[dict[str, object]]
    metrics: dict[str, object]
    risk_alerts: list[dict[str, object]]
    corporate_actions: dict[str, object]
    data_health: dict[str, object]
    risk_state: dict[str, object]
    reconciliation: dict[str, object]
    model_version: dict[str, object]
    report_path: str


def expand_sleeve_targets(recommendation: object) -> tuple[
    dict[str, float], dict[str, str]
]:
    raw_weights: dict[str, float] = {}
    symbol_sectors: dict[str, str] = {}
    primary_contribution: dict[str, float] = {}
    for row in recommendation.to_dict("records"):  # type: ignore[attr-defined]
        symbols = [item for item in str(row["selected"]).split(",") if item]
        if not symbols:
            continue
        weight = float(row["target_weight"]) / len(symbols)
        for symbol in symbols:
            raw_weights[symbol] = raw_weights.get(symbol, 0.0) + weight
            if weight > primary_contribution.get(symbol, -1):
                primary_contribution[symbol] = weight
                symbol_sectors[symbol] = str(row["sector"])
    gross_budget = sum(raw_weights.values())
    target_weights = _capped_pro_rata_allocations(gross_budget, raw_weights, 0.10)
    return target_weights, symbol_sectors


def executable_target_weights(
    theoretical: dict[str, float],
    sectors: dict[str, str],
    marks: dict[str, float],
    equity: float,
    lot_size: int,
    max_symbol_weight: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Convert model weights into an A-share whole-lot executable portfolio.

    Infeasible names do not consume a sleeve's displayed target exposure. Their
    budget is redistributed pro-rata within the same sleeve, subject to the
    single-name cap. If no feasible name remains, the sleeve budget stays cash.
    """
    if equity <= 0:
        return {}, []
    adjustments: list[dict[str, object]] = []
    budget = sum(float(weight) for weight in theoretical.values() if weight > 0)
    candidates = {
        symbol
        for symbol, weight in theoretical.items()
        if weight > 0
        and float(marks.get(symbol, 0)) > 0
        and float(marks[symbol]) * lot_size / equity <= max_symbol_weight + 1e-12
    }
    allocations: dict[str, float] = {}
    while candidates:
        allocations = _capped_pro_rata_allocations(
            budget,
            {symbol: float(theoretical[symbol]) for symbol in candidates},
            max_symbol_weight,
        )
        infeasible = {
            symbol
            for symbol, weight in allocations.items()
            if weight + 1e-12 < float(marks[symbol]) * lot_size / equity
        }
        if not infeasible:
            break
        candidates -= infeasible
    else:
        allocations = {}

    for symbol, theoretical_weight_raw in theoretical.items():
        theoretical_weight = float(theoretical_weight_raw)
        executable_weight = float(allocations.get(symbol, 0.0))
        if abs(executable_weight - theoretical_weight) <= 1e-12:
            continue
        lot_weight = (
            float(marks.get(symbol, 0)) * lot_size / equity
            if marks.get(symbol)
            else None
        )
        adjustments.append(
            {
                "symbol": symbol,
                "sector": sectors.get(symbol, "unmapped"),
                "theoretical_weight": theoretical_weight,
                "executable_weight": executable_weight,
                "minimum_lot_weight": lot_weight,
                "reason": (
                    "INFEASIBLE_ONE_LOT"
                    if executable_weight == 0
                    else "PORTFOLIO_BUDGET_REALLOCATED"
                ),
            }
        )
    return allocations, adjustments


def _capped_pro_rata_allocations(
    budget: float,
    source_weights: dict[str, float],
    cap: float,
) -> dict[str, float]:
    allocations = {symbol: 0.0 for symbol in source_weights}
    active = set(source_weights)
    remaining = budget
    while active and remaining > 1e-12:
        denominator = sum(source_weights[symbol] for symbol in active)
        if denominator <= 0:
            break
        proposed = {
            symbol: remaining * source_weights[symbol] / denominator
            for symbol in active
        }
        capped = {
            symbol
            for symbol, increment in proposed.items()
            if allocations[symbol] + increment >= cap - 1e-12
        }
        if not capped:
            for symbol, increment in proposed.items():
                allocations[symbol] += increment
            remaining = 0.0
            break
        for symbol in capped:
            addition = max(0.0, cap - allocations[symbol])
            allocations[symbol] += addition
            remaining -= addition
            active.remove(symbol)
    return {symbol: weight for symbol, weight in allocations.items() if weight > 1e-12}


def run_multi_sector_daily(
    store: ParquetStore,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
    signal_dir: str | Path,
    report_dir: str | Path,
    planning_only: bool = False,
) -> MultiSectorDailyResult:
    execution_model_id = execution_strategy_id(MULTI_SECTOR_STRATEGY, store)
    broker.initialize_account(config.initial_cash, MULTI_SECTOR_ACCOUNT)
    recommendation = store.read("global_factor_recommendation")
    selected_recommendation = recommendation.loc[
        recommendation["selected"].astype(bool)
    ].copy()
    theoretical_target_weights = {
        str(row["symbol"]): float(row["target_weight"])
        for row in selected_recommendation.to_dict("records")
    }
    symbol_sectors = {
        str(row["symbol"]): str(row.get("sector", "unmapped"))
        for row in recommendation.to_dict("records")
    }
    target_weights = theoretical_target_weights
    data_health_report = run_data_health_checks(
        store, trade_date, set(target_weights), score_table='global_factor_recommendation'
    )
    data_health = asdict(data_health_report)
    quality_ok = data_health_report.status != "BLOCKED"

    account_before = broker.account_snapshot({}, MULTI_SECTOR_ACCOUNT)
    position_symbols = {
        str(row["symbol"]) for row in account_before["positions"]  # type: ignore[index]
    }
    execution_symbols = {
        str(row["symbol"])
        for row in broker.fills(MULTI_SECTOR_ACCOUNT, trade_date)
    }
    symbols = sorted(set(target_weights) | position_symbols | execution_symbols)
    bars: dict[str, ExecutionBar] = {}
    marks: dict[str, float] = {}
    previous_marks: dict[str, float] = {}
    latest_dates: dict[str, str] = {}
    for symbol in symbols:
        history = load_total_return_stock_bars(store, symbol)
        eligible = history.loc[
            history["date"].astype(str).str.replace("-", "").str[:8] <= trade_date
        ]
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        latest_dates[symbol] = str(row["date"]).replace("-", "")[:8]
        marks[symbol] = float(row["raw_close"])
        if len(eligible) > 1:
            previous_marks[symbol] = float(eligible.iloc[-2]["raw_close"])
        if str(row["date"]).replace("-", "")[:8] == trade_date:
            bars[symbol] = ExecutionBar(
                symbol=symbol,
                trade_date=trade_date,
                open=float(row["raw_open"]),
                close=float(row["raw_close"]),
                can_buy=bool(row.get("can_buy", True)),
                can_sell=bool(row.get("can_sell", True)),
            )

    stale_symbols = sorted(
        symbol for symbol in target_weights if latest_dates.get(symbol) != trade_date
    )
    state_dir = Path(report_dir).parent
    registry = ModelRegistry(
        state_dir / "model_registry.sqlite3",
        state_dir / "model-versions",
    )
    version = registry.register(
        model_id=MULTI_SECTOR_STRATEGY,
        lifecycle="SHADOW",
        evidence_stage="FORWARD_PAPER",
        data_cutoff=trade_date,
        universe=list(target_weights),
        config_files=[
            ROOT / "configs" / "default.yaml",
            ROOT / "configs" / "sector_models.yaml",
        ],
        code_files=[
            ROOT / "src" / "moneymore" / "multi_sector_daily.py",
            ROOT / "src" / "moneymore" / "research" / "sector_model.py",
            ROOT / "src" / "moneymore" / "research" / "bank_model.py",
            ROOT / "src" / "moneymore" / "research" / "bank_timing.py",
        ],
        metadata={
            "account_id": MULTI_SECTOR_ACCOUNT,
            "execution": "T+1_OPEN_PAPER",
            "evidence_boundary": "20260727",
        },
    )
    registry.add_artifact(
        version.version_id,
        "RESEARCH_TABLE",
        "HISTORICAL_DIAGNOSTIC",
        "data/curated/sector_model_report",
    )
    registry.add_artifact(
        version.version_id,
        "PORTFOLIO_RECOMMENDATION",
        "FORWARD_CANDIDATE",
        f"global_factor_recommendation:{trade_date}",
    )
    data_fresh = not stale_symbols
    pre_portfolio = broker.account_snapshot(marks, MULTI_SECTOR_ACCOUNT)
    pre_reconciliation = asdict(broker.reconcile(MULTI_SECTOR_ACCOUNT))
    previous_drawdown = _previous_drawdown(store)
    risk_decision = evaluate_account_risk_state(
        config=config,
        equity=float(pre_portfolio["equity"]),
        cash=float(pre_portfolio["cash"]),
        gross_exposure=(
            float(pre_portfolio["market_value"]) / float(pre_portfolio["equity"])
            if pre_portfolio["equity"]
            else 0.0
        ),
        drawdown=previous_drawdown,
        data_health_status=data_health_report.status,
        reconciled=bool(pre_reconciliation["matched"]),
    )
    risk_state = broker.update_risk_state(
        MULTI_SECTOR_ACCOUNT,
        risk_decision.proposed_state,
        risk_decision.reason_code,
        trade_date,
    )
    effective_risk_state = str(risk_state["effective_state"])
    if effective_risk_state in {"REDUCE_ONLY", "SELL_ONLY"}:
        broker.cancel_pending(
            MULTI_SECTOR_ACCOUNT, effective_risk_state, side="BUY"
        )
    elif effective_risk_state == "SUSPENDED":
        broker.cancel_pending(MULTI_SECTOR_ACCOUNT, "RISK_SUSPENDED")
    pending_sells = {
        str(row["symbol"])
        for row in broker.orders()
        if row.get("account_id") == MULTI_SECTOR_ACCOUNT
        and row.get("status") == "PENDING"
        and row.get("side") == "SELL"
    }
    execution_order = (
        sorted(bars, key=lambda symbol: symbol not in pending_sells)
        if quality_ok and effective_risk_state != "SUSPENDED" and not planning_only
        else []
    )
    executions = []
    for symbol in execution_order:
        executions.extend(
            broker.execute_pending(
                bars[symbol], config, account_id=MULTI_SECTOR_ACCOUNT
            )
        )

    corporate_actions = {"settled": [], "registered": []} if planning_only else process_corporate_actions(
        store, broker, MULTI_SECTOR_ACCOUNT, trade_date, set(symbols)
    )
    account = broker.account_snapshot(marks, MULTI_SECTOR_ACCOUNT)
    target_weights, target_adjustments = executable_target_weights(
        theoretical_target_weights,
        symbol_sectors,
        marks,
        float(account["equity"]),
        config.lot_size,
        config.max_position_weight,
    )
    positions = {
        str(row["symbol"]): row for row in account["positions"]  # type: ignore[index]
    }
    orders = []
    decision_symbols = (
        sorted(set(target_weights) | set(positions))
        if data_fresh and quality_ok and effective_risk_state != "SUSPENDED"
        else []
    )
    for symbol in decision_symbols:
        close = marks.get(symbol)
        if close is None:
            continue
        model_target = float(target_weights.get(symbol, 0))
        sector = symbol_sectors.get(symbol, "exit")
        position = positions.get(symbol, {})
        actual_weight = (
            int(position.get("quantity", 0)) * close / float(account["equity"])
            if account["equity"]
            else 0.0
        )
        constrained = constrained_target_weight(
            effective_risk_state, model_target, actual_weight
        )
        if constrained is None:
            continue
        target = constrained
        decision = SignalDecision(
            strategy_id=execution_model_id,
            symbol=symbol,
            as_of_date=trade_date,
            target_weight=target,
            action="HOLD" if target > 0 else "EXIT",
            reason_code=f"{sector.upper()}_TARGET" if target > 0 else "PORTFOLIO_EXIT",
            close=close,
            fast_ma=None,
            slow_ma=None,
            history_bars=0,
        )
        risk = create_order_intent(
            decision,
            PortfolioSnapshot(
                cash=float(account["cash"]),
                equity=float(account["equity"]),
                position_quantity=int(position.get("quantity", 0)),
                reference_price=close,
            ),
            lot_size=config.lot_size,
            max_symbol_weight=config.max_position_weight,
        )
        signal_path = write_signal_artifact(decision, signal_dir)
        decision_status = broker.record_decision(decision)
        submit_status = (
            broker.submit(risk, MULTI_SECTOR_ACCOUNT)
            if decision_status in {"RECORDED", "DUPLICATE"}
            else "DUPLICATE_DECISION"
        )
        signal_key = (
            f"{decision.strategy_id}:{decision.symbol}:{decision.as_of_date}"
        )
        registry.bind(
            version.version_id,
            "SIGNAL",
            signal_key,
            trade_date,
            symbol,
        )
        registry.add_artifact(
            version.version_id,
            "SIGNAL_ARTIFACT",
            "SHADOW_PAPER",
            str(signal_path),
        )
        if risk.intent is not None:
            registry.bind(
                version.version_id,
                "ORDER",
                risk.intent.idempotency_key,
                trade_date,
                symbol,
            )
        orders.append(
            {
                "symbol": symbol,
                "sector": sector,
                "target_weight": target,
                "model_target_weight": model_target,
                "risk_state": effective_risk_state,
                "decision_status": decision_status,
                "submit_status": submit_status,
                "rejection_code": risk.rejection_code,
            }
        )

    portfolio = broker.account_snapshot(marks, MULTI_SECTOR_ACCOUNT)
    fills_today = broker.fills(MULTI_SECTOR_ACCOUNT, trade_date)
    daily_contribution = _daily_pnl_contribution(
        account_before,
        fills_today,
        marks,
        previous_marks,
    )
    attribution = _sleeve_attribution(
        portfolio,
        marks,
        symbol_sectors,
        daily_contribution,
        corporate_actions["settled"],  # type: ignore[arg-type]
    )
    deviations = _weight_deviations(
        portfolio, marks, target_weights, symbol_sectors, orders, config.lot_size
    )
    return_attribution, attribution_reconciliation = _return_attribution(
        account_before=account_before,
        portfolio=portfolio,
        fills=fills_today,
        marks=marks,
        previous_marks=previous_marks,
        bars=bars,
        sectors=symbol_sectors,
        corporate_actions=corporate_actions["settled"],  # type: ignore[arg-type]
    )
    risk_attribution = _portfolio_risk_attribution(
        store, trade_date, portfolio, marks, symbol_sectors
    )
    execution_attribution = _execution_attribution(deviations, executions)
    reconciliation = asdict(broker.reconcile(MULTI_SECTOR_ACCOUNT))
    metrics = _daily_metrics(
        store, trade_date, portfolio, target_weights, reconciliation
    )
    metrics["theoretical_target_exposure"] = sum(
        theoretical_target_weights.values()
    )
    risk_alerts = _risk_alerts(
        config,
        metrics,
        deviations,
        stale_symbols,
        reconciliation,
        data_health_report.checks,
    )
    status = (
        "BLOCKED_STALE_DATA"
        if stale_symbols
        else "BLOCKED_DATA_QUALITY"
        if not quality_ok
        else "SUSPENDED_RISK"
        if effective_risk_state == "SUSPENDED"
        else "PLAN_READY" if planning_only and reconciliation["matched"]
        else "COMPLETED"
        if reconciliation["matched"]
        else "FAILED_RECONCILIATION"
    )
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{trade_date}.json"
    payload = {
        "planning_only": planning_only,
        "trade_date": trade_date,
        "status": status,
        "target_weights": target_weights,
        "theoretical_target_weights": theoretical_target_weights,
        "target_adjustments": target_adjustments,
        "symbol_sectors": symbol_sectors,
        "executions": executions,
        "orders": orders,
        "portfolio": portfolio,
        "attribution": attribution,
        "return_attribution": return_attribution,
        "risk_attribution": risk_attribution,
        "execution_attribution": execution_attribution,
        "attribution_reconciliation": attribution_reconciliation,
        "deviations": deviations,
        "metrics": metrics,
        "risk_alerts": risk_alerts,
        "corporate_actions": corporate_actions,
        "data_health": data_health,
        "risk_state": {
            **risk_state,
            "explanation": risk_decision.explanation,
        },
        "reconciliation": reconciliation,
        "model_version": asdict(version),
        "strategy_universe": universe_summary(active_strategy_universe(store)),
        "created_at": datetime.now(UTC).isoformat(),
        "report_path": str(target),
    }
    should_replace = False
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        should_replace = (
            bool(existing.get('planning_only'))
            or
            existing.get("strategy_universe", {}).get("universe_version")
            != payload["strategy_universe"]["universe_version"]
            or (
                existing.get("status") in {"BLOCKED_DATA_QUALITY", "SUSPENDED_RISK"}
                and payload["status"] != existing.get("status")
            )
        )
    if not target.exists() or should_replace:
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    else:
        payload["created_at"] = existing["created_at"]
    if not planning_only:
      _persist_daily_analytics(
        store,
        trade_date,
        status,
        metrics,
        attribution,
        deviations,
        risk_alerts,
        return_attribution,
        risk_attribution,
        execution_attribution,
        attribution_reconciliation,
    )
    account_history = store.read("multi_sector_account_daily")
    acceptance = evaluate_monthly_cycle(
        account_daily=account_history.to_dict("records"),
        fills=broker.fills(MULTI_SECTOR_ACCOUNT),
        model_versions=registry.snapshot(100)["versions"],
        start_date="20260727",
        observation_date=trade_date,
    )
    if not planning_only: freeze_passed_report(
        acceptance,
        state_dir / "monthly-acceptance",
    )
    result = MultiSectorDailyResult(
        **{key: payload[key] for key in MultiSectorDailyResult.__dataclass_fields__}
    )
    if not reconciliation["matched"]:
        raise RuntimeError(f"multi-sector reconciliation failed: {result.report_path}")
    return result


def _previous_drawdown(store: ParquetStore) -> float:
    try:
        history = store.read("multi_sector_account_daily").sort_values("trade_date")
    except FileNotFoundError:
        return 0.0
    return float(history.iloc[-1]["drawdown"]) if not history.empty else 0.0


def _sleeve_attribution(
    portfolio: dict[str, object],
    marks: dict[str, float],
    symbol_sectors: dict[str, str],
    daily_contribution: dict[str, float] | None = None,
    corporate_actions: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, float]] = {}
    equity = float(portfolio["equity"])
    for position in portfolio["positions"]:  # type: ignore[union-attr]
        symbol = str(position["symbol"])
        sector = symbol_sectors.get(symbol, "unmapped")
        quantity = int(position["quantity"])
        mark = float(marks.get(symbol, 0))
        market_value = quantity * mark
        cost = quantity * float(position["avg_cost"])
        bucket = grouped.setdefault(
            sector,
            {
                "market_value": 0.0,
                "cost": 0.0,
                "unrealized_pnl": 0.0,
                "daily_pnl": 0.0,
            },
        )
        bucket["market_value"] += market_value
        bucket["cost"] += cost
        bucket["unrealized_pnl"] += market_value - cost
        bucket["daily_pnl"] += (daily_contribution or {}).get(symbol, 0.0)
    for action in corporate_actions or []:
        symbol = str(action["symbol"])
        sector = symbol_sectors.get(symbol, "unmapped")
        bucket = grouped.setdefault(
            sector,
            {
                "market_value": 0.0,
                "cost": 0.0,
                "unrealized_pnl": 0.0,
                "daily_pnl": 0.0,
            },
        )
        bucket["daily_pnl"] += float(action.get("cash_amount", 0))
    return [
        {
            "sector": sector,
            **values,
            "actual_weight": values["market_value"] / equity if equity else 0.0,
            "daily_contribution": values["daily_pnl"] / equity if equity else 0.0,
        }
        for sector, values in sorted(grouped.items())
    ]


def _daily_pnl_contribution(
    account_before: dict[str, object],
    fills: list[dict[str, object]],
    marks: dict[str, float],
    previous_marks: dict[str, float],
) -> dict[str, float]:
    starting = {
        str(row["symbol"]): int(row["quantity"])
        for row in account_before["positions"]  # type: ignore[index]
    }
    by_symbol: dict[str, list[dict[str, object]]] = {}
    for fill in fills:
        by_symbol.setdefault(str(fill["symbol"]), []).append(fill)
    result: dict[str, float] = {}
    for symbol in set(starting) | set(by_symbol):
        close = float(marks.get(symbol, 0))
        previous = float(previous_marks.get(symbol, close))
        sold = sum(
            int(fill["quantity"])
            for fill in by_symbol.get(symbol, [])
            if fill["side"] == "SELL"
        )
        pnl = max(0, starting.get(symbol, 0) - sold) * (close - previous)
        for fill in by_symbol.get(symbol, []):
            quantity = int(fill["quantity"])
            price = float(fill["price"])
            fee = float(fill["fee"])
            if fill["side"] == "BUY":
                pnl += quantity * (close - price) - fee
            else:
                pnl += quantity * (price - previous) - fee
        result[symbol] = pnl
    return result


def _return_attribution(
    *,
    account_before: dict[str, object],
    portfolio: dict[str, object],
    fills: list[dict[str, object]],
    marks: dict[str, float],
    previous_marks: dict[str, float],
    bars: dict[str, ExecutionBar],
    sectors: dict[str, str],
    corporate_actions: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    starting = {
        str(row["symbol"]): int(row["quantity"])
        for row in account_before["positions"]  # type: ignore[index]
    }
    by_symbol: dict[str, list[dict[str, object]]] = {}
    for fill in fills:
        by_symbol.setdefault(str(fill["symbol"]), []).append(fill)
    retained: dict[str, int] = {}
    returns: dict[str, float] = {}
    for symbol, quantity in starting.items():
        sold = sum(
            int(fill["quantity"])
            for fill in by_symbol.get(symbol, [])
            if fill["side"] == "SELL"
        )
        retained[symbol] = max(quantity - sold, 0)
        previous = float(previous_marks.get(symbol, marks.get(symbol, 0)))
        close = float(marks.get(symbol, previous))
        returns[symbol] = close / previous - 1 if previous else 0.0
    sector_returns: dict[str, float] = {}
    for sector in sorted(set(sectors.values()) | {"unmapped"}):
        values = [
            returns[symbol]
            for symbol, quantity in retained.items()
            if quantity > 0 and sectors.get(symbol, "unmapped") == sector
        ]
        sector_returns[sector] = sum(values) / len(values) if values else 0.0

    rows: list[dict[str, object]] = []

    def add(component: str, sector: str, amount: float, detail: str) -> None:
        if abs(amount) > 1e-12 or component == "CASH":
            rows.append(
                {
                    "component": component,
                    "sector": sector,
                    "pnl": amount,
                    "detail": detail,
                }
            )

    for symbol, quantity in retained.items():
        if quantity <= 0:
            continue
        previous = float(previous_marks.get(symbol, marks.get(symbol, 0)))
        amount = quantity * (float(marks.get(symbol, previous)) - previous)
        sector = sectors.get(symbol, "unmapped")
        benchmark = quantity * previous * sector_returns.get(sector, 0.0)
        add("SECTOR_ALLOCATION", sector, benchmark, "期初持仓×行业等权收益")
        add("STOCK_SELECTION", sector, amount - benchmark, f"{symbol}相对行业收益")

    for fill in fills:
        symbol = str(fill["symbol"])
        sector = sectors.get(symbol, "unmapped")
        quantity = int(fill["quantity"])
        price = float(fill["price"])
        fee = float(fill["fee"])
        open_price = float(bars[symbol].open)
        close = float(marks.get(symbol, open_price))
        previous = float(previous_marks.get(symbol, open_price))
        if fill["side"] == "BUY":
            timing = quantity * (close - open_price)
            slippage = quantity * (price - open_price)
        else:
            timing = quantity * (open_price - previous)
            slippage = quantity * (open_price - price)
        add("TIMING", sector, timing, f"{symbol}开盘交易后的价格路径")
        add("SLIPPAGE", sector, -slippage, f"{symbol}成交价与开盘价差")
        add("TRANSACTION_COST", sector, -fee, f"{symbol}佣金税费")

    for action in corporate_actions:
        symbol = str(action["symbol"])
        sector = sectors.get(symbol, "unmapped")
        cash = float(action.get("cash_amount", 0))
        stock_value = int(action.get("share_quantity", 0)) * float(
            marks.get(symbol, 0)
        )
        add("DIVIDEND", sector, cash + stock_value, f"{symbol}现金或股份入账")
    add("CASH", "cash", 0.0, "现金收益率按0计")

    starting_equity = float(account_before["cash"]) + sum(
        quantity * float(previous_marks.get(symbol, marks.get(symbol, 0)))
        for symbol, quantity in starting.items()
    )
    ending_equity = float(portfolio["equity"])
    explained_pnl = sum(float(row["pnl"]) for row in rows)
    actual_pnl = ending_equity - starting_equity
    for row in rows:
        row["contribution"] = (
            float(row["pnl"]) / starting_equity if starting_equity else 0.0
        )
    reconciliation = {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "actual_pnl": actual_pnl,
        "explained_pnl": explained_pnl,
        "unexplained_pnl": actual_pnl - explained_pnl,
        "matched": abs(actual_pnl - explained_pnl) <= 0.01,
    }
    return rows, reconciliation


def _portfolio_risk_attribution(
    store: ParquetStore,
    trade_date: str,
    portfolio: dict[str, object],
    marks: dict[str, float],
    sectors: dict[str, str],
) -> list[dict[str, object]]:
    equity = float(portfolio["equity"])
    weights = {
        str(row["symbol"]): int(row["quantity"]) * float(marks.get(row["symbol"], 0))
        / equity
        for row in portfolio["positions"]  # type: ignore[index]
        if equity and int(row["quantity"]) > 0
    }
    panels = []
    for symbol in weights:
        history = load_total_return_stock_bars(store, symbol)
        eligible = history.loc[
            history["date"].astype(str).str.replace("-", "").str[:8] <= trade_date
        ].tail(61)
        if len(eligible) < 20:
            continue
        series = eligible.set_index("date")["signal_close"].pct_change().dropna()
        panels.append(series.rename(symbol))
    if not panels:
        return []
    returns = pd.concat(panels, axis=1).dropna()
    symbols = [symbol for symbol in returns.columns if symbol in weights]
    if not symbols:
        return []
    covariance = returns[symbols].cov() * 252
    weight_series = pd.Series({symbol: weights[symbol] for symbol in symbols})
    portfolio_variance = float(
        weight_series @ covariance @ weight_series
    )
    portfolio_volatility = portfolio_variance**0.5 if portfolio_variance > 0 else 0.0
    marginal = covariance @ weight_series
    concentration_total = float((weight_series**2).sum())
    rows = []
    for symbol in symbols:
        variance_contribution = float(weight_series[symbol] * marginal[symbol])
        standalone = float(
            weight_series[symbol] ** 2 * covariance.loc[symbol, symbol]
        )
        rows.append(
            {
                "symbol": symbol,
                "sector": sectors.get(symbol, "unmapped"),
                "weight": float(weight_series[symbol]),
                "standalone_volatility": float(
                    covariance.loc[symbol, symbol] ** 0.5
                ),
                "volatility_contribution": (
                    variance_contribution / portfolio_volatility
                    if portfolio_volatility
                    else 0.0
                ),
                "correlation_contribution": (
                    (variance_contribution - standalone) / portfolio_variance
                    if portfolio_variance
                    else 0.0
                ),
                "concentration_contribution": (
                    float(weight_series[symbol] ** 2) / concentration_total
                    if concentration_total
                    else 0.0
                ),
                "marginal_risk": (
                    float(marginal[symbol]) / portfolio_volatility
                    if portfolio_volatility
                    else 0.0
                ),
            }
        )
    return rows


def _execution_attribution(
    deviations: list[dict[str, object]],
    executions: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, float | int | str]] = {}
    for row in deviations:
        reason = str(row["reason"])
        bucket = grouped.setdefault(
            reason,
            {
                "reason": reason,
                "symbol_count": 0,
                "signed_weight_gap": 0.0,
                "absolute_weight_gap": 0.0,
                "execution_events": 0,
            },
        )
        bucket["symbol_count"] = int(bucket["symbol_count"]) + 1
        gap = float(row["weight_gap"])
        bucket["signed_weight_gap"] = float(bucket["signed_weight_gap"]) + gap
        bucket["absolute_weight_gap"] = float(bucket["absolute_weight_gap"]) + abs(gap)
    for execution in executions:
        reason = str(
            execution.get("reason_code")
            or execution.get("outcome")
            or "FILLED"
        )
        bucket = grouped.setdefault(
            reason,
            {
                "reason": reason,
                "symbol_count": 0,
                "signed_weight_gap": 0.0,
                "absolute_weight_gap": 0.0,
                "execution_events": 0,
            },
        )
        bucket["execution_events"] = int(bucket["execution_events"]) + 1
    return [dict(grouped[key]) for key in sorted(grouped)]


def _weight_deviations(
    portfolio: dict[str, object],
    marks: dict[str, float],
    targets: dict[str, float],
    sectors: dict[str, str],
    orders: list[dict[str, object]],
    lot_size: int,
) -> list[dict[str, object]]:
    equity = float(portfolio["equity"])
    positions = {
        str(row["symbol"]): int(row["quantity"])
        for row in portfolio["positions"]  # type: ignore[index]
    }
    order_map = {str(row["symbol"]): row for row in orders}
    rows = []
    for symbol in sorted(set(targets) | set(positions)):
        price = float(marks.get(symbol, 0))
        actual = positions.get(symbol, 0) * price / equity if equity else 0.0
        target = float(targets.get(symbol, 0))
        order = order_map.get(symbol, {})
        reason = order.get("rejection_code")
        if reason == "NO_REBALANCE_REQUIRED" and target > 0 and actual == 0:
            reason = "BELOW_ONE_LOT"
        if reason is None and abs(target - actual) > 1e-6:
            reason = "PENDING_T_PLUS_ONE"
        rows.append(
            {
                "symbol": symbol,
                "sector": sectors.get(symbol, "exit"),
                "target_weight": target,
                "actual_weight": actual,
                "weight_gap": actual - target,
                "minimum_lot_value": price * lot_size,
                "reason": reason or "MATCHED",
            }
        )
    return rows


def _daily_metrics(
    store: ParquetStore,
    trade_date: str,
    portfolio: dict[str, object],
    targets: dict[str, float],
    reconciliation: dict[str, object],
) -> dict[str, object]:
    equity = float(portfolio["equity"])
    previous_equity = None
    peak = equity
    try:
        history = store.read("multi_sector_account_daily").sort_values("trade_date")
        prior = history.loc[history["trade_date"].astype(str) < trade_date]
        if not prior.empty:
            previous_equity = float(prior.iloc[-1]["equity"])
            peak = max(float(prior["equity"].max()), equity)
    except FileNotFoundError:
        pass
    return {
        "equity": equity,
        "cash": float(portfolio["cash"]),
        "market_value": float(portfolio["market_value"]),
        "gross_exposure": (
            float(portfolio["market_value"]) / equity if equity else 0.0
        ),
        "target_exposure": sum(targets.values()),
        "daily_return": (
            equity / previous_equity - 1 if previous_equity else 0.0
        ),
        "drawdown": equity / peak - 1 if peak else 0.0,
        "reconciled": bool(reconciliation["matched"]),
    }


def _risk_alerts(
    config: BacktestConfig,
    metrics: dict[str, object],
    deviations: list[dict[str, object]],
    stale_symbols: list[str],
    reconciliation: dict[str, object],
    quality_checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    alerts = []
    if stale_symbols:
        alerts.append(
            {
                "code": "DATA_STALE",
                "severity": "BLOCK",
                "message": f"{len(stale_symbols)}只目标股票缺少当日行情，禁止生成新订单",
            }
        )
    blocked_checks = [row for row in quality_checks if row["status"] == "BLOCK"]
    if blocked_checks:
        alerts.append(
            {
                "code": "DATA_QUALITY_BLOCKED",
                "severity": "BLOCK",
                "message": f"{len(blocked_checks)}项数据质量检查未通过",
            }
        )
    if not reconciliation["matched"]:
        alerts.append(
            {"code": "RECONCILIATION_FAILED", "severity": "BLOCK", "message": "账户对账失败"}
        )
    if float(metrics["gross_exposure"]) > config.max_gross_exposure + 1e-9:
        alerts.append(
            {"code": "GROSS_EXPOSURE_LIMIT", "severity": "BLOCK", "message": "总股票仓位超过上限"}
        )
    if float(metrics["drawdown"]) <= -config.max_drawdown:
        alerts.append(
            {"code": "DRAWDOWN_LIMIT", "severity": "BLOCK", "message": "账户回撤达到熔断线"}
        )
    constrained = sum(row["reason"] == "BELOW_ONE_LOT" for row in deviations)
    if constrained:
        alerts.append(
            {
                "code": "LOT_SIZE_CONSTRAINT",
                "severity": "INFO",
                "message": f"{constrained}只股票目标金额不足一手",
            }
        )
    return alerts


def _persist_daily_analytics(
    store: ParquetStore,
    trade_date: str,
    status: str,
    metrics: dict[str, object],
    attribution: list[dict[str, object]],
    deviations: list[dict[str, object]],
    alerts: list[dict[str, object]],
    return_attribution: list[dict[str, object]],
    risk_attribution: list[dict[str, object]],
    execution_attribution: list[dict[str, object]],
    attribution_reconciliation: dict[str, object],
) -> None:
    store.merge_curated(
        "multi_sector_account_daily",
        [pd.DataFrame([{"trade_date": trade_date, "status": status, **metrics}])],
        ["trade_date"],
    )
    if attribution:
        store.merge_curated(
            "multi_sector_attribution_daily",
            [pd.DataFrame(attribution).assign(trade_date=trade_date)],
            ["trade_date", "sector"],
        )
    if deviations:
        store.merge_curated(
            "multi_sector_deviation_daily",
            [pd.DataFrame(deviations).assign(trade_date=trade_date)],
            ["trade_date", "symbol"],
        )
    if alerts:
        store.merge_curated(
            "multi_sector_risk_alerts",
            [pd.DataFrame(alerts).assign(trade_date=trade_date)],
            ["trade_date", "code"],
        )
    if return_attribution:
        store.merge_curated(
            "multi_sector_return_attribution_daily",
            [pd.DataFrame(return_attribution).assign(trade_date=trade_date)],
            ["trade_date", "component", "sector", "detail"],
        )
    if risk_attribution:
        store.merge_curated(
            "multi_sector_risk_attribution_daily",
            [pd.DataFrame(risk_attribution).assign(trade_date=trade_date)],
            ["trade_date", "symbol"],
        )
    if execution_attribution:
        store.merge_curated(
            "multi_sector_execution_attribution_daily",
            [pd.DataFrame(execution_attribution).assign(trade_date=trade_date)],
            ["trade_date", "reason"],
        )
    store.merge_curated(
        "multi_sector_attribution_reconciliation_daily",
        [
            pd.DataFrame(
                [{"trade_date": trade_date, **attribution_reconciliation}]
            )
        ],
        ["trade_date"],
    )
