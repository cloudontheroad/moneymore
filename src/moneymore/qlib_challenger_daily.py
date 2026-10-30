from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from .config import BacktestConfig
from .data.store import ParquetStore
from .execution.paper import ExecutionBar, PaperBroker
from .execution.risk import PortfolioSnapshot, create_order_intent
from .portfolio_constructor import (
    adjusted_close_panel,
    global_topk_portfolio,
    trailing_return_correlation,
)
from .qlib_challenger import (
    QlibPanelDataset,
    build_challenger_dataset,
    challenger_universe,
    rank_blend_predictions,
    research_gate_diagnostics,
)
from .qlib_exposure import dynamic_target_exposure, previous_report_exposure
from .qlib_governance import bootstrap_qlib_release
from .signals import SignalDecision, write_signal_artifact
from .strategy_universe import execution_strategy_id

QLIB_CHALLENGER_ACCOUNT = "qlib_gru_shadow"
QLIB_CHALLENGER_STRATEGY = "qlib_gru_alpha360_v1"
_LIVE_CORRELATION_CACHE: dict[tuple[object, ...], pd.DataFrame] = {}


@dataclass(frozen=True)
class ChallengerDailyResult:
    trade_date: str
    status: str
    model_id: str
    selected: list[str]
    scores: list[dict[str, object]]
    executions: list[dict[str, object]]
    orders: list[dict[str, object]]
    portfolio: dict[str, object]
    reconciliation: dict[str, object]
    report_path: str


def _prepare_model_for_inference(model: object) -> None:
    network = getattr(model, "gru_model", None)
    if network is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network.to(device)
        model.device = device
    recurrent = getattr(network, "rnn", None)
    flatten = getattr(recurrent, "flatten_parameters", None)
    if callable(flatten):
        flatten()


def _live_correlation(
    store: ParquetStore,
    universe: dict[str, str],
    cutoff: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    key = (str(store.root.resolve()), str(cutoff.date()), tuple(sorted(universe)), lookback)
    cached = _LIVE_CORRELATION_CACHE.get(key)
    if cached is not None:
        return cached
    result = trailing_return_correlation(
        adjusted_close_panel(store, list(universe)), cutoff, lookback
    )
    _LIVE_CORRELATION_CACHE.clear()
    _LIVE_CORRELATION_CACHE[key] = result
    return result


def run_qlib_challenger_daily(
    *,
    root: Path,
    store: ParquetStore,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
    signal_dir: Path,
    report_dir: Path,
    account_id: str = QLIB_CHALLENGER_ACCOUNT,
    strategy_id: str | None = None,
    model_dir: Path | None = None,
    research_path: Path | None = None,
    account_history_table: str = "qlib_challenger_account_daily",
    portfolio_handler: Callable | None = None,
) -> ChallengerDailyResult:
    default_challenger_config = yaml.safe_load(
        (root / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
    )
    model_id = str(default_challenger_config["model_id"])
    strategy_id = execution_strategy_id(strategy_id or model_id, store)
    model_dir = model_dir or root / "state" / "qlib-challenger" / "models"
    model_path = model_dir / f"{model_id}.pkl"
    ensemble_path = model_dir / f"{model_id}_ensemble.json"
    research_path = research_path or (
        root / "state" / "qlib-challenger" / "latest-research.json"
    )
    broker.initialize_account(config.initial_cash, account_id)
    if not (model_path.exists() or ensemble_path.exists()) or not research_path.exists():
        return _finish(
            trade_date,
            "AWAITING_TRAINING",
            model_id,
            [],
            [],
            [],
            [],
            broker.account_snapshot({}, account_id),
            broker,
            report_dir,
            account_id=account_id,
        )
    research = json.loads(research_path.read_text(encoding="utf-8"))
    challenger_config = research.get("protocol") or default_challenger_config
    model_id = str(challenger_config["model_id"])
    deployment = bootstrap_qlib_release(root)
    gru_metrics = next(
        (row for row in research["metrics"] if row["model_id"] == model_id),
        None,
    )
    research_gate_passed, research_gate_failures = research_gate_diagnostics(
        research, default_challenger_config
    )
    execution_policy = challenger_config.get("execution_policy", {})
    experimental_paper_enabled = bool(
        execution_policy.get("experimental_paper_enabled", True)
    )
    if account_id == QLIB_CHALLENGER_ACCOUNT and deployment.get("lifecycle") == "RETIRED":
        experimental_paper_enabled = False
    universe = challenger_universe(root, store)
    frame = build_challenger_dataset(
        store,
        universe,
        sequence_length=int(challenger_config["sequence_length"]),
        label_horizon=int(challenger_config["label_horizon"]),
        require_label=False,
        feature_count=int(challenger_config["model"]["d_feat"]),
        live_as_of=trade_date,
    )
    cutoff = pd.Timestamp(trade_date)
    frame = frame.loc[
        frame.index.get_level_values("datetime") <= cutoff
    ]
    if frame.empty:
        raise ValueError("NO_CONTIGUOUS_QLIB_INPUT: missing or incomplete exchange-session features")
    latest_date = frame.index.get_level_values("datetime").max()
    if latest_date != cutoff:
        raise ValueError(f"STALE_QLIB_INPUT: requested {trade_date}, latest valid features {latest_date}")
    live_dataset = QlibPanelDataset(
        frame,
        {"live": (str(latest_date.date()), str(latest_date.date()))},
    )
    if ensemble_path.exists():
        manifest = json.loads(ensemble_path.read_text(encoding="utf-8"))
        ensemble_predictions = []
        for item in manifest["models"]:
            filename = item["filename"] if isinstance(item, dict) else item
            weight = float(item.get("weight", 1.0)) if isinstance(item, dict) else 1.0
            with (ensemble_path.parent / filename).open("rb") as handle:
                model = pickle.load(handle)
            _prepare_model_for_inference(model)
            ensemble_predictions.append((model.predict(live_dataset, "live"), weight))
        predictions = (
            rank_blend_predictions(ensemble_predictions)
            if manifest.get("method") == "daily_rank_blend"
            else sum(series for series, _ in ensemble_predictions) / len(ensemble_predictions)
        )
    else:
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        _prepare_model_for_inference(model)
        predictions = model.predict(live_dataset, "live")
    score_frame = predictions.rename("score").reset_index()
    score_frame["datetime"] = pd.to_datetime(score_frame["datetime"]).dt.strftime(
        "%Y%m%d"
    )
    score_frame["sector"] = score_frame["instrument"].map(universe)
    portfolio_policy = challenger_config["portfolio_policy"]
    correlation = _live_correlation(
        store,
        universe,
        cutoff,
        int(portfolio_policy["correlation_lookback"]),
    )
    selected, portfolio_weights, selection_metadata = _scheduled_portfolio(
        score_frame,
        strategy_id,
        trade_date,
        report_dir,
        portfolio_policy,
        int(challenger_config["rebalance_interval"]),
        correlation,
    )
    account_before = broker.account_snapshot({}, account_id)
    held = {
        str(row["symbol"])
        for row in account_before["positions"]  # type: ignore[index]
    }
    marks, bars = _latest_market_state(
        store,
        set(universe) | held,
        cutoff,
        trade_date,
    )
    if not experimental_paper_enabled:
        # Explicitly disabled experiments still settle prior T+1 orders, but
        # create no new ones.  A failed research gate is deliberately *not*
        # such a condition: it blocks promotion, not forward paper evidence.
        executions = []
        for symbol, bar in sorted(bars.items()):
            executions.extend(
                broker.execute_pending(bar, config, account_id)
            )
        portfolio = broker.account_snapshot(marks, account_id)
        _record_account_history(
            store, trade_date, portfolio, account_id, account_history_table
        )
        return _finish(
            trade_date,
            "OBSERVATION_ONLY",
            model_id,
            selected,
            score_frame.sort_values("score", ascending=False).to_dict("records"),
            executions,
            [],
            portfolio,
            broker,
            report_dir,
            {
                **selection_metadata,
                "execution_strategy_id": strategy_id,
                "research_gate_passed": research_gate_passed,
                "research_gate_failures": research_gate_failures,
                "deployment_mode": deployment.get("execution_mode"),
                "observation_reason": "EXPERIMENTAL_PAPER_DISABLED",
            },
            account_id=account_id,
        )

    failed_gate_mode = str(execution_policy.get("failed_gate_mode", "observe_with_paper"))
    if not research_gate_passed and failed_gate_mode == "scores_only":
        executions = []
        for symbol, bar in sorted(bars.items()):
            executions.extend(broker.execute_pending(bar, config, account_id))
        broker.cancel_pending(account_id, "RESEARCH_GATE_FAILED_SCORES_ONLY", side="BUY")
        portfolio = broker.account_snapshot(marks, account_id)
        _record_account_history(store, trade_date, portfolio, account_id, account_history_table)
        return _finish(
            trade_date, "RESEARCH_GATE_FAILED_SCORES_ONLY", model_id, selected,
            score_frame.sort_values("score", ascending=False).to_dict("records"),
            executions, [], portfolio, broker, report_dir,
            {
                **selection_metadata,
                "execution_strategy_id": strategy_id,
                "research_gate_passed": False,
                "research_gate_failures": research_gate_failures,
                "deployment_mode": deployment.get("execution_mode"),
                "promotion_eligible": False,
                "protocol_version": challenger_config.get("protocol_version", "legacy"),
                "observation_reason": "RESEARCH_GATE_FAILED_SCORES_ONLY",
            },
            account_id=account_id,
        )

    if portfolio_handler is not None:
        return portfolio_handler(score_frame=score_frame, bootstrap_weights=portfolio_weights,
                                 marks=marks, strategy_id=strategy_id, model_id=model_id)

    executions = []
    for symbol, bar in sorted(bars.items()):
        executions.extend(
            broker.execute_pending(bar, config, account_id)
        )
    account = broker.account_snapshot(marks, account_id)
    positions = {
        str(row["symbol"]): row for row in account["positions"]  # type: ignore[index]
    }
    exposure = dynamic_target_exposure(
        store,
        selected,
        cutoff,
        challenger_config["exposure_policy"],
        previous_exposure=previous_report_exposure(report_dir, trade_date),
    )
    target_gross_exposure = float(exposure["target_gross_exposure"])
    portfolio_weights = {
        symbol: weight * target_gross_exposure
        for symbol, weight in portfolio_weights.items()
    }
    orders = []
    for symbol in sorted(set(selected) | set(positions)):
        close = marks.get(symbol)
        if close is None:
            continue
        target = float(portfolio_weights.get(symbol, 0.0))
        decision = SignalDecision(
            strategy_id=strategy_id,
            symbol=symbol,
            as_of_date=trade_date,
            target_weight=target,
            action="HOLD" if target else "EXIT",
            reason_code="QLIB_GRU_TOPK" if target else "QLIB_GRU_EXIT",
            close=close,
            fast_ma=None,
            slow_ma=None,
            history_bars=int(challenger_config["sequence_length"]),
        )
        risk = create_order_intent(
            decision,
            PortfolioSnapshot(
                cash=float(account["cash"]),
                equity=float(account["equity"]),
                position_quantity=int(positions.get(symbol, {}).get("quantity", 0)),
                reference_price=close,
            ),
            lot_size=config.lot_size,
            max_symbol_weight=config.max_position_weight,
            minimum_rebalance_notional=float(
                execution_policy.get("minimum_rebalance_notional", 0)
            ),
        )
        if risk.rejection_code == "BELOW_MINIMUM_REBALANCE_NOTIONAL":
            broker.cancel_pending_symbol(
                account_id,
                symbol,
                trade_date,
                "BELOW_MINIMUM_REBALANCE_NOTIONAL",
            )
        write_signal_artifact(decision, signal_dir)
        decision_status = broker.record_decision(decision)
        submit_status = (
            broker.submit(risk, account_id)
            if decision_status in {"RECORDED", "DUPLICATE"}
            else "DUPLICATE_DECISION"
        )
        orders.append(
            {
                "symbol": symbol,
                "target_weight": target,
                "decision_status": decision_status,
                "submit_status": submit_status,
                "rejection_code": risk.rejection_code,
            }
        )
    portfolio = broker.account_snapshot(marks, account_id)
    _record_account_history(
        store, trade_date, portfolio, account_id, account_history_table
    )
    reconciliation = asdict(broker.reconcile(account_id))
    status = (
        "EXPERIMENTAL_PAPER"
        if reconciliation["matched"]
        else "FAILED_RECONCILIATION"
    )
    return _finish(
        trade_date,
        status,
        model_id,
        selected,
        score_frame.sort_values("score", ascending=False).to_dict("records"),
        executions,
        orders,
        portfolio,
        broker,
        report_dir,
        {
            **selection_metadata,
            "execution_strategy_id": strategy_id,
            "input_calendar_enforced": True,
            "scored_universe_size": len(score_frame),
            "requested_universe_size": len(universe),
            "research_gate_passed": research_gate_passed,
            "research_gate_failures": research_gate_failures,
            "deployment_mode": deployment.get("execution_mode"),
            "promotion_eligible": research_gate_passed
            and deployment.get("execution_mode") == "PAPER_TRADING",
            "target_gross_exposure": target_gross_exposure,
            "target_weights": portfolio_weights,
            "portfolio_policy": portfolio_policy,
            "exposure_policy": exposure,
        },
        account_id=account_id,
    )


def _latest_market_state(
    store: ParquetStore,
    symbols: set[str],
    cutoff: pd.Timestamp,
    trade_date: str,
) -> tuple[dict[str, float], dict[str, ExecutionBar]]:
    marks: dict[str, float] = {}
    bars: dict[str, ExecutionBar] = {}
    requested = sorted(symbols)
    if not requested:
        return marks, bars
    market = store.read(
        "daily",
        columns=["ts_code", "trade_date", "open", "close"],
        filters=[("ts_code", "in", requested)],
    )
    market = market.loc[market["trade_date"].astype(str) <= trade_date]
    latest = (
        market.sort_values(["ts_code", "trade_date"])
        .groupby("ts_code", as_index=False)
        .tail(1)
    )
    try:
        limits = store.read(
            "stock_limits",
            columns=["ts_code", "trade_date", "up_limit", "down_limit"],
            filters=[("ts_code", "in", requested), ("trade_date", "=", trade_date)],
        ).set_index("ts_code")
    except FileNotFoundError:
        limits = pd.DataFrame()
    tolerance = 1e-8
    for row in latest.itertuples(index=False):
        symbol = str(row.ts_code)
        marks[symbol] = float(row.close)
        if str(row.trade_date) != trade_date:
            continue
        limit = limits.loc[symbol] if symbol in limits.index else None
        up_limit = float(limit["up_limit"]) if limit is not None else None
        down_limit = float(limit["down_limit"]) if limit is not None else None
        bars[symbol] = ExecutionBar(
            symbol=symbol,
            trade_date=trade_date,
            open=float(row.open),
            close=float(row.close),
            can_buy=up_limit is None or float(row.open) < up_limit - tolerance,
            can_sell=down_limit is None or float(row.open) > down_limit + tolerance,
        )
    return marks, bars


def _scheduled_portfolio(
    score_frame: pd.DataFrame,
    model_id: str,
    trade_date: str,
    report_dir: Path,
    policy: dict[str, object],
    interval: int,
    correlation: pd.DataFrame,
) -> tuple[list[str], dict[str, float], dict[str, object]]:
    prior_reports = []
    if report_dir.exists():
        for path in sorted(report_dir.glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if (
                report.get("execution_strategy_id", report.get("model_id")) == model_id
                and str(report.get("trade_date", "")) < trade_date
                and report.get("selected")
            ):
                prior_reports.append(report)
    previous = prior_reports[-1] if prior_reports else None
    days_since_rebalance = (
        int(previous.get("days_since_rebalance", 0)) + 1 if previous else interval
    )
    if previous and days_since_rebalance < interval:
        prior_weights = previous.get("target_weights")
        if isinstance(prior_weights, dict) and prior_weights:
            return sorted(previous["selected"]), {
                str(symbol): float(weight) for symbol, weight in prior_weights.items()
            }, {
                "rebalanced": False,
                "days_since_rebalance": days_since_rebalance,
                "days_until_rebalance": interval - days_since_rebalance,
            }
    incumbents = set(previous.get("selected", [])) if previous else set()
    selected, weights, ranking = global_topk_portfolio(
        score_frame,
        incumbents,
        symbol_column="instrument",
        top_k=int(policy["top_k"]),
        exit_rank=int(policy["exit_rank"]),
        max_replacements=int(policy["max_replacements"]),
        minimum_weight=float(policy["minimum_weight"]),
        maximum_weight=float(policy["maximum_weight"]),
        correlation=correlation,
        correlation_penalty=float(policy["correlation_penalty"]),
        cluster_correlation_threshold=float(policy["cluster_correlation_threshold"]),
        maximum_cluster_members=int(policy["maximum_cluster_members"]),
        weighting_method=str(policy.get("weighting_method", "rank_linear")),
        confidence_temperature=float(policy.get("confidence_temperature", 1.0)),
        confidence_shrinkage=float(policy.get("confidence_shrinkage", 0.35)),
    )
    return selected, weights, {
        "selection_method": "global_rank_weighted_topk",
        "selected_global_ranks": {
            str(row["instrument"]): int(row["global_rank"])
            for row in ranking.loc[ranking["selected"]].to_dict("records")
        },
        "rebalanced": True,
        "days_since_rebalance": 0,
        "days_until_rebalance": interval,
    }


def _record_account_history(
    store: ParquetStore,
    trade_date: str,
    portfolio: dict[str, object],
    account_id: str = QLIB_CHALLENGER_ACCOUNT,
    table: str = "qlib_challenger_account_daily",
) -> None:
    row = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "account_id": account_id,
                "cash": float(portfolio["cash"]),
                "market_value": float(portfolio["market_value"]),
                "equity": float(portfolio["equity"]),
            }
        ]
    )
    store.merge_curated(
        table,
        [row],
        ["trade_date", "account_id"],
    )


def _finish(
    trade_date: str,
    status: str,
    model_id: str,
    selected: list[str],
    scores: list[dict[str, object]],
    executions: list[dict[str, object]],
    orders: list[dict[str, object]],
    portfolio: dict[str, object],
    broker: PaperBroker,
    report_dir: Path,
    selection_metadata: dict[str, object] | None = None,
    *,
    account_id: str = QLIB_CHALLENGER_ACCOUNT,
) -> ChallengerDailyResult:
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / f"{trade_date}.json"
    payload = {
        "trade_date": trade_date,
        "status": status,
        "model_id": model_id,
        "selected": selected,
        "scores": scores,
        "executions": executions,
        "orders": orders,
        "portfolio": portfolio,
        "account_id": account_id,
        "reconciliation": asdict(broker.reconcile(account_id)),
        "created_at": datetime.now(UTC).isoformat(),
        "report_path": str(target),
        **(selection_metadata or {}),
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("status") != status:
            target = report_dir / f"{trade_date}_{status}.json"
            payload["report_path"] = str(target)
            content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
    return ChallengerDailyResult(
        **{key: payload[key] for key in ChallengerDailyResult.__dataclass_fields__}
    )
