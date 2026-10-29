"""Forward-only fixed-risk portfolio candidates. PaperBroker only, no real broker."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .candidate_lifecycle import trading_allowed
from .challenger_diagnostics import trading_dates, write_diagnostic
from .execution.paper import ExecutionBar
from .execution.risk import OrderIntent, RiskResult
from .models import Side
from .portfolio_constructor import global_topk_portfolio
from .research.enhanced_indexing import (
    EnhancedPolicy,
    carry_confirmed_suspensions,
    optimize_snapshot,
)
from .research.integer_execution import plan_integer_execution


def _paths(root, tag):
    return (
        root / "state/qlib-portfolio-plans" / tag,
        root / "state/qlib-portfolio-budgets" / (tag + ".json"),
        root / "state/qlib-portfolio-execution" / tag,
    )


def derived_execution_complete(root, candidates, date):
    """Legacy completion must not hide a newly activated independent account."""
    for candidate in candidates:
        if (
            not candidate.get("portfolio_variant")
            or candidate.get("archived")
            or candidate.get("validation_hold")
            or str(candidate.get("activation_date", candidate["candidate_tag"])) > date
        ):
            continue
        path = (
            root / "state/qlib-portfolio-execution" / candidate["candidate_tag"] / (date + ".json")
        )
        if (
            not path.exists()
            or json.loads(path.read_text(encoding="utf-8")).get("status") != "COMPLETED"
        ):
            return False
    return True


def risk_snapshot(store, scores, date):
    calendar = [d for d in trading_dates(store, date) if d <= date][-121:]
    if len(calendar) != 121 or calendar[-1] != date:
        raise ValueError("INCOMPLETE_RISK_CALENDAR")
    filters = [("trade_date", ">=", calendar[0]), ("trade_date", "<=", date)]
    daily = store.read("daily", filters=filters)
    adj = store.read("adj_factor", filters=filters)
    joined = daily.merge(adj, on=["trade_date", "ts_code"], validate="one_to_one")
    joined["adjusted"] = joined.close * joined.adj_factor
    prices = joined.pivot(index="trade_date", columns="ts_code", values="adjusted").reindex(
        calendar
    )
    try:
        prices, _ = carry_confirmed_suspensions(
            prices, store.read("enhanced_indexing_suspension_proof")
        )
    except FileNotFoundError:
        pass
    returns = prices.pct_change(fill_method=None).iloc[1:]
    basic = store.read(
        "daily_basic", filters=[("trade_date", ">=", calendar[-10]), ("trade_date", "<=", date)]
    )
    size = basic.sort_values("trade_date").groupby("ts_code").tail(1).set_index("ts_code").total_mv
    eligible = [
        s
        for s in scores.index
        if s in returns and np.isfinite(returns[s]).all() and size.get(s, 0) > 0
    ]
    risk = returns[eligible]
    proxy = risk.mean(axis=1)
    styles = pd.DataFrame(
        {
            "momentum_60": (1 + risk.tail(60)).prod() - 1,
            "volatility_20": risk.tail(20).std(),
            "market_beta_120": risk.apply(lambda x: x.cov(proxy) / proxy.var()),
            "log_size": np.log(size.reindex(eligible)),
        }
    )
    if len(styles) < 10 or not np.isfinite(styles.to_numpy()).all():
        raise ValueError("INCOMPLETE_STYLE_RISK_INPUT")
    return risk, size, styles


def prepare_portfolio_candidate(
    *,
    root,
    store,
    broker,
    config,
    candidate,
    trade_date,
    score_frame,
    bootstrap_weights,
    marks,
    strategy_id,
    model_id,
):
    from .qlib_challenger_daily import _finish, _record_account_history

    tag, account_id = str(candidate["candidate_tag"]), str(candidate["account_id"])
    plan_dir, budget_path, _ = _paths(root, tag)
    snapshot = broker.account_snapshot(marks, account_id)
    held = {
        str(p["symbol"]): int(p["quantity"]) for p in snapshot["positions"] if p["quantity"] > 0
    }
    if set(held) - set(marks):
        raise ValueError("HELD_MARK_MISSING")
    scores = score_frame.set_index("instrument").score
    risk, size, styles = risk_snapshot(store, scores, trade_date)
    if set(held) - set(styles.index):
        raise ValueError("HELD_RISK_MISSING")
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else None
    if held:
        if budget is None:
            raise ValueError("EXISTING_HOLDINGS_WITHOUT_FIXED_BUDGET")
        current = pd.Series({s: q * marks[s] for s, q in held.items()})
        current /= current.sum()
        solved = optimize_snapshot(
            scores,
            risk,
            size,
            current,
            EnhancedPolicy(lamb=10.0, objective_preconditioning=True),
            style_controls=styles,
            style_limits=budget["limits"],
            solver_max_iters=1000,
        )
        if solved["status"] != "PASS_OFFLINE_ONLY":
            raise ValueError("FIXED_BUDGET_OPTIMIZER_BLOCKED:" + ",".join(solved["failures"]))
        targets = solved["weights"]
    else:
        # A new cash account must not inherit the source account's stale
        # five-day holdings when those names no longer have current scores.
        research = json.loads(Path(candidate["research_path"]).read_text(encoding="utf-8"))
        policy = research["protocol"]["portfolio_policy"]
        fresh = score_frame.loc[score_frame.instrument.isin(styles.index)]
        _, targets, _ = global_topk_portfolio(
            fresh,
            set(),
            symbol_column="instrument",
            top_k=int(policy["top_k"]),
            exit_rank=int(policy["exit_rank"]),
            max_replacements=int(policy["max_replacements"]),
            minimum_weight=float(policy["minimum_weight"]),
            maximum_weight=float(policy["maximum_weight"]),
            correlation=risk.corr(),
            correlation_penalty=float(policy["correlation_penalty"]),
            cluster_correlation_threshold=float(policy["cluster_correlation_threshold"]),
            maximum_cluster_members=int(policy["maximum_cluster_members"]),
            weighting_method=str(policy.get("weighting_method", "rank_linear")),
            confidence_temperature=float(policy.get("confidence_temperature", 1.0)),
            confidence_shrinkage=float(policy.get("confidence_shrinkage", 0.35)),
        )
        solved = {
            "status": "FRESH_RISK_ELIGIBLE_SOURCE_MODEL_BOOTSTRAP",
            "risk_eligible_assets": len(styles),
            "excluded_score_assets": sorted(set(scores.index) - set(styles.index)),
            "ignored_stale_source_targets": sorted(set(bootstrap_weights) - set(styles.index)),
        }
    if {s for s, w in targets.items() if w > 1e-6} - set(styles.index):
        raise ValueError("BOOTSTRAP_TARGET_RISK_MISSING")
    if abs(sum(targets.values()) - 1) > 1e-6:
        raise ValueError("FULL_STOCK_TARGET_REQUIRED")
    standardized = (styles - styles.mean()) / styles.std(ddof=0).replace(0, 1)
    limits = budget["limits"] if budget else standardized.abs().max().to_dict()
    plan = {
        "candidate_tag": tag,
        "account_id": account_id,
        "signal_date": trade_date,
        "created_at": datetime.now(UTC).isoformat(),
        "strategy_id": strategy_id,
        "model_id": model_id,
        "target_weights": targets,
        "styles": styles.to_dict("index"),
        "style_limits": limits,
        "bootstrap": budget is None,
        "source_candidate_tag": candidate["source_candidate_tag"],
        "real_orders_enabled": False,
        "promotion_eligible": False,
    }
    write_diagnostic(plan_dir / (trade_date + ".json"), plan)
    preview = plan_integer_execution(
        targets, marks, held, held, snapshot["cash"], snapshot["equity"], config, styles, limits
    )
    _record_account_history(store, trade_date, snapshot, account_id, "qlib_candidate_account_daily")
    calendar = trading_dates(store, trade_date)
    following = (
        calendar[calendar.index(trade_date) + 1]
        if calendar.index(trade_date) + 1 < len(calendar)
        else None
    )
    return _finish(
        trade_date,
        "EXPERIMENTAL_PAPER",
        model_id,
        [s for s, w in targets.items() if w > 1e-6],
        score_frame.to_dict("records"),
        [],
        [{**o, "submit_status": "AWAITING_OPEN_REPLAN"} for o in preview.get("orders", [])],
        snapshot,
        broker,
        root / "state/qlib-candidate-shadow" / tag,
        {
            "portfolio_variant": "fixed_budget_v1",
            "display_name": candidate["display_name"],
            "source_candidate_tag": candidate["source_candidate_tag"],
            "execution_strategy_id": strategy_id,
            "promotion_eligible": False,
            "target_weights": targets,
            "target_gross_exposure": 1.0,
            "optimizer": solved,
            "planning_status": "AWAITING_OPEN_REPLAN",
            "planned_trade_date": following,
            "reference_plan_status": preview["status"],
            "reference_plan_reason": preview.get("reason"),
            "fixed_budget": budget,
            "execution_note": "模拟账户；开盘有效实时报价重算后执行，不使用今晚参考股数直接成交。",
        },
        account_id=account_id,
    )


def execute_portfolio_candidate_open(
    *, root, store, broker, config, candidate, provider, trade_date, now=None
):
    from .qlib_challenger_daily import _record_account_history
    from .research.style_v2_experiment import candidate_execution_config, execution_policy_hash

    config = candidate_execution_config(config, candidate)

    tag, account = str(candidate["candidate_tag"]), str(candidate["account_id"])
    if candidate.get("validation_hold"):
        return {"status": "WEIGHT_VALIDATION_HOLD", "account_id": account}
    plan_dir, budget_path, receipt_dir = _paths(root, tag)
    receipt_path = receipt_dir / (trade_date + ".json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] in {"COMPLETED", "PREPARED", "EXECUTION_MISMATCH"}:
            return receipt  # Never repeat an ambiguous or completed submission.
    local = (now or datetime.now(ZoneInfo("Asia/Shanghai"))).astimezone(ZoneInfo("Asia/Shanghai"))
    if (
        local.strftime("%Y%m%d") != trade_date
        or local.weekday() >= 5
        or not (time(9, 30) <= local.time() <= time(11, 30) or time(13) <= local.time() < time(15))
    ):
        return {"status": "OUTSIDE_EXECUTION_SESSION", "account_id": account}
    if not trading_allowed(root, tag, trade_date):
        return {"status": "LIFECYCLE_FROZEN", "account_id": account}
    files = [p for p in plan_dir.glob("*.json") if p.stem < trade_date]
    if not files:
        return {"status": "AWAITING_PRIOR_CLOSE_PLAN", "account_id": account}
    path = max(files, key=lambda p: p.stem)
    calendar = trading_dates(store, trade_date)
    if trade_date not in calendar or calendar[calendar.index(trade_date) - 1] != path.stem:
        return {"status": "STALE_SIGNAL_PLAN", "account_id": account}
    plan = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("execution_policy") and (
        plan.get("execution_policy_sha256") != execution_policy_hash(candidate)
        or plan.get("execution_slippage_bps") != config.slippage_bps
    ):
        return {"status": "EXECUTION_POLICY_PLAN_MISMATCH", "account_id": account}
    if candidate.get("factor_policy", {}).get("experimental_linear_model"):
        from .research.style_v2_experiment import validate_experiment

        validate_experiment(root, candidate["factor_policy"])
        expected_policy = hashlib.sha256(json.dumps(candidate["factor_policy"], sort_keys=True).encode()).hexdigest()
        if plan.get("policy_sha256") != expected_policy or plan.get("real_orders_enabled") is not False:
            return {"status": "EXPERIMENT_POLICY_OR_PAPER_SCOPE_CHANGED", "account_id": account}
    if candidate.get("portfolio_variant") == "factor_baseline_v2":
        created = pd.to_datetime(plan.get("created_at"), utc=True, errors="coerce")
        if pd.isna(created) or created > pd.Timestamp(local):
            return {"status": "PLAN_PROVENANCE_BLOCKED", "account_id": account}
    broker.settle_corporate_actions(account, trade_date)
    broker.settle_available_positions(account, trade_date)
    positions = broker.account_snapshot({}, account)["positions"]
    held = {str(p["symbol"]): int(p["quantity"]) for p in positions if p["quantity"] > 0}
    available = {str(p["symbol"]): int(p["available_quantity"]) for p in positions}
    symbols = sorted({s for s, w in plan["target_weights"].items() if w > 1e-6} | set(held))
    quotes = provider.intraday_quotes(symbols)
    rows = (
        {}
        if quotes is None or quotes.empty
        else {str(r["symbol"]): r for r in quotes.to_dict("records")}
    )
    valid = {
        s: r
        for s, r in rows.items()
        if r.get("execution_ready") is True
        and np.isfinite(float(r.get("last", 0)))
        and float(r.get("last", 0)) > 0
        and pd.to_datetime(r.get("timestamp"), unit="ms", utc=True)
        .tz_convert("Asia/Shanghai")
        .strftime("%Y%m%d")
        == trade_date
    }
    if set(symbols) - set(valid):
        receipt = {
            "status": "QUOTE_QUALITY_BLOCKED",
            "account_id": account,
            "trade_date": trade_date,
            "missing_symbols": sorted(set(symbols) - set(valid)),
        }
        write_diagnostic(receipt_path, receipt)
        return receipt
    marks = {s: float(valid[s]["last"]) for s in symbols}
    snapshot = broker.account_snapshot(marks, account)
    if candidate.get("portfolio_variant") == "factor_baseline_v2":
        from .execution.risk_state import evaluate_account_risk_state

        try:
            history = store.read("qlib_candidate_account_daily")
            history = history.loc[
                history.account_id.eq(account) & history.trade_date.astype(str).le(trade_date)
            ]
            peak = max(
                float(history.equity.max()) if not history.empty else config.initial_cash,
                snapshot["equity"],
            )
        except FileNotFoundError:
            peak = max(config.initial_cash, snapshot["equity"])
        risk = evaluate_account_risk_state(
            config,
            snapshot["equity"],
            snapshot["cash"],
            snapshot["market_value"] / snapshot["equity"],
            snapshot["equity"] / peak - 1,
            "PASS",
            broker.reconcile(account).matched,
        )
        risk_state = broker.update_risk_state(
            account, risk.proposed_state, risk.reason_code, trade_date
        )
        if risk_state["effective_state"] != "NORMAL":
            receipt = {
                "status": "ACCOUNT_RISK_BLOCKED",
                "account_id": account,
                "trade_date": trade_date,
                "risk_state": risk_state,
                "real_orders_enabled": False,
            }
            write_diagnostic(receipt_path, receipt)
            return receipt
    styles = pd.DataFrame.from_dict(plan["styles"], orient="index")
    can_buy = {s: float(valid[s].get("up_limit", 0)) > marks[s] for s in symbols}
    can_sell = {s: 0 < float(valid[s].get("down_limit", 0)) < marks[s] for s in symbols}
    if candidate.get("execution_policy"):
        from .costs import execution_price

        can_buy = {s: execution_price(marks[s], Side.BUY, config.slippage_bps) < float(valid[s].get("up_limit", 0)) for s in symbols}
        can_sell = {s: 0 < float(valid[s].get("down_limit", 0)) < execution_price(marks[s], Side.SELL, config.slippage_bps) for s in symbols}
    # Missing limits are a quality block, not permission to trade.
    executable = plan_integer_execution(
        plan["target_weights"],
        marks,
        held,
        available,
        snapshot["cash"],
        snapshot["equity"],
        config,
        styles,
        plan["style_limits"],
        can_buy=can_buy,
        can_sell=can_sell,
    )
    receipt = {
        "status": "PLAN_BLOCKED",
        "account_id": account,
        "trade_date": trade_date,
        "signal_date": plan["signal_date"],
        "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plan": executable,
        "price_source": getattr(provider, "name", "unknown") + "_VALID_CURRENT_QUOTE",
        "execution_slippage_bps": config.slippage_bps,
        "execution_policy_sha256": execution_policy_hash(candidate),
        "created_at": datetime.now(UTC).isoformat(),
        "real_orders_enabled": False,
        "quote_timestamps": {s: valid[s]["timestamp"] for s in symbols},
    }
    if executable["status"] != "PASS_REFERENCE_QUOTES_ONLY":
        write_diagnostic(receipt_path, receipt)
        return receipt
    if any(o["status"] == "PENDING" for o in broker.orders() if o["account_id"] == account):
        receipt["status"] = "UNRESOLVED_PENDING_ORDERS"
        write_diagnostic(receipt_path, receipt)
        return receipt
    receipt["status"] = "PREPARED"
    write_diagnostic(receipt_path, receipt)
    for order in executable["orders"]:
        intent = OrderIntent(
            idempotency_key=f"{account}:{trade_date}:{order['symbol']}:{order['side']}",
            strategy_id=plan["strategy_id"],
            symbol=order["symbol"],
            side=Side(order["side"]),
            quantity=int(order["quantity"]),
            signal_date=plan["signal_date"],
            reason_code=plan.get("reason_code", "QLIB_FIXED_BUDGET_FORWARD_PAPER"),
        )
        if broker.submit(RiskResult(True, intent, None), account) != "PENDING":
            receipt["status"] = "EXECUTION_MISMATCH"
            write_diagnostic(receipt_path, receipt)
            return receipt
    for order in executable["orders"]:
        s = order["symbol"]
        broker.execute_pending(
            ExecutionBar(s, trade_date, marks[s], marks[s], can_buy[s], can_sell[s]),
            config,
            account,
        )
    actual = broker.account_snapshot(marks, account)
    quantities = {p["symbol"]: int(p["quantity"]) for p in actual["positions"] if p["quantity"] > 0}
    expected = {s: q for s, q in executable["target_quantities"].items() if q > 0}
    matched = quantities == expected and abs(actual["cash"] - executable["estimated_cash"]) < 0.01
    receipt.update(
        status="COMPLETED" if matched else "EXECUTION_MISMATCH",
        portfolio=actual,
        reconciliation=asdict(broker.reconcile(account)),
    )
    if (
        matched
        and plan["bootstrap"]
        and plan.get("freeze_budget", True)
        and not budget_path.exists()
    ):
        write_diagnostic(
            budget_path,
            {
                "limits": {k: abs(v) for k, v in executable["active_style_exposures"].items()},
                "frozen_at": datetime.now(UTC).isoformat(),
                "trade_date": trade_date,
                "calibration": "INITIAL_KNOWN_EXPOSURE_NOT_TRAINED_OR_OPTIMAL",
                "plan_sha256": receipt["plan_sha256"],
            },
        )
    _record_account_history(store, trade_date, actual, account, "qlib_candidate_account_daily")
    write_diagnostic(receipt_path, receipt)
    return receipt
