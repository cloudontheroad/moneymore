"""Independent, forward-only factor V2 paper candidate; never a real broker."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from .challenger_diagnostics import trading_dates, write_diagnostic
from .corporate_actions import process_corporate_actions
from .portfolio_constructor import global_topk_portfolio
from .qlib_challenger_daily import _finish, _record_account_history
from .research.baseline_v2_model import archive_v2, build_v2_snapshot, refresh_official_basic
from .research.integer_execution import plan_integer_execution


def prepare_factor_v2(*, root, store, broker, config, candidate, trade_date, signal_date=None):
    """Close planning only; bootstrap may use yesterday's known close, marked honestly."""
    signal_date = signal_date or trade_date
    tag, account = candidate["candidate_tag"], candidate["account_id"]
    if trade_date < candidate["activation_date"] or signal_date > trade_date:
        raise ValueError("FACTOR_V2_NOT_ACTIVE_OR_FUTURE_SIGNAL")
    calendar = trading_dates(store, signal_date)
    prior = max(d for d in calendar if d < signal_date)
    refresh_official_basic(root, store, prior)
    # Freeze policy in registry at activation; later edits require a new version/account.
    policy = candidate["factor_policy"]
    model_policy, variants, capture, exclusions = build_v2_snapshot(
        root, store, signal_date, policy
    )
    evidence = archive_v2(root, signal_date, model_policy, variants, capture, exclusions)
    scores = variants["industry_size_comparable"]
    broker.initialize_account(config.initial_cash, account)
    held_snapshot = broker.account_snapshot({}, account)
    held = {
        p["symbol"]: int(p["quantity"]) for p in held_snapshot["positions"] if p["quantity"] > 0
    }
    actions = process_corporate_actions(store, broker, account, trade_date, set(held))
    held_snapshot = broker.account_snapshot({}, account)
    held = {
        p["symbol"]: int(p["quantity"]) for p in held_snapshot["positions"] if p["quantity"] > 0
    }
    # Data-loss is not an instruction to sell a held security.
    if set(held) - set(scores.symbol):
        raise ValueError("HELD_V2_INPUT_INCOMPLETE_HOLD_AND_ALERT")
    price_panel = capture["prices"]
    price_panel.index = pd.to_datetime(price_panel.index, format="%Y%m%d")
    p = policy["portfolio"]
    correlation = (
        price_panel[scores.symbol]
        .pct_change(fill_method=None)
        .tail(int(p["correlation_lookback"]))
        .corr()
    )
    selected, targets, ranking = global_topk_portfolio(
        scores,
        set(held),
        symbol_column="symbol",
        top_k=int(p["top_k"]),
        exit_rank=int(p["exit_rank"]),
        max_replacements=int(p["max_replacements"]),
        gross_exposure=float(p["gross_exposure"]),
        minimum_weight=float(p["minimum_weight"]),
        maximum_weight=float(p["maximum_weight"]),
        correlation=correlation,
        cluster_correlation_threshold=float(p["cluster_correlation_threshold"]),
        maximum_cluster_members=int(p["maximum_cluster_members"]),
        weighting_method=p["weighting_method"],
    )
    if len(selected) != int(p["top_k"]) or abs(sum(targets.values()) - 1) > 1e-6:
        raise ValueError("INCOMPLETE_V2_TARGET_SUPPORT")
    raw_close = store.read(
        "daily",
        filters=[
            ("trade_date", "=", signal_date),
            ("ts_code", "in", sorted(set(targets) | set(held))),
        ],
    )
    marks = dict(zip(raw_close.ts_code, raw_close.close, strict=True))
    if (set(targets) | set(held)) - set(marks):
        raise ValueError("MISSING_V2_CLOSE_MARK")
    snapshot = broker.account_snapshot(marks, account)
    styles = pd.DataFrame({"execution_only": 0.0}, index=scores.symbol.tolist())
    limits = {"execution_only": 0.0}
    strategy_id = policy["version"]
    plan = {
        "candidate_tag": tag,
        "account_id": account,
        "signal_date": signal_date,
        "created_at": datetime.now(UTC).isoformat(),
        "strategy_id": strategy_id,
        "model_id": strategy_id,
        "target_weights": targets,
        "styles": styles.to_dict("index"),
        "style_limits": limits,
        "bootstrap": not held,
        "freeze_budget": False,
        "real_orders_enabled": False,
        "promotion_eligible": False,
        "reason_code": "FACTOR_BASELINE_V2_FORWARD_PAPER",
        "score_evidence": evidence["path"],
        "policy_sha256": evidence["policy_sha256"],
        "style_policy": "FACTOR_COMPARABILITY_NOT_PORTFOLIO_STYLE_OR_BOARD_QUOTA",
    }
    plan_path = root / "state/qlib-portfolio-plans" / tag / (signal_date + ".json")
    write_diagnostic(plan_path, plan)
    preview = plan_integer_execution(
        targets, marks, held, held, snapshot["cash"], snapshot["equity"], config, styles, limits
    )
    following = min((d for d in calendar if d > signal_date), default=None)
    _record_account_history(store, trade_date, snapshot, account, "qlib_candidate_account_daily")
    score_frame = ranking.rename(columns={"symbol": "instrument"}).copy()
    metadata = {
        "portfolio_variant": "factor_baseline_v2",
        "strategy_family": "factor_baseline",
        "display_name": policy["display_name"],
        "execution_strategy_id": strategy_id,
        "promotion_eligible": False,
        "target_weights": targets,
        "target_gross_exposure": sum(targets.values()),
        "planned_trade_date": following,
        "signal_date": signal_date,
        "planning_status": "AWAITING_OPEN_REPLAN",
        "reference_plan_status": preview["status"],
        "score_evidence": evidence,
        "corporate_actions": actions,
        "exposure_policy": {
            "method": "FULL_STOCK_BUDGET_NO_SECTOR_QUOTA",
            "target_gross_exposure": 1.0,
        },
    }
    return _finish(
        trade_date,
        "EXPERIMENTAL_PAPER",
        strategy_id,
        selected,
        score_frame[["instrument", "score"]].to_dict("records"),
        [],
        [{**o, "submit_status": "AWAITING_OPEN_REPLAN"} for o in preview.get("orders", [])],
        snapshot,
        broker,
        root / "state/qlib-candidate-shadow" / tag,
        metadata,
        account_id=account,
    )


def register_factor_v2(root, store, broker, config, *, now=None):
    policy = yaml.safe_load((root / "configs/baseline_v2.yaml").read_text(encoding="utf-8"))
    local = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    date = local.strftime("%Y%m%d")
    if date < policy["activation_date"] or policy["real_orders_enabled"]:
        raise ValueError("INVALID_V2_ACTIVATION")
    tag = policy["candidate_tag"]
    path = root / "state/factor-baseline-candidates" / (tag + ".json")
    if (root / "state/qlib-challenger/candidates" / tag / "_TRAINING_COMPLETE").exists() or (
        root / "state/qlib-portfolio-candidates" / (tag + ".json")
    ).exists():
        raise ValueError("CANDIDATE_TAG_ALREADY_USED_BY_ANOTHER_STRATEGY")
    if path.exists():
        candidate = json.loads(path.read_text(encoding="utf-8"))
    else:
        candidate = {
            "candidate_tag": tag,
            "account_id": "qlib_candidate_" + tag,
            "display_name": policy["display_name"],
            "portfolio_variant": "factor_baseline_v2",
            "strategy_family": "factor_baseline",
            "activation_date": date,
            "created_at": datetime.now(UTC).isoformat(),
            "factor_policy": policy,
            "real_orders_enabled": False,
            "promotion_eligible": False,
            "research_gate_scope": "SCORE_CONTRACT_CHECKED_FORWARD_PAPER_ONLY_NOT_RETURN_VALIDATED",
            "model_retrained": False,
        }
    # Registration on today's calendar date cannot manufacture yesterday's fills or NAV.
    prior = max(d for d in trading_dates(store, date) if d < date)
    report_path = root / "state/qlib-candidate-shadow" / tag / (date + ".json")
    if not report_path.exists():
        result = prepare_factor_v2(
            root=root,
            store=store,
            broker=broker,
            config=config,
            candidate=candidate,
            trade_date=date,
            signal_date=prior,
        )
    else:
        result = json.loads(report_path.read_text(encoding="utf-8"))
    # Outside close predictions: don't poison first-created forward evaluation.
    bootstrap_path = root / "state/factor-baseline-bootstrap" / (tag + ".json")
    registered = json.loads(report_path.read_text(encoding="utf-8"))
    if not bootstrap_path.exists() and str(registered.get("signal_date", date)) < date:
        write_diagnostic(bootstrap_path, registered)
    write_diagnostic(path, candidate)
    return result
