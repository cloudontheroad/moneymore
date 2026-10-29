from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .candidate_lifecycle import FROZEN, lifecycle_rows, run_lifecycle_review, trading_allowed
from .config import BacktestConfig
from .data.store import ParquetStore
from .execution.paper import PaperBroker
from .qlib_challenger import research_gate_diagnostics
from .qlib_challenger_daily import ChallengerDailyResult, run_qlib_challenger_daily

CANDIDATE_HISTORY_TABLE = "qlib_candidate_account_daily"


def candidate_account_id(tag: str) -> str:
    return f"qlib_candidate_{tag}"


def candidate_catalog(root: Path) -> list[dict[str, Any]]:
    base = root / "state" / "qlib-challenger" / "candidates"
    config = yaml.safe_load(
        (root / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
    )
    model_id = str(config["model_id"])
    rows: list[dict[str, Any]] = []
    lifecycle = lifecycle_rows(root)
    for path in sorted(base.glob("*/research.json"), reverse=True):
        if not (path.parent / "_TRAINING_COMPLETE").exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tag = str(payload.get("candidate_tag") or path.parent.name)
        protocol = payload.get("protocol") or config
        artifact_model_id = str(protocol.get("model_id", model_id))
        metric = next(
            (row for row in payload.get("metrics", []) if row.get("model_id") == artifact_model_id),
            {},
        )
        stability = payload.get("stability", {})
        gate_passed, gate_failures = research_gate_diagnostics(payload, config)
        report_dir = root / "state" / "qlib-candidate-shadow" / tag
        reports = sorted(report_dir.glob("*.json")) if report_dir.exists() else []
        latest = (
            max(
                (json.loads(item.read_text(encoding="utf-8")) for item in reports),
                key=lambda row: str(row.get("trade_date", "")),
            )
            if reports
            else {}
        )
        observation_days = len(
            {
                str(json.loads(item.read_text(encoding="utf-8")).get("trade_date"))
                for item in reports
            }
        )
        rows.append(
            {
                "candidate_tag": tag,
                "account_id": candidate_account_id(tag),
                "created_at": payload.get("created_at"),
                "data_cutoff": payload.get("data_cutoff"),
                "test_start": payload.get("effective_segments", {}).get("test", [None])[0],
                "test_end": payload.get("effective_segments", {}).get("test", [None, None])[-1],
                "rank_ic": metric.get("rank_ic"),
                "rank_ic_ir": metric.get("rank_ic_ir"),
                "cost_adjusted_top_k_excess_return": metric.get(
                    "cost_adjusted_top_k_excess_return"
                ),
                "average_turnover": metric.get("average_turnover"),
                "positive_seed_ratio": stability.get("positive_seed_ratio"),
                "research_gate_passed": gate_passed,
                "research_gate_failures": gate_failures,
                "protocol_version": protocol.get("protocol_version", "legacy"),
                "observation_days": observation_days,
                "latest_trade_date": latest.get("trade_date"),
                "latest_status": latest.get("status", "AWAITING_OBSERVATION"),
                "latest_portfolio": latest.get("portfolio"),
                "latest_report": latest,
                "research_path": str(path),
                "model_dir": str(path.parent / "models"),
                "lifecycle": lifecycle.get(tag, {}),
                "archived": lifecycle.get(tag, {}).get("status") in {
                    "EARLY_REJECTED", "REJECTED", "GATE_REJECTED", "REVIEW_EXPIRED"
                },
            }
        )
    # Portfolio variants reuse an immutable model bundle, not a fabricated
    # training-complete marker. They retain their own reports and paper account.
    by_tag = {r["candidate_tag"]: r for r in rows}
    for registry in sorted((root / "state/qlib-portfolio-candidates").glob("*.json")):
        variant = json.loads(registry.read_text(encoding="utf-8"))
        tag = str(variant["candidate_tag"])
        if len(tag) != 8 or not tag.isdigit() or tag in by_tag:
            raise ValueError("INVALID_OR_DUPLICATE_PORTFOLIO_CANDIDATE_TAG")
        source = by_tag[str(variant["source_candidate_tag"])]
        reports = sorted((root / "state/qlib-candidate-shadow" / tag).glob("*.json"))
        payloads = [json.loads(p.read_text(encoding="utf-8")) for p in reports]
        latest = max(payloads, key=lambda p: str(p.get("trade_date", ""))) if payloads else {}
        receipts = sorted((root / "state/qlib-portfolio-execution" / tag).glob("*.json"))
        latest_execution = json.loads(receipts[-1].read_text(encoding="utf-8")) if receipts else {}
        rows.append({**source, **variant, "account_id": candidate_account_id(tag),
                     "observation_days": len({p["trade_date"] for p in payloads}),
                     "latest_trade_date": latest.get("trade_date"),
                     "latest_status": latest.get("status", "AWAITING_OBSERVATION"),
                     "latest_report": latest, "latest_portfolio": latest.get("portfolio"),
                     "latest_execution": latest_execution,
                     "protocol_version": "portfolio_fixed_budget_forward_v1",
                     "lifecycle": lifecycle.get(tag, {}),
                     "archived": lifecycle.get(tag, {}).get("status") in {
                         "EARLY_REJECTED", "REJECTED", "GATE_REJECTED", "REVIEW_EXPIRED"}})
    used_tags = {row["candidate_tag"] for row in rows}
    for registry in sorted((root / "state/factor-baseline-candidates").glob("*.json")):
        variant = json.loads(registry.read_text(encoding="utf-8"))
        from .research.style_v2_experiment import public_style_details
        variant.update(public_style_details(root, variant))
        tag = str(variant["candidate_tag"])
        if len(tag) != 8 or not tag.isdigit() or tag in used_tags:
            raise ValueError("INVALID_OR_DUPLICATE_FACTOR_CANDIDATE_TAG")
        payloads = [json.loads(p.read_text(encoding="utf-8"))
                    for p in (root / "state/qlib-candidate-shadow" / tag).glob("*.json")]
        latest = max(payloads, key=lambda p: str(p.get("trade_date", ""))) if payloads else {}
        receipts = sorted((root / "state/qlib-portfolio-execution" / tag).glob("*.json"))
        rows.append({**variant, "research_path": str(registry),
                     "research_gate_passed": True, "research_gate_failures": [],
                     "data_cutoff": latest.get("signal_date", latest.get("trade_date")),
                     "protocol_version": "factor_baseline_v2_forward_contract",
                     "observation_days": len({p["trade_date"] for p in payloads}),
                     "latest_trade_date": latest.get("trade_date"),
                     "latest_status": latest.get("status", "AWAITING_OBSERVATION"),
                     "latest_report": latest, "latest_portfolio": latest.get("portfolio"),
                     "latest_execution": json.loads(receipts[-1].read_text(encoding="utf-8")) if receipts else {},
                     "rank_ic": None, "rank_ic_ir": None,
                     "cost_adjusted_top_k_excess_return": None,
                     "lifecycle": lifecycle.get(tag, {}),
                     "archived": lifecycle.get(tag, {}).get("status") in {
                         "EARLY_REJECTED", "REJECTED", "GATE_REJECTED", "REVIEW_EXPIRED"}})
        used_tags.add(tag)
    return sorted(rows, key=lambda r: r["candidate_tag"], reverse=True)


def latest_candidate(root: Path) -> dict[str, Any] | None:
    catalog = candidate_catalog(root)
    return next((row for row in catalog if not row.get("archived")), None)


def run_candidate_daily(
    *,
    candidate: dict[str, Any],
    root: Path,
    store: ParquetStore,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
) -> ChallengerDailyResult:
    tag = str(candidate["candidate_tag"])
    if candidate.get("validation_hold"):
        raise ValueError("WEIGHT_VALIDATION_HOLD")
    if not trading_allowed(root, tag, trade_date):
        broker.cancel_pending(candidate_account_id(tag), "CANDIDATE_LIFECYCLE_FROZEN")
        raise ValueError(f"candidate {tag} is frozen by lifecycle review")
    if candidate.get("portfolio_variant") == "factor_baseline_v2":
        from .baseline_v2_candidate import prepare_factor_v2
        return prepare_factor_v2(root=root, store=store, broker=broker, config=config,
                                 candidate=candidate, trade_date=trade_date)
    model_id = str(
        yaml.safe_load(
            (root / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
        )["model_id"]
    )
    handler = None
    if candidate.get("portfolio_variant") == "fixed_budget_v1":
        from .portfolio_candidate import prepare_portfolio_candidate
        handler = lambda **kwargs: prepare_portfolio_candidate(
            root=root, store=store, broker=broker, config=config, candidate=candidate,
            trade_date=trade_date, **kwargs)
    return run_qlib_challenger_daily(
        root=root,
        store=store,
        broker=broker,
        config=config,
        trade_date=trade_date,
        signal_dir=root / "state" / "qlib-candidate-signals" / tag,
        report_dir=root / "state" / "qlib-candidate-shadow" / tag,
        account_id=candidate_account_id(tag),
        strategy_id=f"{model_id}_candidate_{tag}",
        model_dir=Path(str(candidate["model_dir"])),
        research_path=Path(str(candidate["research_path"])),
        account_history_table=CANDIDATE_HISTORY_TABLE,
        portfolio_handler=handler,
    )


def run_candidate_queue_daily(
    *,
    root: Path,
    store: ParquetStore,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
) -> list[ChallengerDailyResult]:
    challenger_config = yaml.safe_load(
        (root / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
    )
    policy = challenger_config.get("candidate_observation_policy", {})
    formal_days = int(policy.get("formal_review_days", 60))
    candidates = candidate_catalog(root)
    lifecycle = run_lifecycle_review(root, store, candidates, policy, trade_date)
    results = []
    for candidate in reversed(candidates):
        tag = str(candidate["candidate_tag"])
        if tag > trade_date:
            continue
        if candidate.get("validation_hold"):
            continue
        if lifecycle.get(tag, {}).get("status") in FROZEN:
            broker.cancel_pending(candidate_account_id(tag), "CANDIDATE_LIFECYCLE_FROZEN")
            continue
        if not policy.get("enabled") and int(candidate.get("observation_days", 0)) >= formal_days:
            broker.cancel_pending(
                candidate_account_id(tag),
                "CANDIDATE_FORMAL_REVIEW_EVIDENCE_FROZEN",
            )
            continue
        existing_report = (
            root / "state" / "qlib-candidate-shadow" / tag / f"{trade_date}.json"
        )
        if existing_report.exists():
            payload = json.loads(existing_report.read_text(encoding="utf-8"))
            if candidate.get("portfolio_variant") == "factor_baseline_v2" and str(payload.get("signal_date", trade_date)) < trade_date:
                results.append(run_candidate_daily(candidate=candidate, root=root, store=store,
                                                   broker=broker, config=config, trade_date=trade_date))
                continue
            results.append(
                ChallengerDailyResult(
                    trade_date=str(payload["trade_date"]),
                    status=str(payload["status"]),
                    model_id=str(payload["model_id"]),
                    selected=[str(item) for item in payload.get("selected", [])],
                    scores=list(payload.get("scores", [])),
                    executions=list(payload.get("executions", [])),
                    orders=list(payload.get("orders", [])),
                    portfolio=dict(payload.get("portfolio", {})),
                    reconciliation=dict(payload.get("reconciliation", {})),
                    report_path=str(existing_report),
                )
            )
            continue
        results.append(
            run_candidate_daily(
                candidate=candidate,
                root=root,
                store=store,
                broker=broker,
                config=config,
                trade_date=trade_date,
            )
        )
    # Review again after today's close snapshots: checkpoints take effect immediately.
    lifecycle = run_lifecycle_review(root, store, candidates, policy, trade_date)
    for tag, state in lifecycle.items():
        if state.get("status") in FROZEN:
            broker.cancel_pending(candidate_account_id(tag), "CANDIDATE_LIFECYCLE_FROZEN")
    return results


def run_latest_candidate_daily(
    *,
    root: Path,
    store: ParquetStore,
    broker: PaperBroker,
    config: BacktestConfig,
    trade_date: str,
) -> ChallengerDailyResult | None:
    """Backward-compatible single-candidate entry point."""
    candidate = latest_candidate(root)
    if (candidate is None or candidate.get("validation_hold") or str(candidate["candidate_tag"]) > trade_date
            or not trading_allowed(root, str(candidate["candidate_tag"]), trade_date)):
        return None
    return run_candidate_daily(
        candidate=candidate,
        root=root,
        store=store,
        broker=broker,
        config=config,
        trade_date=trade_date,
    )
