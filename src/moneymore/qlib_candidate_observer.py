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
    if not base.exists():
        return rows
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
    return rows


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
    if not trading_allowed(root, tag, trade_date):
        broker.cancel_pending(candidate_account_id(tag), "CANDIDATE_LIFECYCLE_FROZEN")
        raise ValueError(f"candidate {tag} is frozen by lifecycle review")
    model_id = str(
        yaml.safe_load(
            (root / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
        )["model_id"]
    )
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
    if (candidate is None or str(candidate["candidate_tag"]) > trade_date
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
