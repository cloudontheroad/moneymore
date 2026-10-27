from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data.store import ParquetStore
from .execution.paper import PaperBroker


def bootstrap_qlib_release(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "qlib_challenger.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_id = str(config["model_id"])
    model_dir = root / "state" / "qlib-challenger" / "models"
    manifest_path = model_dir / f"{model_id}_ensemble.json"
    if not manifest_path.exists():
        return {
            "status": "AWAITING_ARTIFACT",
            "model_id": model_id,
            "execution_mode": "OBSERVATION_ONLY",
            "releases": [],
            "transitions": [],
        }
    artifact_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    release_id = f"{model_id}@{artifact_hash[:12]}"
    database = root / "state" / "qlib-governance" / "governance.sqlite3"
    _initialize(database)
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO releases(
                release_id, model_id, lifecycle, artifact_hash,
                artifact_path, data_cutoff, created_at
            ) VALUES (?, ?, 'SHADOW', ?, ?, ?, ?)
            """,
            (
                release_id,
                model_id,
                artifact_hash,
                str(manifest_path),
                str(config["test"]["end"]).replace("-", ""),
                now,
            ),
        )
    pointer = _deployment_path(root)
    if not pointer.exists():
        _write_json(
            pointer,
            {
                "release_id": release_id,
                "model_id": model_id,
                "lifecycle": "SHADOW",
                "execution_mode": "PAPER_TRADING",
                "reason": "INITIAL_EXPERIMENTAL_RELEASE",
                "updated_at": now,
            },
        )
    return deployment_snapshot(root)


def register_qlib_candidate(
    root: Path,
    *,
    model_id: str,
    artifact_path: Path,
    data_cutoff: str,
) -> dict[str, Any]:
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    release_id = f"{model_id}@{artifact_hash[:12]}"
    database = root / "state" / "qlib-governance" / "governance.sqlite3"
    _initialize(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO releases(
                release_id, model_id, lifecycle, artifact_hash,
                artifact_path, data_cutoff, created_at
            ) VALUES (?, ?, 'CANDIDATE', ?, ?, ?, ?)
            """,
            (
                release_id,
                model_id,
                artifact_hash,
                str(artifact_path),
                data_cutoff,
                datetime.now(UTC).isoformat(),
            ),
        )
    return {"release_id": release_id, "lifecycle": "CANDIDATE"}


def promote_qlib_candidate(
    root: Path,
    *,
    release_id: str,
    gate_passed: bool,
    reason: str,
) -> dict[str, Any]:
    if not gate_passed:
        raise ValueError("candidate cannot be promoted before its gate passes")
    return transition_release(
        root,
        release_id=release_id,
        lifecycle="ACTIVE",
        execution_mode="PAPER_TRADING",
        reason=reason,
    )


def rollback_qlib_release(root: Path, reason: str) -> dict[str, Any]:
    current = deployment_snapshot(root)
    candidates = [
        row
        for row in current.get("releases", [])
        if row["release_id"] != current.get("release_id")
        and row["lifecycle"] in {"SHADOW", "ACTIVE", "RETIRED"}
    ]
    if not candidates:
        raise ValueError("no previous release is available for rollback")
    target = candidates[0]
    return transition_release(
        root,
        release_id=str(target["release_id"]),
        lifecycle="ACTIVE",
        execution_mode="PAPER_TRADING",
        reason=f"ROLLBACK:{reason}",
    )


def transition_release(
    root: Path,
    *,
    release_id: str,
    lifecycle: str,
    execution_mode: str,
    reason: str,
    automatic: bool = False,
) -> dict[str, Any]:
    allowed_lifecycles = {"CANDIDATE", "SHADOW", "ACTIVE", "DEGRADED", "RETIRED"}
    allowed_modes = {"OBSERVATION_ONLY", "PAPER_TRADING"}
    if lifecycle not in allowed_lifecycles or execution_mode not in allowed_modes:
        raise ValueError("invalid lifecycle transition")
    current = bootstrap_qlib_release(root)
    database = root / "state" / "qlib-governance" / "governance.sqlite3"
    with sqlite3.connect(database) as connection:
        release = connection.execute(
            "SELECT model_id FROM releases WHERE release_id = ?", (release_id,)
        ).fetchone()
        if release is None:
            raise ValueError(f"unknown release: {release_id}")
        now = datetime.now(UTC).isoformat()
        connection.execute(
            "UPDATE releases SET lifecycle = ? WHERE release_id = ?",
            (lifecycle, release_id),
        )
        connection.execute(
            """
            INSERT INTO transitions(
                release_id, from_lifecycle, to_lifecycle, from_mode,
                to_mode, reason, transition_type, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                release_id,
                current.get("lifecycle"),
                lifecycle,
                current.get("execution_mode"),
                execution_mode,
                reason,
                "AUTOMATIC" if automatic else "MANUAL",
                now,
            ),
        )
    _write_json(
        _deployment_path(root),
        {
            "release_id": release_id,
            "model_id": release[0],
            "lifecycle": lifecycle,
            "execution_mode": execution_mode,
            "reason": reason,
            "updated_at": now,
        },
    )
    return deployment_snapshot(root)


def evaluate_qlib_drift(
    root: Path,
    store: ParquetStore,
    broker: PaperBroker,
    trade_date: str,
    *,
    apply_protection: bool = True,
) -> dict[str, Any]:
    deployment = bootstrap_qlib_release(root)
    config = yaml.safe_load(
        (root / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
    )
    drift = config["drift_monitoring"]
    reports = _latest_reports(root / "state" / "qlib-challenger-shadow")
    recent_reports = reports[-int(drift["recent_window"]) :]
    scores = [
        float(row["score"])
        for report in recent_reports
        for row in report.get("scores", [])
    ]
    reference_reports = reports[: int(drift["reference_window"])]
    reference_scores = [
        float(row["score"])
        for report in reference_reports
        for row in report.get("scores", [])
    ]
    score_psi = (
        _psi(reference_scores, scores)
        if len(reports)
        >= int(drift["reference_window"]) + int(drift["recent_window"])
        else None
    )
    turnover = _average_turnover(recent_reports)
    forward_path = root / "state" / "qlib-challenger" / "forward-evaluation.json"
    dated_forward = forward_path.parent / "forward-evaluation-history" / f"{trade_date}.json"
    if dated_forward.exists():
        forward_path = dated_forward
    forward = (
        json.loads(forward_path.read_text(encoding="utf-8"))
        if forward_path.exists()
        else {"daily": []}
    )
    daily_ic = pd.Series(
        [row["rank_ic"] for row in forward.get("daily", [])
         if str(row.get("maturity_date", "")).replace("-", "") <= trade_date], dtype=float
    )
    rolling_ic = (
        float(daily_ic.tail(int(drift["recent_window"])).mean())
        if len(daily_ic)
        else None
    )
    feature_psi = _feature_drift(store, int(drift["recent_window"]))
    fills = broker.fills("qlib_gru_shadow")
    recent_fill_cost = sum(
        float(row["fee"])
        for row in fills
        if str(row["trade_date"]) >= _window_start(recent_reports)
    )
    try:
        account = store.read("qlib_challenger_account_daily")
        latest_equity = float(account.sort_values("trade_date").iloc[-1]["equity"])
    except (FileNotFoundError, IndexError):
        latest_equity = 0.0
    cost_ratio = recent_fill_cost / latest_equity if latest_equity else 0.0
    matured = len(daily_ic)
    observed = len(recent_reports)
    breaches = []
    if matured >= int(drift["minimum_observations"]):
        if rolling_ic is not None and rolling_ic < float(drift["minimum_rolling_rank_ic"]):
            breaches.append("ROLLING_RANK_IC")
        if score_psi is not None and score_psi > float(drift["maximum_score_psi"]):
            breaches.append("SCORE_PSI")
        if (
            bool(drift.get("enforce_feature_psi", False))
            and feature_psi is not None
            and feature_psi > float(drift["maximum_feature_psi"])
        ):
            breaches.append("FEATURE_PSI")
        if turnover is not None and turnover > float(drift["maximum_turnover"]):
            breaches.append("TURNOVER")
        if cost_ratio > float(drift["maximum_cost_ratio"]):
            breaches.append("COST_RATIO")
    status = "WARMING_UP" if matured < int(drift["minimum_observations"]) else (
        "BREACH" if breaches else "PASS"
    )
    if apply_protection and breaches and deployment["execution_mode"] == "PAPER_TRADING":
        deployment = transition_release(
            root,
            release_id=str(deployment["release_id"]),
            lifecycle="DEGRADED",
            execution_mode="OBSERVATION_ONLY",
            reason="DRIFT:" + ",".join(breaches),
            automatic=True,
        )
        broker.cancel_pending("qlib_gru_shadow", "QLIB_DRIFT_DEGRADED", side="BUY")
    payload = {
        "trade_date": trade_date,
        "status": status,
        "observed_prediction_days": observed,
        "matured_forward_days": matured,
        "protection_mode": "EXISTING_AUTOMATIC_POLICY" if apply_protection else "DIAGNOSTIC_ONLY",
        "rolling_rank_ic": rolling_ic,
        "score_psi": score_psi,
        "maximum_feature_psi": feature_psi,
        "feature_psi_evidence": "POINT_IN_TIME_FACTOR_PROXY",
        "feature_psi_enforced": bool(drift.get("enforce_feature_psi", False)),
        "average_turnover": turnover,
        "cost_ratio": cost_ratio,
        "breaches": breaches,
        "deployment": deployment,
        "training_policy": {
            "frequency": config["training_policy"]["frequency"],
            "automatic_promotion": False,
            "next_action": "TRAIN_CANDIDATE_ONLY",
        },
    }
    _write_json(root / "state" / "qlib-governance" / "latest-drift.json", payload)
    return payload


def deployment_snapshot(root: Path) -> dict[str, Any]:
    pointer = _deployment_path(root)
    if not pointer.exists():
        return {}
    deployment = json.loads(pointer.read_text(encoding="utf-8"))
    database = root / "state" / "qlib-governance" / "governance.sqlite3"
    _initialize(database)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        releases = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM releases ORDER BY created_at DESC"
            )
        ]
        transitions = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM transitions ORDER BY id DESC LIMIT 100"
            )
        ]
    return {**deployment, "releases": releases, "transitions": transitions}


def _feature_drift(store: ParquetStore, window: int) -> float | None:
    try:
        frame = store.read("point_in_time_features").sort_values("as_of_date")
    except FileNotFoundError:
        return None
    dates = sorted(frame["as_of_date"].astype(str).unique())
    if len(dates) < window * 2:
        return None
    reference = frame.loc[frame["as_of_date"].astype(str).isin(dates[:window])]
    recent = frame.loc[frame["as_of_date"].astype(str).isin(dates[-window:])]
    values = []
    excluded = {
        "as_of_date",
        "symbol",
        "market_available_rule",
        "financial_available_rule",
    }
    for column in frame.columns:
        if column in excluded:
            continue
        value = _psi(reference[column], recent[column])
        if value is not None:
            values.append(value)
    return max(values) if values else None


def _psi(reference: Any, current: Any, bins: int = 10) -> float | None:
    left = pd.to_numeric(pd.Series(reference), errors="coerce").dropna()
    right = pd.to_numeric(pd.Series(current), errors="coerce").dropna()
    if len(left) < bins * 2 or len(right) < bins * 2 or left.nunique() < 2:
        return None
    edges = np.unique(left.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    left_dist = pd.cut(left, edges).value_counts(normalize=True, sort=False)
    right_dist = pd.cut(right, edges).value_counts(normalize=True, sort=False)
    left_values = left_dist.to_numpy().clip(1e-6)
    right_values = right_dist.to_numpy().clip(1e-6)
    return float(
        np.sum((right_values - left_values) * np.log(right_values / left_values))
    )


def _average_turnover(reports: list[dict[str, Any]]) -> float | None:
    selections = [set(report.get("selected", [])) for report in reports]
    if len(selections) < 2:
        return None
    values = [
        len(current.symmetric_difference(previous))
        / max(len(current) + len(previous), 1)
        for previous, current in pairwise(selections)
    ]
    return float(np.mean(values))


def _latest_reports(path: Path) -> list[dict[str, Any]]:
    by_date: dict[str, dict[str, Any]] = {}
    for report_path in sorted(path.glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        date = str(report.get("trade_date", ""))
        if date and str(report.get("created_at", "")) >= str(
            by_date.get(date, {}).get("created_at", "")
        ):
            by_date[date] = report
    return [by_date[key] for key in sorted(by_date)]


def _window_start(reports: list[dict[str, Any]]) -> str:
    return str(reports[0]["trade_date"]) if reports else "99991231"


def _deployment_path(root: Path) -> Path:
    return root / "state" / "qlib-governance" / "deployment.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _initialize(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS releases (
                release_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                lifecycle TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                data_cutoff TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                release_id TEXT NOT NULL,
                from_lifecycle TEXT,
                to_lifecycle TEXT NOT NULL,
                from_mode TEXT,
                to_mode TEXT NOT NULL,
                reason TEXT NOT NULL,
                transition_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
