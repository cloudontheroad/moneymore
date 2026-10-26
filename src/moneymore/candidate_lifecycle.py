"""Prospective paper-candidate reviews. No orders, fills or NAV rewriting here."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARCHIVED = {"EARLY_REJECTED", "REJECTED", "GATE_REJECTED", "REVIEW_EXPIRED"}
FROZEN = ARCHIVED | {"FORMAL_REVIEW_READY", "REVIEW_HOLD"}


def lifecycle_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "state" / "candidate-lifecycle.sqlite3"
    if not path.exists():
        return {}
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        return {
            tag: json.loads(payload)
            for tag, payload in connection.execute(
                "SELECT candidate_tag, payload FROM candidate_state"
            )
        }


def trading_allowed(root: Path, tag: str, trade_date: str) -> bool:
    path = root / "state" / "candidate-lifecycle.sqlite3"
    if not path.exists():
        return True
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT payload FROM candidate_state WHERE candidate_tag=?", (tag,)
        ).fetchone()
    state = json.loads(row[0]) if row else {}
    # Once stopped, historical compensation must not resurrect a retired account.
    return state.get("status") not in FROZEN


def _history(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty or not {"trade_date", "equity"}.issubset(frame):
        return pd.DataFrame(columns=["trade_date", "equity"])
    result = frame[["trade_date", "equity"]].copy()
    result["trade_date"] = result.trade_date.astype(str).str.replace("-", "", regex=False)
    result["equity"] = pd.to_numeric(result.equity, errors="coerce")
    return (
        result.loc[
            result.trade_date.between(start, end) & np.isfinite(result.equity) & (result.equity > 0)
        ]
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
    )


def review_candidate(
    candidate: dict[str, Any],
    histories: dict[str, pd.DataFrame],
    *,
    as_of: str,
    trustworthy_start: str | None,
    policy: dict[str, Any],
    previous: dict[str, Any] | None = None,
    trial_count: int = 1,
) -> dict[str, Any]:
    from .long_term_review import evaluate_paired_performance

    previous = previous or {}
    if previous.get("status") in ARCHIVED or as_of < str(previous.get("review_date", "")):
        return previous
    state = {
        "candidate_tag": candidate["candidate_tag"],
        "review_date": as_of,
        "policy_version": policy.get("version", "candidate_lifecycle_v1"),
        "policy_effective_date": policy["effective_date"],
        "promotion_automatic": False,
        "history_preserved": True,
        "position_policy": "FREEZE_LEDGER_NO_SYNTHETIC_LIQUIDATION",
        "checkpoint": previous.get("checkpoint", 0),
        "common_observation_days": previous.get("common_observation_days", 0),
    }
    if not candidate.get("research_gate_passed"):
        return {
            **state,
            "status": "GATE_REJECTED",
            "reason": "研究门禁未通过，退出候选竞赛",
            "gate_failures": candidate.get("research_gate_failures", []),
        }
    if not trustworthy_start:
        return {**state, "status": "REVIEW_HOLD", "reason": "缺少可信数据起点，暂停评审及新委托"}
    trustworthy_start = str(previous.get("trustworthy_start_date") or trustworthy_start)
    state["trustworthy_start_date"] = trustworthy_start
    frames = {key: _history(value, trustworthy_start, as_of) for key, value in histories.items()}
    market_dates = frames["baseline"].trade_date
    if previous.get("status") in {"FORMAL_REVIEW_READY", "REVIEW_HOLD"} and previous.get(
        "formal_review_date"
    ):
        elapsed = int((market_dates > previous["formal_review_date"]).sum())
        if elapsed >= int(policy.get("review_expiry_trading_days", 10)):
            return {
                **previous,
                "review_date": as_of,
                "status": "REVIEW_EXPIRED",
                "reason": "正式评审有效期已过，归档保留全部历史；不自动晋级",
            }
        return previous
    common = sorted(set.intersection(*(set(frame.trade_date) for frame in frames.values())))
    days = len(common)
    state["common_observation_days"] = days
    formal_days = int(policy.get("formal_review_days", 60))
    checkpoints = [int(day) for day in policy.get("checkpoints", [20, 40, 60])]
    due = max((day for day in checkpoints if day <= days), default=0)
    if (
        previous
        and due <= int(previous.get("checkpoint", 0))
        and previous.get("status") != "REVIEW_HOLD"
    ):
        return {**previous, **state}
    if due == 0:
        return {**state, "status": "OBSERVING", "reason": "未到首个固定检查点，不按短期收益淘汰"}
    # Evaluate only evidence available at this check; never manufacture past review dates.
    aligned = {
        key: frame.loc[frame.trade_date.isin(common)].head(due) for key, frame in frames.items()
    }
    comparisons = {
        name: evaluate_paired_performance(
            aligned[name],
            aligned["candidate"],
            trustworthy_start_date=trustworthy_start,
            trial_count=max(trial_count, 1) * len(checkpoints) * 2,
        )
        for name in ("baseline", "challenger")
    }
    state.update(
        checkpoint=due,
        comparisons=comparisons,
        evidence_end=str(aligned["candidate"].iloc[-1].trade_date),
    )
    significant_underperformance = all(
        result.get("annualized_excess_ci_high") is not None
        and result["annualized_excess_ci_high"] < 0
        and result.get("qlib_risk_normalized_return") is not None
        and result.get("factor_risk_normalized_return") is not None
        and result["qlib_risk_normalized_return"] < result["factor_risk_normalized_return"]
        for result in comparisons.values()
    )
    if significant_underperformance:
        return {
            **state,
            "status": "REJECTED" if due >= formal_days else "EARLY_REJECTED",
            "reason": "固定检查点：相对两个对照的成本后超额置信上界均低于零，且风险调整收益均落后",
        }
    if due >= formal_days:
        return {
            **state,
            "status": "FORMAL_REVIEW_READY",
            "formal_review_date": as_of,
            "reason": "完成正式观察，冻结新交易，等待人工审查收益、回撤、成本与漂移；不自动晋级",
        }
    return {
        **state,
        "status": "CONTINUE_TO_60D",
        "reason": "检查点未发现足以淘汰的证据，继续独立观察",
    }


def run_lifecycle_review(
    root: Path,
    store: Any,
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
    trade_date: str,
) -> dict[str, dict[str, Any]]:
    if not policy.get("enabled") or trade_date < str(policy["effective_date"]):
        return lifecycle_rows(root)

    def read(table: str) -> pd.DataFrame:
        try:
            return store.read(table)
        except FileNotFoundError:
            return pd.DataFrame()

    baseline = read("multi_sector_account_daily")
    challenger = read("qlib_challenger_account_daily")
    history = read("qlib_candidate_account_daily")
    boundary = root / "state" / "point-in-time" / "latest.json"
    trust = (
        (
            json.loads(boundary.read_text(encoding="utf-8"))
            .get("historical_replay_gate", {})
            .get("earliest_trustworthy_date")
        )
        if boundary.exists()
        else None
    )
    if trust:
        trust = str(trust).replace("-", "")
    database = root / "state" / "candidate-lifecycle.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database, timeout=30) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS candidate_state (candidate_tag TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS reviews (candidate_tag TEXT, review_date TEXT, payload TEXT, PRIMARY KEY(candidate_tag, review_date))"
        )
        connection.execute("BEGIN IMMEDIATE")
        previous = {
            tag: json.loads(payload)
            for tag, payload in connection.execute(
                "SELECT candidate_tag, payload FROM candidate_state"
            )
        }
        for candidate in candidates:
            tag = str(candidate["candidate_tag"])
            if tag > trade_date:
                continue
            own = (
                history.loc[history.account_id.astype(str) == candidate["account_id"]]
                if not history.empty
                else history
            )
            state = review_candidate(
                candidate,
                {"baseline": baseline, "challenger": challenger, "candidate": own},
                as_of=trade_date,
                trustworthy_start=trust,
                policy=policy,
                previous=previous.get(tag),
                trial_count=len(candidates),
            )
            if state == previous.get(tag):
                continue
            payload = json.dumps(_finite_json(state), ensure_ascii=False, allow_nan=False)
            connection.execute(
                "INSERT INTO candidate_state VALUES (?, ?) ON CONFLICT(candidate_tag) DO UPDATE SET payload=excluded.payload",
                (tag, payload),
            )
            connection.execute(
                "INSERT INTO reviews VALUES (?, ?, ?) ON CONFLICT(candidate_tag, review_date) DO UPDATE SET payload=excluded.payload",
                (tag, trade_date, payload),
            )
    return lifecycle_rows(root)


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
