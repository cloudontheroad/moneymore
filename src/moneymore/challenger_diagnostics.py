"""Read-only model diagnostics; derived artifacts never alter paper ledgers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


def write_diagnostic(path: Path, payload: dict[str, Any]) -> None:
    as_of = str(payload.get("as_of_date", ""))
    if as_of:
        dated = path.parent / f"{path.stem}-history" / f"{as_of}.json"
        _atomic_json(dated, payload)
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if str(previous.get("as_of_date", "")) > as_of:
                return
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(clean_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def prediction_reports(directory: Path, as_of: str) -> list[dict[str, Any]]:
    by_date: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        raw = path.read_bytes()
        row = json.loads(raw)
        date = str(row.get("trade_date", ""))
        if not date or date > as_of or not row.get("scores"):
            continue
        row = {**row, "source_sha256": hashlib.sha256(raw).hexdigest(), "source_file": path.name}
        by_date.setdefault(date, []).append(row)
    # Select the first recorded prediction for each date, not the last repair.
    return [
        min(rows, key=lambda row: str(row.get("created_at") or "9999"))
        for _, rows in sorted(by_date.items())
    ]


def trading_dates(store: Any, as_of: str) -> list[str]:
    frame = store.read("trade_calendar")
    return sorted(
        set(
            frame.loc[
                (frame.exchange == "SSE") & (frame.is_open.astype(int) == 1), "cal_date"
            ].astype(str)
        )
    )


def align_trading_observations(bars: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """Keep missing sessions as NaN, never compress them into adjacent bars."""
    if bars.empty:
        return bars.copy()
    result = bars.copy()
    result["date"] = pd.to_datetime(result.date).dt.strftime("%Y%m%d")
    first, last = result.date.min(), result.date.max()
    dates = [date for date in calendar if first <= date <= last]
    if not dates:
        raise ValueError("No exchange sessions cover model input bars")
    return (
        result.drop_duplicates("date", keep="last")
        .set_index("date")
        .reindex(dates)
        .rename_axis("date")
        .reset_index()
    )


def evaluate_forward_reports(
    reports: list[dict[str, Any]],
    daily: pd.DataFrame,
    calendar: list[str],
    *,
    as_of: str,
    horizon: int = 5,
    minimum_coverage: float = 0.95,
) -> dict[str, Any]:
    if horizon < 1:
        raise ValueError("label horizon must be positive")
    bars = daily.copy()
    bars["trade_date"] = bars.trade_date.astype(str)
    bars = bars.loc[bars.trade_date <= as_of].drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    price = bars.pivot(index="trade_date", columns="ts_code", values="close")
    days = sorted(set(calendar))
    outcomes, official = [], []
    for report in reports:
        date = str(report["trade_date"])
        if date > as_of:
            continue
        record: dict[str, Any] = {
            "observation_date": date,
            "source_sha256": report.get("source_sha256"),
            "source_file": report.get("source_file"),
            "prediction_created_at": report.get("created_at"),
            "rank_ic": None,
            "raw_rank_ic": None,
            "selected_excess_return": None,
        }
        if date not in days:
            outcomes.append({**record, "status": "INVALID_OBSERVATION_DATE"})
            continue
        location = days.index(date)
        if location + horizon >= len(days) or days[location + horizon] > as_of:
            outcomes.append({**record, "status": "AWAITING_LABEL_HORIZON"})
            continue
        maturity = days[location + horizon]
        record["maturity_date"] = maturity
        scores = pd.DataFrame(report.get("scores", []))
        required = {"instrument", "score", "sector", "datetime"}
        if not required.issubset(scores) or scores.instrument.duplicated().any():
            outcomes.append({**record, "status": "INVALID_SCORES"})
            continue
        if set(scores.datetime.astype(str).str.replace("-", "", regex=False)) != {date}:
            outcomes.append({**record, "status": "SCORE_DATE_MISMATCH"})
            continue
        if date not in price.index or maturity not in price.index:
            outcomes.append({**record, "status": "MISSING_MARKET_DATA"})
            continue
        start, end = price.loc[date], price.loc[maturity]
        scores["score"] = pd.to_numeric(scores.score, errors="coerce")
        scores["raw_return"] = scores.instrument.map(end) / scores.instrument.map(start) - 1
        valid = scores.loc[
            np.isfinite(scores.score)
            & np.isfinite(scores.raw_return)
            & (scores.instrument.map(start) > 0)
            & (scores.instrument.map(end) > 0)
        ].copy()
        coverage = len(valid) / len(scores) if len(scores) else 0
        valid["label"] = valid.raw_return - valid.groupby("sector").raw_return.transform("mean")
        rank_ic = (
            valid.score.corr(valid.label, method="spearman")
            if len(valid) > 2 and valid.label.nunique() > 1 and valid.score.nunique() > 1
            else None
        )
        raw_ic = (
            valid.score.corr(valid.raw_return, method="spearman")
            if len(valid) > 2 and valid.raw_return.nunique() > 1 and valid.score.nunique() > 1
            else None
        )
        selected = set(report.get("selected", []))
        selected_valid = selected.issubset(set(valid.instrument)) and bool(selected)
        mean_selected = (
            valid.loc[valid.instrument.isin(selected), "label"].mean() if selected_valid else None
        )
        if rank_ic is not None and not np.isfinite(rank_ic):
            rank_ic = None
        selected_valid = (
            selected_valid and mean_selected is not None and bool(np.isfinite(mean_selected))
        )
        captured = pd.to_datetime(report.get("created_at"), utc=True, errors="coerce")
        deadline = pd.Timestamp(days[location + 1] + " 09:30", tz="Asia/Shanghai").tz_convert("UTC")
        not_before = pd.Timestamp(date + " 15:00", tz="Asia/Shanghai").tz_convert("UTC")
        prospective = not pd.isna(captured) and not_before <= captured < deadline
        status = (
            "MATURED"
            if coverage >= minimum_coverage and rank_ic is not None and selected_valid
            else "INCOMPLETE_LABELS"
        )
        record.update(
            status=status,
            coverage=coverage,
            sample_count=len(valid),
            expected_samples=len(scores),
            missing_symbols=sorted(set(scores.instrument) - set(valid.instrument)),
            rank_ic=rank_ic,
            raw_rank_ic=raw_ic,
            selected_excess_return=mean_selected,
            prospective=bool(prospective),
            promotion_evidence=bool(prospective and status == "MATURED"),
        )
        outcomes.append(record)
        if record["promotion_evidence"]:
            official.append(record)
    values = pd.Series([row["rank_ic"] for row in official], dtype=float)
    latest = max((row["observation_date"] for row in outcomes), default=None)
    stale = latest is None or latest < max((date for date in days if date <= as_of), default=as_of)
    return clean_json(
        {
            "as_of_date": as_of,
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "STALE_PREDICTIONS"
            if stale
            else "READY"
            if len(official)
            else "COLLECTING_EVIDENCE",
            "label_definition": "RAW_CLOSE_T_PLUS_H_OVER_CLOSE_T_MINUS_ONE_SECTOR_DEMEANED_MATCHES_TRAINING",
            "label_horizon": horizon,
            "minimum_coverage": minimum_coverage,
            "automatic_promotion_eligible": False,
            "matured_days": len(official),
            "diagnostic_matured_days": sum(row["status"] == "MATURED" for row in outcomes),
            "pending_days": sum(row["status"] == "AWAITING_LABEL_HORIZON" for row in outcomes),
            "incomplete_days": sum(
                row["status"] not in {"MATURED", "AWAITING_LABEL_HORIZON"} for row in outcomes
            ),
            "rank_ic": values.mean() if len(values) else None,
            "rolling_rank_ic": values.tail(20).mean() if len(values) else None,
            "selected_excess_return": pd.Series(
                [row["selected_excess_return"] for row in official], dtype=float
            ).mean()
            if official
            else None,
            "rank_ic_ir": values.mean() / values.std(ddof=1)
            if len(values) > 1 and values.std(ddof=1) > 0
            else None,
            "daily": official,
            "all_observations": outcomes,
            "evidence_boundary": "Created after close and before next SSE open; locally recorded, not independently timestamp-attested. Backfilled predictions excluded from promotion.",
        }
    )


def run_forward_evaluations(root: Path, store: Any, as_of: str) -> dict[str, Any]:
    from .qlib_candidate_observer import candidate_catalog

    accounts = [
        (
            "qlib_gru_shadow",
            root / "state/qlib-challenger-shadow",
            root / "state/qlib-challenger/forward-evaluation.json",
            root / "state/qlib-challenger/latest-research.json",
        )
    ]
    accounts.extend(
        (
            row["account_id"],
            root / "state/qlib-candidate-shadow" / row["candidate_tag"],
            root / "state/qlib-candidate-forward" / f"{row['candidate_tag']}.json",
            Path(row["research_path"]),
        )
        for row in candidate_catalog(root)
    )
    loaded = {
        account: prediction_reports(directory, as_of) for account, directory, _, _ in accounts
    }
    symbols = sorted(
        {
            str(score["instrument"])
            for rows in loaded.values()
            for row in rows
            for score in row["scores"]
        }
    )
    start = min((row["trade_date"] for rows in loaded.values() for row in rows), default=as_of)
    daily = (
        store.read(
            "daily",
            columns=["ts_code", "trade_date", "close"],
            filters=[
                ("trade_date", ">=", start),
                ("trade_date", "<=", as_of),
                ("ts_code", "in", symbols),
            ],
        )
        if symbols
        else pd.DataFrame(columns=["ts_code", "trade_date", "close"])
    )
    calendar = trading_dates(store, as_of)
    summaries = []
    for account, _, target, research_path in accounts:
        research = (
            json.loads(research_path.read_text(encoding="utf-8")) if research_path.exists() else {}
        )
        horizon = int(research.get("protocol", {}).get("label_horizon", 5))
        result = evaluate_forward_reports(
            loaded[account], daily, calendar, as_of=as_of, horizon=horizon
        )
        result["account_id"] = account
        write_diagnostic(target, result)
        summaries.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"daily", "all_observations"}
            }
        )
    payload = {"as_of_date": as_of, "accounts": summaries, "writes_orders": False}
    write_diagnostic(root / "state/challenger-diagnostics/forward-summary.json", payload)
    return payload


def run_challenger_review(root: Path, store: Any, as_of: str) -> dict[str, Any]:
    """Run in evidence order; recommendations are NOT lifecycle commands."""
    from .challenger_attribution import run_style_attribution
    from .qlib_candidate_observer import candidate_catalog

    forward = run_forward_evaluations(root, store, as_of)
    attribution = run_style_attribution(root, store, as_of)
    archived = {row["account_id"] for row in candidate_catalog(root) if row["archived"]}
    styles = {row["account_id"]: row for row in attribution.get("accounts", [])}
    recommendations = []
    for evidence in forward["accounts"]:
        account = evidence["account_id"]
        style = styles.get(account, {})
        if account in archived:
            status, reason = "ALREADY_ARCHIVED", "沿用已归档决定，不自动复活"
        elif style.get("status") != "COMPLETE":
            status, reason = (
                "REPAIR_EVIDENCE_FIRST",
                "风格匹配证据不完整，先修复输入；不能按残缺归因淘汰模型",
            )
        elif evidence["matured_days"] < 20:
            status, reason = "OBSERVE_NO_PROMOTION", "成熟前瞻样本不足20日，保留隔离观察，不晋级"
        elif (
            (evidence.get("rolling_rank_ic") or 0) < 0
            and style["contributions"]["selection"] < 0
            and all(
                value["matched_days"] == style["expected_return_days"]
                and value["contributions"]["selection"] < 0
                for value in style.get("sensitivity", {}).values()
            )
        ):
            status, reason = (
                "RESEARCH_REDESIGN",
                "前瞻排序与风格匹配后的选股贡献均弱，建议新研究协议；不继续复制同类训练版本",
            )
        elif style["contributions"]["style"] < 0 and style["contributions"]["selection"] >= 0:
            status, reason = (
                "REVIEW_RISK_EXPOSURE",
                "亏损含负风格贡献，而匹配后选股非负；优先研究风险暴露，不直接否定选股模型",
            )
        else:
            status, reason = "CONTINUE_ISOLATED_REVIEW", "继续隔离观察，尚不足以作永久淘汰结论"
        recommendations.append(
            {
                "account_id": account,
                "recommendation": status,
                "reason": reason,
                "automatic_action": False,
                "matured_days": evidence["matured_days"],
            }
        )
    payload = {
        "as_of_date": as_of,
        "forward": forward,
        "style_attribution": attribution,
        "recommendations": recommendations,
        "automatic_model_replacement": False,
        "writes_orders": False,
        "rewrites_account_history": False,
    }
    write_diagnostic(root / "state/challenger-diagnostics/latest.json", payload)
    return payload
