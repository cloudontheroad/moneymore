"""Resumable, isolated financial-data repair for a prospective baseline version.

Never writes the live fina_indicator table, orders, holdings or equity histories.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..challenger_diagnostics import write_diagnostic
from .store import ParquetStore

TABLE = "fina_indicator_research_v2"


def research_financial_frame(store: ParquetStore) -> pd.DataFrame:
    """Read only core/provenance columns, with separately approved arithmetic supplements."""
    from pyarrow import parquet as pq

    names = pq.read_schema(store.curated / (TABLE + ".parquet")).names
    wanted = [
        "ts_code",
        "ann_date",
        "end_date",
        "roe",
        "q_sales_yoy",
        "conflict__roe",
        "conflict__q_sales_yoy",
        "_payload_capture",
    ]
    frame = store.read(TABLE, columns=[c for c in wanted if c in names])
    for field in ("roe", "q_sales_yoy"):
        if field not in frame:
            frame[field] = float("nan")
    frame["growth_source"] = "TUSHARE_FINA_INDICATOR"
    try:
        supplement = store.read("quarterly_revenue_supplement_v2")
    except FileNotFoundError:
        return frame
    supplement = supplement.loc[
        supplement.status.eq("AGREEMENT_OF_CUMULATIVE_AND_SINGLE_QUARTER")
    ].copy()
    if supplement.empty:
        return frame
    supplement["growth_source"] = "TUSHARE_INCOME_CROSS_CHECKED_ARITHMETIC"
    supplement["roe"] = float("nan")
    supplement["conflict__roe"] = False
    supplement["conflict__q_sales_yoy"] = False
    return pd.concat(
        [frame, supplement[[c for c in [*wanted, "growth_source"] if c in supplement]]],
        ignore_index=True,
    )


def visible_financials(frame: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Same conservative announcement+one-calendar-day rule as live baseline."""
    frame = frame.copy()
    if frame.empty:
        return frame
    available = pd.to_datetime(frame.ann_date, format="%Y%m%d", errors="coerce") + pd.offsets.Day(1)
    return (
        frame.loc[available <= pd.Timestamp(as_of)]
        .sort_values(["ts_code", "ann_date", "end_date"])
        .groupby("ts_code", as_index=False)
        .tail(1)
    )


def financial_features_as_of(frame: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Latest non-null value per field, with its own announcement and period provenance.

    Never substitutes cumulative growth for quarterly growth, or missing values with zero.
    This is a NEW research-version rule, not the legacy latest-whole-row behavior.
    """
    base = visible_financials(frame, as_of).set_index("ts_code")
    if base.empty:
        return base.reset_index()
    available = pd.to_datetime(frame.ann_date, format="%Y%m%d", errors="coerce") + pd.offsets.Day(1)
    eligible = frame.loc[available <= pd.Timestamp(as_of)].sort_values(
        ["ts_code", "end_date", "ann_date"]
    )
    for field in ("roe", "q_sales_yoy"):
        valid = (
            pd.to_numeric(eligible[field], errors="coerce")
            .replace([float("inf"), -float("inf")], float("nan"))
            .notna()
        )
        candidates = eligible.loc[valid].copy()
        if field == "q_sales_yoy" and "growth_source" in eligible:
            is_supplement = eligible.growth_source.eq("TUSHARE_INCOME_CROSS_CHECKED_ARITHMETIC")
            provider = eligible.loc[~is_supplement]
            provider_valid = provider.loc[
                pd.to_numeric(provider[field], errors="coerce").notna(), ["ts_code", "end_date"]
            ]
            if f"conflict__{field}" in provider:
                conflicts = provider.loc[
                    provider[f"conflict__{field}"].eq(True), ["ts_code", "end_date"]
                ]
                provider_valid = pd.concat([provider_valid, conflicts])
            known = pd.MultiIndex.from_frame(provider_valid.drop_duplicates())
            candidates = candidates.loc[
                ~(
                    candidates.growth_source.eq("TUSHARE_INCOME_CROSS_CHECKED_ARITHMETIC")
                    & pd.MultiIndex.from_frame(candidates[["ts_code", "end_date"]]).isin(known)
                )
            ]
        latest = candidates.groupby("ts_code", as_index=False).tail(1).set_index("ts_code")
        base[field] = latest[field]
        base[f"{field}_ann_date"] = latest.ann_date
        base[f"{field}_end_date"] = latest.end_date
        if field == "q_sales_yoy" and "growth_source" in latest:
            base["growth_source"] = latest.growth_source
        conflict_column = f"conflict__{field}"
        if conflict_column in eligible:
            conflict_input = (
                eligible.loc[eligible.growth_source.ne("TUSHARE_INCOME_CROSS_CHECKED_ARITHMETIC")]
                if "growth_source" in eligible
                else eligible
            )
            newest = conflict_input.groupby("ts_code", as_index=False).tail(1).set_index("ts_code")
            blocked = newest[conflict_column].eq(True).reindex(base.index, fill_value=False)
            base.loc[blocked, field] = float("nan")
            base[conflict_column] = blocked
    base["ann_date"] = base[["roe_ann_date", "q_sales_yoy_ann_date"]].fillna("").max(axis=1)
    base["end_date"] = base[["roe_end_date", "q_sales_yoy_end_date"]].fillna("").max(axis=1)
    return base.reset_index()


def financial_quality(
    frame: pd.DataFrame, symbols: list[str], as_of: str, *, per_field: bool = False
) -> dict:
    visible = (
        financial_features_as_of(frame, as_of) if per_field else visible_financials(frame, as_of)
    )
    visible = visible.loc[visible.ts_code.isin(symbols)] if not visible.empty else visible
    expected = set(symbols)
    complete = set()
    fresh = set()
    if not visible.empty:
        valid = (
            visible[["roe", "q_sales_yoy"]]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .all(axis=1)
        )
        complete = set(visible.loc[valid, "ts_code"])
        # Operational age guard, not a claim that every issuer should report today.
        period_columns = ["roe_end_date", "q_sales_yoy_end_date"] if per_field else ["end_date"]
        current = valid.copy()
        for column in period_columns:
            period = pd.to_datetime(visible[column], format="%Y%m%d", errors="coerce")
            current &= (pd.Timestamp(as_of) - period).dt.days.between(0, 200)
        fresh = set(visible.loc[current, "ts_code"])
    missing = sorted(expected - complete)
    stale = sorted(complete - fresh)
    return {
        "as_of_date": as_of,
        "requested": len(expected),
        "complete": len(complete),
        "fresh_complete": len(fresh),
        "missing_symbols": missing,
        "stale_symbols": stale,
        "coverage": len(complete) / len(expected) if expected else 0,
        "age_guard_days": 200,
        "gate": "PASS" if expected and not missing and not stale else "BLOCK",
        "scope": "RESEARCH_VERSION_ONLY",
        "promotion_allowed": False,
        "feature_rule": "LATEST_VALID_PER_FIELD" if per_field else "LEGACY_LATEST_WHOLE_ROW",
        "missing_field_counts": {
            field: len(expected) - int(pd.to_numeric(visible[field], errors="coerce").notna().sum())
            if field in visible
            else len(expected)
            for field in ("roe", "q_sales_yoy")
        },
    }


def sync_research_financials(
    root: Path,
    store: ParquetStore,
    provider,
    symbols: list[str],
    as_of: str,
    *,
    max_requests: int = 30,
    interval_seconds: float = 1.1,
    checkpoint_name: str = "checkpoint.json",
    flush_every: int = 10,
) -> dict:
    """One request per symbol, no blind retries; checkpoint every response, batch writes."""
    directory = root / "state/fundamental-research-v2"
    directory.mkdir(parents=True, exist_ok=True)
    # Dedicated SQLite lock: OS releases on crash, no stale PID removal or production DB locks.
    with sqlite3.connect(directory / "sync-lock.sqlite3", timeout=0) as lock:
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            return {"status": "ALREADY_RUNNING", "requests": 0}
        if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith(".json"):
            raise ValueError("INVALID_CHECKPOINT_NAME")
        checkpoint = directory / checkpoint_name
        state = (
            json.loads(checkpoint.read_text(encoding="utf-8"))
            if checkpoint.exists()
            else {"symbols": {}}
        )
        last_error_at = state.get("last_error", {}).get("at")
        if (
            last_error_at
            and (datetime.now(UTC) - datetime.fromisoformat(last_error_at)).total_seconds() < 21600
        ):
            return {"status": "ERROR_COOLDOWN", "requests": 0, "orders_enabled": False}
        records = state["symbols"]
        try:
            existing = store.read(TABLE)
        except FileNotFoundError:
            existing = pd.DataFrame(
                columns=["ts_code", "ann_date", "end_date", "roe", "q_sales_yoy"]
            )
        ordered = sorted(
            set(symbols), key=lambda s: (records.get(s, {}).get("checked_date", ""), s)
        )
        pending = [
            s
            for s in ordered
            if (
                pd.Timestamp(as_of)
                - pd.Timestamp(records.get(s, {}).get("checked_date", "19000101"))
            ).days
            >= 7
        ]
        frames = []
        requests = 0
        status = "UP_TO_DATE"

        def flush():
            nonlocal existing
            if frames:
                from .financial_reconciliation import KEYS, reconcile_indicator_payload

                # Preserve raw duplicates separately; only the isolated curated research
                # table receives explicit per-field null/conflict reconciliation.
                store.save_snapshots("fina_indicator_payload_v2", frames, "tushare", KEYS)
                normalized = []
                for key, frame in frames:
                    frame = frame.copy()
                    frame["_payload_capture"] = key.split("_")[-1]
                    normalized.append(reconcile_indicator_payload(frame))
                store.merge_curated(TABLE, normalized, KEYS)
                existing = store.read(TABLE)
                frames.clear()
            # Only persist checked successes after their data is durable.
            write_diagnostic(checkpoint, state)
            result = {
                **financial_quality(
                    research_financial_frame(store) if not existing.empty else existing,
                    symbols,
                    as_of,
                    per_field=True,
                ),
                "status": status,
                "requests_this_run": requests,
                "remaining_due": max(0, len(pending) - requests),
                "captured_at": datetime.now(UTC).isoformat(),
            }
            quality_name = "quality.json" if checkpoint_name == "checkpoint.json" else Path(checkpoint_name).stem + "-quality.json"
            write_diagnostic(directory / quality_name, result)
            return result

        last_request_started = None
        flush_every = max(1, min(int(flush_every), 50))
        for symbol in pending[: max(0, min(max_requests, 1000))]:
            if requests:
                remaining_interval = max(1.1, interval_seconds) - (
                    time.monotonic() - last_request_started
                )
                if remaining_interval > 0:
                    time.sleep(remaining_interval)
            last_request_started = time.monotonic()
            requests += 1
            try:
                fetch = getattr(
                    provider, "financial_indicators_revisioned", provider.financial_indicators
                )
                frame = fetch(symbol, "20150101", as_of)
                required = {"ts_code", "ann_date", "end_date", "roe", "q_sales_yoy"}
                if frame is None or not required.issubset(frame.columns):
                    raise ValueError("Invalid financial schema")
                if not frame.empty:
                    if not frame.ts_code.eq(symbol).all():
                        raise ValueError("Unexpected symbol")
                    future_rows = pd.Series(False, index=frame.index)
                    for column in ("ann_date", "end_date"):
                        parsed = pd.to_datetime(frame[column], format="%Y%m%d", errors="coerce")
                        if parsed.isna().any():
                            raise ValueError("Invalid financial dates")
                        future_rows |= parsed > pd.Timestamp(as_of)
                    if future_rows.any():
                        # A valid later announcement must not abort the issuer's
                        # entire earlier history. Keep it outside the active table.
                        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                        store.save_snapshot(
                            "fina_indicator_future_quarantine_v2",
                            frame.loc[future_rows],
                            "tushare",
                            ["ts_code", "ann_date", "end_date"],
                            symbol.replace(".", "_") + "_" + stamp,
                        )
                        frame = frame.loc[~future_rows].copy()
                    captured = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                    if not frame.empty:
                        frames.append((f"{symbol.replace('.', '_')}_{captured}", frame))
                records[symbol] = {
                    "checked_date": as_of,
                    "rows": len(frame),
                    "result": "OK" if len(frame) else "EMPTY",
                }
                state.pop("last_error", None)
                status = "PROGRESS"
                if requests % flush_every == 0:
                    flush()
            except Exception as error:  # noqa: BLE001 - circuit breaker; never log provider secrets
                message = str(error).lower()
                reason = (
                    "PERMISSION_OR_QUOTA"
                    if any(
                        s in message for s in ("权限", "积分", "频", "limit", "permission", "quota")
                    )
                    else type(error).__name__
                )
                state["last_error"] = {
                    "symbol": symbol,
                    "date": as_of,
                    "reason": reason,
                    "at": datetime.now(UTC).isoformat(),
                }
                status = "PAUSED_ERROR"
                break
        if status == "PROGRESS" and requests >= len(pending):
            status = "CYCLE_COMPLETE"
        state["status"] = status
        return flush()


def build_financial_research_comparison(
    root: Path, store: ParquetStore, symbols: list[str], as_of: str
) -> dict:
    """Research-only scores; never constructs targets or touches recommendation tables."""
    import yaml

    from ..research.score_evidence import archive_score_evidence
    from ..research.sector_model import build_global_factor_snapshot

    try:
        staged = research_financial_frame(store)
    except FileNotFoundError:
        staged = pd.DataFrame(columns=["ts_code", "ann_date", "end_date", "roe", "q_sales_yoy"])
    quality = financial_quality(staged, symbols, as_of, per_field=True)
    result = {
        **quality,
        "version": "global_multifactor_financial_v2",
        "status": "BLOCKED_DATA" if quality["gate"] != "PASS" else "RESEARCH_ONLY",
        "orders_enabled": False,
        "automatic_promotion": False,
    }
    if quality["gate"] == "PASS":

        class StagedStore:
            def read(self, table, **kwargs):
                if table == "fina_indicator":
                    features = financial_features_as_of(staged, as_of)
                    return features[kwargs["columns"]] if kwargs.get("columns") else features
                return store.read(table, **kwargs)

        policy = yaml.safe_load((root / "configs/sector_models.yaml").read_text(encoding="utf-8"))[
            "global_selection"
        ]
        capture = {}
        scores = build_global_factor_snapshot(
            StagedStore(), symbols, policy["factors"], audit_capture=capture
        )
        capture["financial_provenance"] = financial_features_as_of(staged, as_of)
        score_date = pd.Timestamp(scores.date.max()).strftime("%Y%m%d")
        if score_date != as_of:
            result.update(status="BLOCKED_MARKET_DATE", score_date=score_date)
        else:
            scores["raw_rank"] = scores.score.rank(ascending=False, method="first")
            manifest = archive_score_evidence(
                root, capture, scores, policy, version=result["version"]
            )
            result.update(
                score_evidence=manifest,
                scored=len(scores),
                complete_factor_rows=manifest["complete_factor_rows"],
                raw_top20=scores.sort_values("score", ascending=False)
                .head(20)[["symbol", "score", "active_weight"]]
                .to_dict("records"),
            )
            if manifest["data_gate"] != "PASS":
                result["status"] = "BLOCKED_OTHER_FACTORS"
    write_diagnostic(root / "state/fundamental-research-v2/comparison.json", result)
    return result
