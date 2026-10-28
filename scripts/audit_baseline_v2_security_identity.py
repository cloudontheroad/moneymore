"""Audit alias overlaps and preview identity-safe research inputs; no production writes."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_readiness import validate_snapshot
from moneymore.research.security_identity import SecurityIdentity

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    config = root / "configs/baseline_v2_market_evidence.yaml"
    identity = SecurityIdentity(
        yaml.safe_load(config.read_text(encoding="utf-8"))["symbol_changes"]
    )
    symbols = sorted({r[k] for r in identity.changes for k in ["old", "new"]})
    store = ParquetStore(root / "data")
    directory = base / "identity-audits" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    calendar = trading_dates(store, "20261028")
    scope_start = calendar[calendar.index("20240131") - 252]
    table_reports, conflicts_total, blocking_conflicts = {}, 0, 0
    for table, keys, fields in [
        ("daily", ["trade_date"], ["open", "high", "low", "close"]),
        ("adj_factor", ["trade_date"], ["adj_factor"]),
        ("fina_indicator_research_v2", ["ann_date", "end_date"], ["roe", "q_sales_yoy"]),
    ]:
        frame = store.read(table, filters=[("ts_code", "in", symbols)])
        conflicts = identity.conflicts(frame, keys, fields)
        conflicts.to_parquet(directory / (table + "-conflicts.parquet"), index=False)
        conflicts_total += len(conflicts)
        relevant = conflicts.loc[conflicts[keys[0]].between(scope_start, "20251230")]
        blocking_conflicts += len(relevant)
        # Preview only: any source disagreement blocks downstream release of this run.
        normalized = identity.market_rows(frame, keys[0], fields, strict=False, record_keys=keys)
        normalized.to_parquet(directory / (table + "-identity-preview.parquet"), index=False)
        table_reports[table] = {
            "source_rows": len(frame),
            "preview_rows": len(normalized),
            "conflicts": len(conflicts),
            "scope_conflicts": len(relevant),
            "source_logical_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).values.tobytes()
            ).hexdigest(),
        }
    industry = store.read("baseline_v2_industry_membership", filters=[("ts_code", "in", symbols)])
    identity.industry_rows(industry).to_parquet(
        directory / "industry-identity-preview.parquet", index=False
    )
    instruments = store.read("instruments", filters=[("ts_code", "in", symbols)])
    instruments["security_id"] = instruments.ts_code.map(identity.stable)
    instruments.to_parquet(directory / "instruments-identity-preview.parquet", index=False)
    ready = json.loads((base / "readiness.json").read_text(encoding="utf-8"))
    basic = store.read("baseline_v2_daily_basic")
    groups = {str(d): f for d, f in basic.groupby("trade_date")}
    daily, pool = [], []
    for row in ready["schedule"]:
        source = row["source_date"]
        frame = groups.get(source)
        if frame is None:
            path = base / (source + ".parquet")
            marker = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if marker["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("SNAPSHOT_HASH_MISMATCH")
            frame = pd.read_parquet(path)
        old_top = validate_snapshot(frame, source)
        duplicate_slots = 1000 - old_top.ts_code.map(identity.stable).nunique()
        cols = ["close", "pb", "pe_ttm", "dv_ttm", "total_mv"]
        c = identity.conflicts(frame, ["trade_date"], cols)
        conflicts_total += len(c)
        blocking_conflicts += len(c)
        if len(c):
            c.to_parquet(directory / (source + "-basic-conflicts.parquet"), index=False)
        normalized = identity.market_rows(frame, "trade_date", cols, strict=False)
        new_top = validate_snapshot(normalized, source)
        new_top["signal_date"] = row["signal_date"]
        pool.append(new_top)
        daily.append(
            {
                "signal_date": row["signal_date"],
                "source_date": source,
                "duplicate_original_slots": int(duplicate_slots),
                "snapshot_value_conflicts": len(c),
                "changed_members": len(
                    set(old_top.ts_code.map(identity.stable)) ^ set(new_top.security_id)
                ),
                "unique_corrected_identities": new_top.security_id.nunique(),
            }
        )
    pd.DataFrame(daily).to_parquet(directory / "daily-universe-comparison.parquet", index=False)
    pd.concat(pool, ignore_index=True).to_parquet(
        directory / "identity-top1000-preview.parquet", index=False
    )
    result = {
        "status": "BLOCKED_ALIAS_SOURCE_CONFLICT"
        if blocking_conflicts
        else "IDENTITY_PREVIEW_AUDITED_NOT_RELEASED",
        "output_directory": str(directory),
        "source_value_conflicts": conflicts_total,
        "blocking_scope_conflicts": blocking_conflicts,
        "scope_start": scope_start,
        "scope_end": "20251230",
        "tables": table_reports,
        "signals": len(daily),
        "days_with_duplicate_original_slots": sum(r["duplicate_original_slots"] > 0 for r in daily),
        "corrected_preview_unique_rows": sum(r["unique_corrected_identities"] for r in daily),
        "evidence_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "real_orders_enabled": False,
        "production_changed": False,
        "limitations": [
            "PREVIEW_ONLY_NOT_CONNECTED_TO_SCORING_OR_EXECUTION",
            "ORIGINAL_FINANCIAL_VINTAGES_UNPROVEN",
        ],
    }
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(base / "identity-audit.json", result)
    print(json.dumps(result))
