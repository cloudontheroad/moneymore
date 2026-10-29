"""Remove the explicitly retired, empty 20261028 factor prototype only."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic


def main():
    root = Path(__file__).resolve().parents[1]
    state = (root / "state").resolve()
    tag, account = "20261028", "qlib_candidate_20261028"
    registry = state / "factor-baseline-candidates" / (tag + ".json")
    tombstone = state / "removed-factor-candidates" / (tag + ".json")
    previous = json.loads(tombstone.read_text(encoding="utf-8")) if tombstone.exists() else None
    registry_source = registry if registry.exists() else Path(previous["backup"]) / "factor-baseline-candidates" / (tag + ".json")
    candidate = json.loads(registry_source.read_text(encoding="utf-8"))
    if candidate["account_id"] != account or not candidate.get("validation_hold"):
        raise ValueError("ONLY_THE_PAUSED_EMPTY_PROTOTYPE_MAY_BE_REMOVED")
    for directory in ("qlib-portfolio-candidates", "factor-baseline-candidates"):
        for path in (state / directory).glob("*.json"):
            if path != registry and json.loads(path.read_text(encoding="utf-8")).get("source_candidate_tag") == tag:
                raise ValueError("OTHER_CANDIDATE_DEPENDS_ON_PROTOTYPE")
    database = state / "paper_orders.sqlite3"
    with sqlite3.connect(database, timeout=10) as connection:
        for table in ("positions", "fills", "orders"):
            if connection.execute(f"SELECT count(*) FROM {table} WHERE account_id=?", (account,)).fetchone()[0]:
                raise ValueError("PROTOTYPE_IS_NOT_EMPTY:" + table)
    backup = state / "removed-candidate-backups" / (tag + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    backup.mkdir(parents=True, exist_ok=False)
    paths = [registry, state / "qlib-candidate-shadow" / tag, state / "qlib-portfolio-plans" / tag,
             state / "qlib-portfolio-execution" / tag, state / "qlib-portfolio-budgets" / (tag + ".json"),
             state / "factor-baseline-bootstrap" / (tag + ".json"),
             state / "qlib-candidate-forward" / (tag + ".json"),
             state / "qlib-candidate-forward" / (tag + "-history")]
    moved = []
    for path in paths:
        resolved = path.resolve()
        destination = backup / path.relative_to(state)
        if not resolved.is_relative_to(state) or resolved == state or not destination.resolve().is_relative_to(backup):
            raise ValueError("REMOVAL_TARGET_OUTSIDE_EXPLICIT_STATE_SCOPE")
        if path.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)
            moved.append(str(path.relative_to(state)))
    deleted_rows = {}
    for db, key, value in [(database, "account_id", account), (state / "candidate-lifecycle.sqlite3", "candidate_tag", tag)]:
        if not db.exists():
            continue
        with sqlite3.connect(db, timeout=10) as connection:
            connection.execute("BEGIN IMMEDIATE")
            tables = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            snapshots = {}
            for table in tables:
                if not table.replace("_", "").isalnum():
                    continue
                columns = [r[1] for r in connection.execute(f'PRAGMA table_info("{table}")')]
                if key not in columns:
                    continue
                rows = connection.execute(f'SELECT * FROM "{table}" WHERE "{key}"=?', (value,)).fetchall()
                if rows:
                    snapshots[table] = [dict(zip(columns, row, strict=True)) for row in rows]
            write_diagnostic(backup / (db.stem + "-rows.json"), snapshots)
            for table in snapshots:
                count = connection.execute(f'DELETE FROM "{table}" WHERE "{key}"=?', (value,)).rowcount
                deleted_rows[db.name + ":" + table] = count
    removed_history = {}
    for path in (root / "data/processed").glob("*.parquet"):
        import pyarrow.parquet as pq

        if "account_id" not in pq.read_schema(path).names:
            continue
        frame = pd.read_parquet(path)
        mask = frame.account_id.eq(account)
        if not mask.any():
            continue
        frame.loc[mask].to_parquet(backup / (path.stem + "-removed-rows.parquet"), index=False)
        temporary = path.with_suffix(".candidate-cleanup.tmp")
        frame.loc[~mask].to_parquet(temporary, index=False)
        temporary.replace(path)
        removed_history[path.name] = int(mask.sum())
    def without_removed(value):
        if isinstance(value, list):
            return [without_removed(v) for v in value if not (isinstance(v, dict) and v.get("account_id") == account)]
        if isinstance(value, dict):
            return {k: without_removed(v) for k, v in value.items() if k != account}
        return value
    for name in ("latest.json", "forward-summary.json"):
        path = state / "challenger-diagnostics" / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            write_diagnostic(backup / ("diagnostics-" + name), payload)
            write_diagnostic(path, without_removed(payload))
    report = {"status": "REMOVED_FROM_SYSTEM", "candidate_tag": tag, "account_id": account,
              "removed_at": datetime.now(UTC).isoformat(), "reason": "USER_REQUESTED_OBSOLETE_PAUSED_PROTOTYPE_REMOVAL",
              "backup": str(backup), "previous_backup": previous.get("backup") if previous else None, "moved_paths": moved, "deleted_rows": deleted_rows,
              "removed_history_rows": removed_history, "new_v2_20261029_unchanged": True}
    write_diagnostic(state / "removed-factor-candidates" / (tag + ".json"), report)
    write_diagnostic(backup / "removal.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
