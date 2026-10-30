from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[1]


def _validate_candidate(candidate_dir: Path) -> tuple[bool, str]:
    research_path = candidate_dir / "research.json"
    marker_path = candidate_dir / "_RESEARCH_COMPLETE"
    model_paths = sorted((candidate_dir / "models").glob("*_seed*.pkl"))
    if not marker_path.exists() or not research_path.exists():
        return False, "completion marker or research.json is missing"
    try:
        payload = json.loads(research_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"research.json is unreadable: {exc}"
    if not payload.get("metrics") or not payload.get("stability"):
        return False, "research.json has no metrics or stability result"
    if len(model_paths) < 5:
        return False, f"only {len(model_paths)} seed models were written"
    try:
        for model_path in model_paths[:5]:
            with model_path.open("rb") as stream:
                pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        return False, f"model artifact is unreadable: {exc}"
    return True, "validated research report and five seed models"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.database) as connection:
        schedule_row = connection.execute(
            "SELECT schedule_key FROM weekly_training_runs WHERE id = ?",
            (args.run_id,),
        ).fetchone()
    schedule_key = str(schedule_row[0]) if schedule_row else "unknown"
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(SHANGHAI).isoformat()
    with sqlite3.connect(args.database) as connection:
        connection.execute(
            "UPDATE weekly_training_runs SET status='RUNNING', pid=?, started_at=? WHERE id=?",
            (os.getpid(), started_at, args.run_id),
        )
    exit_code = 1
    error = None
    try:
        with args.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{started_at}] weekly candidate training started\n")
            log.flush()
            # MiniQMT external connectivity is retired. Data quality is audited
            # separately; training must never initiate a legacy client session.
            candidate_tag = schedule_key.replace("-", "")
            completion_marker = (
                ROOT
                / "state"
                / "qlib-challenger"
                / "candidates"
                / candidate_tag
                / "_TRAINING_COMPLETE"
            )
            completion_marker.unlink(missing_ok=True)
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "research_qlib_challenger.py")],
                cwd=ROOT,
                env={
                    **os.environ,
                    "MONEYMORE_CANDIDATE_TAG": candidate_tag,
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            exit_code = completed.returncode
            artifacts_valid, validation_message = _validate_candidate(
                completion_marker.parent
            )
            if (
                exit_code == 3221226505
                and artifacts_valid
            ):
                log.write(
                    "Windows PyTorch shutdown returned 3221226505 after durable "
                    "research completion; accepting validated artifacts.\n"
                )
                exit_code = 0
            if exit_code == 0 and not artifacts_valid:
                exit_code = 1
                error = f"candidate validation failed: {validation_message}"
            if exit_code:
                error = error or f"research process exited with code {exit_code}"
            else:
                log.write(f"Candidate validation passed: {validation_message}.\n")
                completion_marker.write_text(
                    f"schedule_key={schedule_key}\n"
                    f"run_id={args.run_id}\n"
                    f"completed_at={datetime.now(SHANGHAI).isoformat()}\n",
                    encoding="utf-8",
                )
    except Exception as exc:  # noqa: BLE001 - durable job boundary
        error = f"{type(exc).__name__}: {exc}"
    finished_at = datetime.now(SHANGHAI).isoformat()
    status = "COMPLETED" if exit_code == 0 and error is None else "FAILED"
    with sqlite3.connect(args.database) as connection:
        connection.execute(
            """
            UPDATE weekly_training_runs
            SET status=?, finished_at=?, exit_code=?, error=? WHERE id=?
            """,
            (status, finished_at, exit_code, error, args.run_id),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
