"""One isolated local research/refit batch, no trading-account registration."""
import json
import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from moneymore.training_artifact_checks import validate_research_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "state/weekly-learning/20261030-learning-batch-a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(exist_ok=args.resume)
    if args.resume and (RUN / "status.json").exists():
        previous = (RUN / "status.json").read_text(encoding="utf-8")
        (RUN / f'previous-status-{datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")}.json').write_text(previous, encoding="utf-8")
    status = {"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat(),
              "status": "RUNNING_RESEARCH", "orders_enabled": False}
    def save():
        (RUN / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    save()
    try:
        with (RUN / "training.log").open("a", encoding="utf-8") as log:
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                   "MONEYMORE_CANDIDATE_TAG": "20261030", "MONEYMORE_ISOLATED_RESEARCH": "1"}
            research = ROOT / "state/weekly-learning/isolated-research/candidates/20261030"
            if not args.resume:
                completed = subprocess.run([sys.executable, str(ROOT / "scripts/research_qlib_challenger.py")],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=21600)
                if completed.returncode not in (0, 3221226505, -1073740791):
                    raise RuntimeError(f"research failed: {completed.returncode}")
                status["research_exit_code"] = completed.returncode
            status["research_checkpoint"] = validate_research_checkpoint(research)
            log.write("Validated durable research checkpoint; continuing latest refit without repeating research.\n")
            log.flush()
            status["status"] = "RUNNING_LATEST_REFIT"
            save()
            subprocess.run([sys.executable, str(ROOT / "scripts/refit_latest_challenger.py"),
                "--research-dir", str(research), "--as-of", "20261029", "--output", str(RUN / "refit"),
                "--suspension-proof", str(ROOT / "state/weekly-learning/20261030-missing-evidence/suspension.parquet"),
                "--execute"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=21600)
        status["status"] = "COMPLETED_ISOLATED_NOT_RELEASED"
    except Exception as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["updated_at"] = datetime.now(UTC).isoformat()
        save()


if __name__ == "__main__":
    main()
