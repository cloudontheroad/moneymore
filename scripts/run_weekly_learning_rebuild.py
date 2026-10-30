"""One isolated local research/refit batch, no trading-account registration."""
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "state/weekly-learning/20261030-learning-batch-a"


def main():
    RUN.mkdir(exist_ok=False)
    status = {"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat(),
              "status": "RUNNING_RESEARCH", "orders_enabled": False}
    def save():
        (RUN / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    save()
    try:
        with (RUN / "training.log").open("a", encoding="utf-8") as log:
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                   "MONEYMORE_CANDIDATE_TAG": "20261030", "MONEYMORE_ISOLATED_RESEARCH": "1"}
            subprocess.run([sys.executable, str(ROOT / "scripts/research_qlib_challenger.py")],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=21600)
            status["status"] = "RUNNING_LATEST_REFIT"
            save()
            research = ROOT / "state/weekly-learning/isolated-research/candidates/20261030"
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
