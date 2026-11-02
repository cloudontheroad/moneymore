"""Sequential isolated target/weight ablation; does not change live accounts."""
import json
import os
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
import yaml
from moneymore.training_artifact_checks import validate_research_checkpoint

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "state/weekly-learning/objective-blend-20261102"


def main():
    RUN.mkdir(exist_ok=False)
    config = yaml.safe_load((ROOT / "configs/qlib_challenger.yaml").read_text(encoding="utf-8"))
    protocols = ["sector_total_return", "absolute_total_return", "risk_adjusted_total_return"]
    status = {"pid": os.getpid(), "status": "RUNNING", "orders_enabled": False, "completed": []}
    def save():
        status["updated_at"] = datetime.now(UTC).isoformat()
        (RUN / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    save()
    try:
        for protocol in protocols:
            current = {**config, "label_protocol": protocol, "learn_blend_weights": True,
                "model_id": f"qlib_gru_{protocol}_v3"}
            path = RUN / f"{protocol}.yaml"
            path.write_text(yaml.safe_dump(current, allow_unicode=True), encoding="utf-8")
            tag = f"20261102_{protocol}"
            status["current"] = protocol
            save()
            with (RUN / f"{protocol}.log").open("a", encoding="utf-8") as log:
                env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                    "MONEYMORE_CANDIDATE_TAG": tag, "MONEYMORE_ISOLATED_RESEARCH": "1", "MONEYMORE_RESEARCH_CONFIG": str(path)}
                result = subprocess.run([sys.executable, str(ROOT / "scripts/research_qlib_challenger.py")],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=21600)
            if result.returncode not in (0, 3221226505, -1073740791):
                raise RuntimeError(f"research failed: {protocol}, exit={result.returncode}")
            folder = ROOT / "state/weekly-learning/isolated-research/candidates" / tag
            checkpoint = validate_research_checkpoint(folder)
            status["completed"].append({"protocol": protocol, "folder": str(folder), "checkpoint": checkpoint})
            save()
        status["status"] = "COMPLETED_RESEARCH_NOT_RELEASED"
    except Exception as exc:
        status.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
