import json
from pathlib import Path

from moneymore.research.baseline_v2_release import publish_status, read_json
from moneymore.research.baseline_v2_release_runtime import run_runtime_audit

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    directory = Path(read_json(root / "state/baseline-v2-release/latest.json")["directory"])
    manifest = read_json(directory / "manifest.json")
    print(json.dumps(run_runtime_audit(root, manifest)))
    print(json.dumps(publish_status(manifest)))
