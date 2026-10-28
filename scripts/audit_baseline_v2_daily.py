"""Read-only full daily-universe audit; requires completed exact-session inputs."""

import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_audit import audit_daily_inputs

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(audit_daily_inputs(root, ParquetStore(root / "data"))))
