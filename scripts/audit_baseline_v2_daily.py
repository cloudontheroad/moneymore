"""Read-only full daily-universe audit; requires completed exact-session inputs."""

import argparse
import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_audit import audit_daily_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    identity = None
    if args.identity:
        import yaml

        from moneymore.research.security_identity import SecurityIdentity

        identity = SecurityIdentity(
            yaml.safe_load(
                (root / "configs/baseline_v2_market_evidence.yaml").read_text(encoding="utf-8")
            )["symbol_changes"]
        )
    print(json.dumps(audit_daily_inputs(root, ParquetStore(root / "data"), identity=identity)))
