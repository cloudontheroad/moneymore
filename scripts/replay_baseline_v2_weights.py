"""Offline ledger-derived monthly fixed-versus-learned weight comparison."""

import argparse
import json
from pathlib import Path

from moneymore.research.baseline_v2_continuous_replay import audit_saved_replay, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.audit_only:
        print(json.dumps(audit_saved_replay(root)))
    else:
        result = run(root)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "pairs": result["pairs"],
                    "output_directory": result["output_directory"],
                }
            )
        )
