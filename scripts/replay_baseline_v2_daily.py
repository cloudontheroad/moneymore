"""Evidence-bounded formal daily PaperBroker replay. No production ledger access."""

import argparse
import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_replay import STRESS_SCENARIOS, run_daily_replay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="20240208")
    parser.add_argument("--scenario", choices=list(STRESS_SCENARIOS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            run_daily_replay(root, ParquetStore(root / "data"), args.end, scenario=args.scenario)
        )
    )
