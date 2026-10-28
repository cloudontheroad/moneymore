"""Run isolated baseline scoring ablations; never submits orders."""

import argparse
import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2 import run_baseline_v2_research
from moneymore.strategy_universe import active_strategy_universe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    report = run_baseline_v2_research(
        root, store, active_strategy_universe(store).symbol.tolist(), args.as_of
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "as_of_date",
                    "status",
                    "requested",
                    "scored",
                    "promotion_gate",
                    "evidence_path",
                    "input_semantics",
                    "variants",
                )
            },
            ensure_ascii=False,
            default=str,
        )
    )
