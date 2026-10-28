"""Frozen fixed/learned daily ranking diagnostic; no fitting or orders."""

import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_scores import run_daily_scores

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_daily_scores(root, ParquetStore(root / "data"))
    print(json.dumps(result))
