"""Finite offline 5/10-bps full replays; no scheduler, network or real orders."""

import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_replay import run_daily_replay

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for scenario in ("slippage_5bps", "slippage_10bps"):
        result = run_daily_replay(root, ParquetStore(root / "data"), "20251231", scenario=scenario)
        print(json.dumps(result), flush=True)
