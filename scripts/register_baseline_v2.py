"""Activate an independent, forward paper-only factor baseline V2."""

import json
from pathlib import Path

from moneymore.baseline_v2_candidate import register_factor_v2
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = register_factor_v2(
        root,
        ParquetStore(root / "data"),
        PaperBroker(root / "state/paper_orders.sqlite3"),
        BacktestConfig.from_yaml(root / "configs/default.yaml"),
    )
    print(
        json.dumps(
            {
                k: result.get(k) if isinstance(result, dict) else getattr(result, k, None)
                for k in ("trade_date", "status", "report_path")
            },
            ensure_ascii=True,
        )
    )
