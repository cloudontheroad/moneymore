"""Repair the full candidate pool in an isolated table; never promotes a strategy."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.fundamental_research import (
    build_financial_research_comparison,
    sync_research_financials,
)
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider
from moneymore.strategy_universe import active_strategy_universe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--max-requests", type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    store = ParquetStore(root / "data")
    result = sync_research_financials(
        root,
        store,
        TushareProvider(),
        active_strategy_universe(store).symbol.tolist(),
        args.as_of,
        max_requests=args.max_requests,
    )
    if result.get("status") != "ALREADY_RUNNING":
        comparison = build_financial_research_comparison(
            root, store, active_strategy_universe(store).symbol.tolist(), args.as_of
        )
        result["comparison_status"] = comparison["status"]
    print(
        json.dumps(
            {k: v for k, v in result.items() if not k.endswith("_symbols")}, ensure_ascii=False
        )
    )
