"""Bounded research-only histories for new daily Top1000 union members."""

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from moneymore.data.fundamental_research import sync_research_financials
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-requests", type=int, default=60)
    parser.add_argument(
        "--period-recheck",
        action="store_true",
        help="Recheck remaining historical holes using explicit revision metadata",
    )
    args = parser.parse_args()
    if not 0 < args.max_requests <= 100:
        parser.error("request budget must be 1..100")
    root = Path(__file__).resolve().parents[1]
    readiness = json.loads(
        (root / "state/baseline-v2-validation/daily-inputs/readiness.json").read_text(
            encoding="utf-8"
        )
    )
    if readiness["missing"] or not readiness["union_complete"]:
        raise ValueError("COMPLETE_DAILY_UNION_REQUIRED")
    store = ParquetStore(root / "data")
    known = set(store.read("fina_indicator_research_v2", columns=["ts_code"]).ts_code)
    symbols = sorted(set(readiness["observed_union_missing_research_financials"]) - known)
    checkpoint = "daily-union-checkpoint.json"
    if args.period_recheck:
        audit = json.loads(
            (root / "state/baseline-v2-validation/daily-inputs/coverage-audit.json").read_text(
                encoding="utf-8"
            )
        )
        detail = pd.read_parquet(Path(audit["output_directory"]) / "sample-exclusions.parquet")
        symbols = sorted(
            detail.loc[detail.reason.eq("REQUIRED_FINANCIAL_MISSING"), "symbol"].unique()
        )
        checkpoint = "daily-period-recheck.json"
    if not symbols:
        print(json.dumps({"status": "NO_ABSENT_SYMBOL_HISTORIES", "requests": 0}))
    else:
        load_dotenv(root / ".env", override=False)
        result = sync_research_financials(
            root,
            store,
            TushareProvider(),
            symbols,
            "20261027",
            max_requests=args.max_requests,
            interval_seconds=1.3,
            checkpoint_name=checkpoint,
            flush_every=20,
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if not isinstance(v, (dict, list, pd.DataFrame))}
            )
        )
