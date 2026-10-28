"""Bounded independent review of frozen unexplained adjustment steps."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.alphafeed_provider import AlphaFeedProvider
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.baseline_v2_adjustment_verification import verify_adjustments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-requests", type=int, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    print(
        json.dumps(
            verify_adjustments(
                root,
                ParquetStore(root / "data"),
                TushareProvider(),
                AlphaFeedProvider(root=root)._client,
                max_requests=args.max_requests,
            )
        )
    )
