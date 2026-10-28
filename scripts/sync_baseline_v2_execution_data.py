"""Resume bounded official execution evidence collection; never writes live data."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.baseline_v2_execution_data import sync_execution_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-requests", type=int, default=25)
    parser.add_argument("--adjustments", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    if args.adjustments:
        from moneymore.research.baseline_v2_execution_data import sync_unexplained_adjustments

        try:
            print(
                json.dumps(
                    sync_unexplained_adjustments(
                        root, TushareProvider(), max_requests=args.max_requests
                    )
                )
            )
        except Exception as error:  # noqa: BLE001 - do not expose provider credentials
            print(
                json.dumps({"status": "STOPPED_ERROR_NO_RETRY", "error_type": type(error).__name__})
            )
    else:
        print(
            json.dumps(sync_execution_data(root, TushareProvider(), max_requests=args.max_requests))
        )
