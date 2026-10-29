"""Whole union scan plus optional bounded official dividend acquisition."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_release import publish_status, read_json
from moneymore.research.baseline_v2_source_economics import scan_union, sync_union_dividends

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = Path(read_json(root / "state/baseline-v2-release/latest.json")["directory"])
    manifest = read_json(directory / "manifest.json")
    result = scan_union(root, ParquetStore(root / "data"), manifest)
    print(json.dumps(result), flush=True)
    if args.sync:
        from moneymore.data.tushare_provider import TushareProvider

        load_dotenv(root / ".env", override=False)
        print(json.dumps(sync_union_dividends(root, TushareProvider(), manifest)), flush=True)
    print(json.dumps(publish_status(manifest)))
