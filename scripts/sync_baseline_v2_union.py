"""Bounded resumable local research job; not a Codex automation or trading worker."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.baseline_v2_release import read_json
from moneymore.research.baseline_v2_source_economics import sync_union_dividends

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = Path(read_json(root / "state/baseline-v2-release/latest.json")["directory"])
    manifest = read_json(directory / "manifest.json")
    load_dotenv(root / ".env", override=False)
    provider = TushareProvider()
    for batch in range(min(60, max(0, args.batches))):
        result = sync_union_dividends(root, provider, manifest)
        print(json.dumps({"batch": batch + 1, **result}), flush=True)
        if result["status"] != "BOUNDED_BATCH_COMPLETE" or result.get("pending_after", 0) == 0:
            break
