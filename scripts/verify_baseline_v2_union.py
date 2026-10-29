"""At most eight independent event pairs; no production writes or orders."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.alphafeed_provider import AlphaFeedProvider
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_release import read_json
from moneymore.research.baseline_v2_union_verification import verify_union_batch
from moneymore.research.release_evidence import union_evidence_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--pairs-per-batch", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = Path(read_json(root / "state/baseline-v2-release/latest.json")["directory"])
    load_dotenv(root / ".env", override=False)
    client = AlphaFeedProvider(root=root)._client
    manifest = union_evidence_manifest(root, directory)
    store = ParquetStore(root / "data")
    for batch in range(min(64, max(0, args.batches))):
        result = verify_union_batch(root, store, client, manifest, max_pairs=args.pairs_per_batch)
        print(json.dumps({"batch": batch + 1, **result}), flush=True)
        if (
            result["status"] != "BOUNDED_INDEPENDENT_BATCH_COMPLETE"
            or not result.get("remaining_pairs", 0)
            or not result.get("new_pairs", 0)
        ):
            break
