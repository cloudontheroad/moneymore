"""Finite local evidence build + bounded serial source verification, not a scheduler.

Never submits orders, changes production prices, approves recovery, or promotes V2.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.alphafeed_provider import AlphaFeedProvider
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_release import read_json
from moneymore.research.baseline_v2_union_verification import verify_union_batch

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    # Code has stabilized before freezing. If any bound code/data changes while
    # this runs, verify_manifest aborts instead of carrying stale evidence forward.
    subprocess.run(
        [sys.executable, str(root / "scripts/prepare_baseline_v2_release.py")],
        cwd=root,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        check=True,
    )
    directory = Path(read_json(root / "state/baseline-v2-release/latest.json")["directory"])
    manifest = read_json(directory / "manifest.json")
    load_dotenv(root / ".env", override=False)
    client = AlphaFeedProvider(root=root)._client
    store = ParquetStore(root / "data")
    # At most 1,024 HTTP requests total, one at a time with >=1.3s interval.
    # A single provider failure ends the whole finite run; there is no retry loop.
    for batch in range(64):
        result = verify_union_batch(root, store, client, manifest, max_pairs=64)
        print(json.dumps({"batch": batch + 1, **result}), flush=True)
        if (
            result["status"] != "BOUNDED_INDEPENDENT_BATCH_COMPLETE"
            or not result.get("remaining_pairs", 0)
            or not result.get("new_pairs", 0)
        ):
            break
