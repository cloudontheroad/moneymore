"""Run bounded offline release evidence; never enables a candidate or live orders."""

import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_release import (
    freeze_release,
    publish_status,
    run_attribution,
    run_cost_screen,
    run_financial_inventory,
    run_robustness,
)
from moneymore.research.baseline_v2_release_runtime import run_runtime_audit
from moneymore.research.baseline_v2_release_style import run_release_style
from moneymore.research.baseline_v2_release_stress import collect_execution_stress
from moneymore.research.baseline_v2_source_economics import scan_union

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    manifest = freeze_release(root)
    store = ParquetStore(root / "data")
    for name, task in [
        ("attribution", lambda: run_attribution(root, store, manifest)),
        ("financial", lambda: run_financial_inventory(root, store, manifest)),
        ("robustness", lambda: run_robustness(manifest)),
        ("cost_screen", lambda: run_cost_screen(manifest)),
        ("union_economics", lambda: scan_union(root, store, manifest)),
        ("style", lambda: run_release_style(manifest)),
        ("runtime", lambda: run_runtime_audit(root, manifest)),
        ("execution_stress", lambda: collect_execution_stress(root, manifest)),
    ]:
        result = task()
        print(json.dumps({"stage": name, "status": result["status"]}), flush=True)
    print(json.dumps(publish_status(manifest)))
