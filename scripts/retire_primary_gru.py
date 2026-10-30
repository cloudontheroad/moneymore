"""Explicit user-authorized paper retirement; preserve positions and ledger."""
import json
from pathlib import Path
from moneymore.execution.paper import PaperBroker
from moneymore.qlib_governance import deployment_snapshot, transition_release

root = Path(__file__).resolve().parents[1]
current = deployment_snapshot(root)
if current["lifecycle"] != "RETIRED":
    transition_release(root, release_id=current["release_id"], lifecycle="RETIRED",
        execution_mode="OBSERVATION_ONLY", reason="USER_RETIRED_PRIMARY_GRU_20261030")
cancelled = PaperBroker(root / "state/paper_orders.sqlite3").cancel_pending(
    "qlib_gru_shadow", "USER_RETIRED_PRIMARY_GRU_20261030")
print(json.dumps({"lifecycle": "RETIRED", "cancelled_pending": cancelled, "positions_liquidated": False}))
