"""Preview (default) or apply candidate governance, never submit/fill trades."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from moneymore.candidate_lifecycle import (
    FROZEN,
    lifecycle_rows,
    review_candidate,
    run_lifecycle_review,
)
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.qlib_candidate_observer import candidate_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    policy = yaml.safe_load((root / "configs/qlib_challenger.yaml").read_text(encoding="utf-8"))[
        "candidate_observation_policy"
    ]
    store = ParquetStore(root / "data")
    candidates = candidate_catalog(root)
    if args.apply:
        rows = run_lifecycle_review(root, store, candidates, policy, today)
        broker = PaperBroker(root / "state/paper_orders.sqlite3")
        cancelled = sum(
            broker.cancel_pending(f"qlib_candidate_{tag}", "CANDIDATE_LIFECYCLE_FROZEN")
            for tag, row in rows.items()
            if row.get("status") in FROZEN
        )
    else:
        history = store.read("qlib_candidate_account_daily")
        baseline = store.read("multi_sector_account_daily")
        challenger = store.read("qlib_challenger_account_daily")
        boundary = json.loads(
            (root / "state/point-in-time/latest.json").read_text(encoding="utf-8")
        )
        trust = boundary.get("historical_replay_gate", {}).get("earliest_trustworthy_date")
        previous = lifecycle_rows(root)
        rows = {
            row["candidate_tag"]: review_candidate(
                row,
                {
                    "baseline": baseline,
                    "challenger": challenger,
                    "candidate": history.loc[history.account_id.astype(str) == row["account_id"]],
                },
                as_of=today,
                trustworthy_start=trust,
                policy=policy,
                previous=previous.get(row["candidate_tag"]),
                trial_count=len(candidates),
            )
            for row in candidates
            if row["candidate_tag"] <= today
        }
        cancelled = 0
    print(
        json.dumps(
            {
                "applied": args.apply,
                "cancelled_pending_paper_orders": cancelled,
                "candidates": list(rows.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
