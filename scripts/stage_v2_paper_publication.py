"""Local fail-closed publication preflight. Default mode never creates accounts.

--stage prepares an eligible isolated account with DISABLED task intents only;
this is not activation. --rollback pauses a previously prepared version.
No real broker, credentials, desktop client or legacy account is accessed.
"""

import argparse
import json
from pathlib import Path

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest, publish_status, read_json
from moneymore.research.paper_publication import PaperPublication

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--stage", action="store_true")
    actions.add_argument("--rollback")
    parser.add_argument("--reason")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    publisher = PaperPublication(root)
    if args.rollback:
        if not args.reason:
            parser.error("--rollback requires --reason")
        print(json.dumps(publisher.rollback(args.rollback, reason=args.reason)))
    else:
        pointer = read_json(root / "state/baseline-v2-release/latest.json")
        directory = Path(pointer["directory"]).resolve()
        if not directory.is_relative_to(root / "state/baseline-v2-release/runs"):
            raise ValueError("PUBLICATION_BUNDLE_OUTSIDE_RELEASE_ROOT")
        if digest(directory / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("PUBLICATION_MANIFEST_POINTER_CHANGED")
        status = publish_status(read_json(directory / "manifest.json"))
        result = {
            "status": "BLOCKED_NOT_PUBLISHED",
            "blocked_stages": status["observation_blocked_stages"],
            "account_created": False,
            "published": False,
            "real_orders_enabled": False,
        }
        if status["observation_activation_authorized"]:
            path = directory / "provisional-paper-contract.json"
            contract = read_json(path)
            request = {
                "release_directory": str(directory),
                "contract_sha256": digest(path),
                "account_id": contract.get("account_id", "MISSING_FROZEN_ACCOUNT"),
            }
            publisher.preflight(request)
            result = (
                publisher.prepare(request)
                if args.stage
                else {**result, "status": "ELIGIBLE_FOR_STAGING_NOT_ACTIVE", "blocked_stages": []}
            )
        elif args.stage:
            result["stage_request_rejected"] = True
        write_diagnostic(root / "state/baseline-v2-release/publication-preflight.json", result)
        print(json.dumps(result))
