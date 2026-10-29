"""Explicit policy migration; preserve old PIT-blocked bundle and all failures.

Existing unchanged-input analyses are carried forward with verified lineage,
not represented as freshly rerun research. Only the new vendor controls stage
uses the revised financial acceptance policy. Does not create/activate accounts.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import (
    digest,
    publish_status,
    read_json,
    verify_manifest,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-release"
    pointer = read_json(base / "latest.json")
    parent = Path(pointer["directory"]).resolve()
    if (
        not parent.is_relative_to(base / "runs")
        or digest(parent / "manifest.json") != pointer["manifest_sha256"]
    ):
        raise ValueError("PARENT_RELEASE_POINTER_INVALID")
    old = read_json(parent / "manifest.json")
    if old["protocol"].get("version") == "baseline_v2_release_vendor_controls_v3_20261029":
        if not args.refresh:
            raise ValueError("VENDOR_POLICY_ALREADY_MIGRATED_EXPLICIT_REFRESH_REQUIRED")
        prior = Path(read_json(parent / "attribution.json")["carried_from_stage"]).resolve().parent
        if (
            not prior.is_relative_to(base / "runs")
            or digest(prior / "manifest.json") != old["parent_manifest_sha256"]
        ):
            raise ValueError("ORIGINAL_PARENT_LINEAGE_CHANGED")
        parent, old = prior, read_json(prior / "manifest.json")
    verify_manifest(old)
    policy = root / "configs/baseline_v2_release_vendor.yaml"
    controls_path = base / "vendor-financial-controls.json"
    controls = read_json(controls_path)
    for path, checksum in controls["artifact_hashes"].items():
        if digest(path) != checksum:
            raise ValueError("VENDOR_CONTROL_EVIDENCE_CHANGED")
    if controls["historical_point_in_time_attested"] is not False:
        raise ValueError("VENDOR_CONTROLS_MUST_NOT_CLAIM_ORIGINAL_PIT")
    directory = (
        base
        / "runs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-vendor-" + uuid4().hex[:8])
    )
    directory.mkdir()
    additional = [
        policy,
        controls_path,
        Path(__file__),
        *map(Path, controls["artifact_hashes"]),
        *[
            root / "src/moneymore/research" / p
            for p in (
                "paper_risk_epoch.py",
                "paper_release_jobs.py",
                "paper_publication.py",
                "paper_signal_planner.py",
                "adjustment_overlay.py",
                "release_evidence.py",
            )
        ],
    ]
    manifest = {
        **old,
        "directory": str(directory),
        "frozen_at": datetime.now(UTC).isoformat(),
        "protocol": yaml.safe_load(policy.read_text(encoding="utf-8")),
        "parent_manifest_sha256": digest(parent / "manifest.json"),
        "policy_change": "USER_AUTHORIZED_VENDOR_CONTROLS_REPLACE_EXHAUSTIVE_ORIGINAL_FINANCIAL_ATTESTATION",
        "input_hashes": {**old["input_hashes"], **{str(p): digest(p) for p in additional}},
    }
    verify_manifest(manifest)
    write_diagnostic(directory / "manifest.json", manifest)
    checksum = digest(directory / "manifest.json")
    for name in manifest["protocol"]["required_stages"]:
        if name == "financial_controls":
            result = {
                **controls,
                "manifest_sha256": checksum,
                "artifact_hashes": {
                    **controls["artifact_hashes"],
                    str(controls_path): digest(controls_path),
                },
            }
        else:
            path = parent / (name + ".json")
            if not path.exists():
                continue
            previous = read_json(path)
            if previous.get("manifest_sha256") != digest(parent / "manifest.json"):
                raise ValueError("PARENT_STAGE_NOT_BOUND")
            for p, value in previous.get("artifact_hashes", {}).items():
                if digest(p) != value:
                    raise ValueError("PARENT_STAGE_ARTIFACT_CHANGED")
            result = {
                **previous,
                "manifest_sha256": checksum,
                "carried_from_stage": str(path),
                "analysis_rerun_claimed": False,
                "artifact_hashes": {
                    **previous.get("artifact_hashes", {}),
                    str(path): digest(path),
                    str(parent / "manifest.json"): digest(parent / "manifest.json"),
                },
            }
        write_diagnostic(directory / (name + ".json"), result)
    contract = read_json(parent / "provisional-paper-contract.json")
    contract.update(manifest_sha256=checksum, paper_activation_authorized=False)
    write_diagnostic(directory / "provisional-paper-contract.json", contract)
    result = publish_status(manifest)
    write_diagnostic(
        base / "latest.json", {"directory": str(directory), "manifest_sha256": checksum}
    )
    print(result)
