"""Build a versioned overlay from independently proven steps; not a release gate."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.adjustment_overlay import non_economic_overlay
from moneymore.research.baseline_v2_release import digest, read_json
from moneymore.research.release_evidence import union_evidence_manifest

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-release"
    manifest = union_evidence_manifest(root, read_json(base / "latest.json")["directory"])
    source = Path(manifest["directory"]) / "full-union-native-prices.parquet"
    native_sha = digest(source)
    native = pd.read_parquet(source)
    progress = read_json(base / "union-independent-progress.json")
    proof_root = Path(progress["directory"]).resolve()
    if not proof_root.is_relative_to(base / "union-independent"):
        raise ValueError("INDEPENDENT_EVIDENCE_OUTSIDE_RESEARCH_ROOT")
    targets, artifacts = [], {str(source): native_sha}
    # Freeze this list: concurrently arriving proofs belong to the next version.
    proofs = sorted(
        p for p in proof_root.glob("*.json") if p.name not in {"progress.json", "cooldown.json"}
    )
    for path in proofs:
        proof = read_json(path)
        if proof["native_source_sha256"] != native_sha:
            raise ValueError("OVERLAY_PROOF_NATIVE_SOURCE_CHANGED")
        if proof.get("verified") is not True:
            continue
        if proof.get("reason") != "VERIFIED_NON_ECONOMIC_VENDOR_FACTOR_STEP":
            raise ValueError("UNSUPPORTED_OVERLAY_EVIDENCE_CLASS")
        for artifact, checksum in proof["artifact_hashes"].items():
            if artifact not in artifacts:
                if digest(artifact) != checksum:
                    raise ValueError("OVERLAY_PROOF_ARTIFACT_CHANGED")
                artifacts[artifact] = checksum
            elif artifacts[artifact] != checksum:
                raise ValueError("CONFLICTING_PROOF_ARTIFACT_HASH")
        targets.append((proof["symbol"], proof["trade_date"]))
        artifacts[str(path)] = digest(path)
    output = non_economic_overlay(native, targets)
    directory = (
        base
        / "adjustment-overlays"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    directory.mkdir(parents=True)
    target = directory / "native-with-derived-factors.parquet"
    output.to_parquet(target, index=False)
    artifacts.update(
        {
            str(target): digest(target),
            str(Path(__file__)): digest(__file__),
            str(root / "src/moneymore/research/adjustment_overlay.py"): digest(
                root / "src/moneymore/research/adjustment_overlay.py"
            ),
        }
    )
    result = {
        "status": "PARTIAL_VERIFIED_OVERLAY_NOT_FULL_ECONOMIC_RELEASE",
        "output_directory": str(directory),
        "rows": len(output),
        "verified_steps_removed": len(targets),
        "source_adj_factor_overwritten": False,
        "other_steps_preserved": True,
        "missing_factors_filled": False,
        "production_changed": False,
        "whole_release_authorized": False,
        "artifact_hashes": artifacts,
    }
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(base / "adjustment-overlay.json", result)
    print({k: v for k, v in result.items() if k != "artifact_hashes"})
