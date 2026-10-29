"""Resolve preserved input evidence after an explicit release-policy migration."""

from pathlib import Path

from .baseline_v2_release import digest, read_json, verify_manifest


def union_evidence_manifest(root, directory):
    allowed = (Path(root) / "state/baseline-v2-release/runs").resolve()
    visited = set()
    for _ in range(4):
        directory = Path(directory).resolve()
        if not directory.is_relative_to(allowed) or directory in visited:
            raise ValueError("EVIDENCE_LINEAGE_ESCAPE_OR_CYCLE")
        visited.add(directory)
        manifest = read_json(directory / "manifest.json")
        if Path(manifest["directory"]).resolve() != directory:
            raise ValueError("EVIDENCE_MANIFEST_DIRECTORY_MISMATCH")
        verify_manifest(manifest)
        if (directory / "full-union-native-prices.parquet").is_file() and (
            directory / "full-union-action-alignment.parquet"
        ).is_file():
            return manifest
        stage = read_json(directory / "full_union_economics.json")
        previous = Path(stage["carried_from_stage"]).resolve()
        if (
            stage.get("manifest_sha256") != digest(directory / "manifest.json")
            or stage.get("artifact_hashes", {}).get(str(previous)) != digest(previous)
            or manifest.get("parent_manifest_sha256") != digest(previous.parent / "manifest.json")
        ):
            raise ValueError("EVIDENCE_LINEAGE_HASH_CHANGED")
        directory = previous.parent
    raise ValueError("EVIDENCE_LINEAGE_TOO_DEEP")
