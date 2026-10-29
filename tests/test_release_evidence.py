import json

import pytest

from moneymore.research.baseline_v2_release import digest
from moneymore.research.release_evidence import union_evidence_manifest


def setup_lineage(tmp_path):
    base = tmp_path / "state/baseline-v2-release/runs"
    old, new = base / "original", base / "vendor"
    old.mkdir(parents=True)
    new.mkdir()
    source = old / "full-union-native-prices.parquet"
    source.write_bytes(b"SYNTHETIC")
    alignment = old / "full-union-action-alignment.parquet"
    alignment.write_bytes(b"SYNTHETIC")
    old_manifest = old / "manifest.json"
    old_manifest.write_text(
        json.dumps(
            {
                "directory": str(old),
                "input_hashes": {str(source): digest(source), str(alignment): digest(alignment)},
            }
        )
    )
    old_stage = old / "full_union_economics.json"
    old_stage.write_text('{"status":"SYNTHETIC_NOT_PASS"}')
    (new / "manifest.json").write_text(
        json.dumps(
            {
                "directory": str(new),
                "input_hashes": {},
                "parent_manifest_sha256": digest(old_manifest),
            }
        )
    )
    (new / "full_union_economics.json").write_text(
        json.dumps(
            {
                "manifest_sha256": digest(new / "manifest.json"),
                "carried_from_stage": str(old_stage),
                "artifact_hashes": {str(old_stage): digest(old_stage)},
            }
        )
    )
    return old, new


def test_policy_migration_reuses_original_evidence_directory(tmp_path):
    old, new = setup_lineage(tmp_path)
    assert union_evidence_manifest(tmp_path, new)["directory"] == str(old)


def test_parent_stage_tamper_not_silently_followed(tmp_path):
    old, new = setup_lineage(tmp_path)
    (old / "full_union_economics.json").write_text("changed")
    with pytest.raises(ValueError, match="LINEAGE_HASH"):
        union_evidence_manifest(tmp_path, new)


def test_parent_native_data_tamper_rejected(tmp_path):
    old, new = setup_lineage(tmp_path)
    (old / "full-union-native-prices.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="INPUT_HASH"):
        union_evidence_manifest(tmp_path, new)
