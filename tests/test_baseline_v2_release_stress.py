import json

import pytest

from moneymore.research.baseline_v2_release_stress import collect_execution_stress


def test_missing_full_replays_never_pass(tmp_path):
    source = tmp_path / "reference"
    source.mkdir()
    (source / "frozen-protocol.json").write_text(json.dumps({"scoring": {"v": 1}}))
    directory = tmp_path / "release"
    directory.mkdir()
    (directory / "manifest.json").write_text("{}")
    result = collect_execution_stress(
        tmp_path, {"directory": str(directory), "replay_directory": str(source)}
    )
    assert result["status"] != "PASS"
    assert len(result["missing_scenarios"]) == 5
    assert not result["real_orders_enabled"]


def test_mismatched_model_cannot_supply_stress_evidence(tmp_path):
    source, replay = tmp_path / "reference", tmp_path / "stress"
    source.mkdir()
    replay.mkdir()
    (source / "frozen-protocol.json").write_text(json.dumps({"scoring": {"v": 1}}))
    (replay / "frozen-protocol.json").write_text(json.dumps({"scoring": {"v": 2}}))
    pointer = (
        tmp_path
        / "state/baseline-v2-validation/daily-inputs/stress-replays/slippage_5bps/latest.json"
    )
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"output_directory": str(replay)}))
    with pytest.raises(ValueError, match="STRESS_SCORING_CONTRACT"):
        collect_execution_stress(tmp_path, {"replay_directory": str(source)})
