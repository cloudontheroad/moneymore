import json
import pickle

import pytest

from moneymore.training_artifact_checks import validate_research_checkpoint


def checkpoint(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    names = ["gru_seed17.pkl", "qlib_lightgbm_alpha360_v1.pkl"]
    for name in names:
        (models / name).write_bytes(pickle.dumps({"test": 1}))
    (models / "gru_ensemble.json").write_text(json.dumps({"models": [{"filename": n} for n in names]}))
    (tmp_path / "research.json").write_text(json.dumps({"protocol": {"model_id": "gru", "model": {"seeds": [17]}},
        "time_contract": {"present": True}, "metrics": [1],
        "research_selected_rounds": {n[:-4]: 3 for n in names}}))
    (tmp_path / "_RESEARCH_COMPLETE").touch()
    return tmp_path


def test_valid_checkpoint_and_corrupt_artifact(tmp_path):
    folder = checkpoint(tmp_path)
    assert len(validate_research_checkpoint(folder)["models"]) == 2
    (folder / "models/gru_seed17.pkl").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        validate_research_checkpoint(folder)


def test_completion_marker_required(tmp_path):
    folder = checkpoint(tmp_path)
    (folder / "_RESEARCH_COMPLETE").unlink()
    with pytest.raises(ValueError, match="marker"):
        validate_research_checkpoint(folder)
