"""Non-executing validation before accepting a durable research checkpoint."""
import hashlib
import json
import pickletools
from pathlib import Path


def validate_research_checkpoint(folder: Path):
    if not (folder / "_RESEARCH_COMPLETE").is_file():
        raise ValueError("research completion marker missing")
    report_path = folder / "research.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report["protocol"]
    if not report.get("time_contract") or not report.get("metrics"):
        raise ValueError("research evidence incomplete")
    ensemble = folder / "models" / f'{config["model_id"]}_ensemble.json'
    members = json.loads(ensemble.read_text(encoding="utf-8"))["models"]
    expected = {f'{config["model_id"]}_seed{seed}.pkl' for seed in config["model"]["seeds"]}
    expected.add("qlib_lightgbm_alpha360_v1.pkl")
    if {r["filename"] for r in members} != expected or len(members) != len(expected):
        raise ValueError("ensemble membership incomplete")
    hashes = {}
    for name in sorted(expected):
        if Path(name).name != name:
            raise ValueError("invalid model name")
        rounds = report.get("research_selected_rounds", {}).get(Path(name).stem)
        if type(rounds) is not int or rounds < 1:
            raise ValueError("selected rounds missing")
        raw = (folder / "models" / name).read_bytes()
        # Inspect pickle syntax without executing CUDA deserialization in the
        # supervisor. The refit phase separately loads actual local models.
        operations = list(pickletools.genops(raw))
        if not operations or operations[-1][0].name != "STOP":
            raise ValueError("incomplete model pickle")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {"research_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "models": hashes, "validation": "complete files and pickle syntax; runtime load occurs in refit"}
