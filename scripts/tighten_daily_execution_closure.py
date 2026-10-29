"""Tighten a hash-verified ordered exit_rank support to the proven top_k support."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    parent = json.loads((base / "execution-closure.json").read_text(encoding="utf-8"))
    path = Path(parent["output_directory"]) / "candidate-support.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != parent["support_sha256"]:
        raise ValueError("PARENT_SUPPORT_HASH_MISMATCH")
    scores = json.loads((base / "daily-scores.json").read_text(encoding="utf-8"))
    if scores["panel_sha256"] != parent["score_panel_sha256"]:
        raise ValueError("FROZEN_SCORE_CLOSURE_MISMATCH")
    frozen = json.loads(
        (Path(scores["output_directory"]) / "frozen-inputs.json").read_text(encoding="utf-8")
    )
    top_k = int(frozen["policy"]["portfolio"]["top_k"])
    frame = pd.read_parquet(path)
    if (
        len(frame.groupby(["date", "variant"])) != 926
        or not frame.groupby(["date", "variant"]).symbol.nunique().ge(top_k).all()
    ):
        raise ValueError("COMPLETE_ORDERED_CLOSURE_REQUIRED")
    narrowed = frame.groupby(["date", "variant"], sort=False).head(top_k)
    directory = base / "execution-closures" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    narrowed.to_parquet(directory / "candidate-support.parquet", index=False)
    result = {
        **parent,
        "output_directory": str(directory),
        "required_symbols": sorted(narrowed.symbol.unique()),
        "parent_support_sha256": parent["support_sha256"],
        "support_sha256": hashlib.sha256(
            (directory / "candidate-support.parquet").read_bytes()
        ).hexdigest(),
        "proof": "With k retained names, the deterministic first top_k has at least top_k-k nonheld names. New additions are a subset; retained names were prior additions. Verified only for frozen constructor and same-identity corporate actions.",
    }
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(base / "execution-closure.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "required_symbols": len(result["required_symbols"]),
                "output_directory": str(directory),
            }
        )
    )
