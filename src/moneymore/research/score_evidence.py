"""Immutable as-produced factor evidence, independent of paper execution history."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ..challenger_diagnostics import write_diagnostic


def archive_score_evidence(
    root: Path,
    capture: dict[str, pd.DataFrame],
    ranking: pd.DataFrame,
    policy: dict,
    *,
    version: str = "global_multifactor_v1",
) -> dict:
    processed = capture["processed_factors"]
    enriched = processed.copy()
    for factor, weight in policy["factors"].items():
        enriched[f"effective_weight__{factor}"] = (
            weight / enriched.active_weight.where(enriched.active_weight > 0)
        ).where(enriched[factor].notna(), 0.0)
        enriched[f"contribution__{factor}"] = (
            pd.to_numeric(enriched[factor], errors="coerce").fillna(0)
            * enriched[f"effective_weight__{factor}"]
        )
    ranked = ranking.copy()
    ranked["raw_score_rank"] = ranked.score.rank(ascending=False, method="first")
    date = pd.Timestamp(processed.date.max()).strftime("%Y%m%d")
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    directory = root / "state/score-evidence" / version / date / run
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, frame in {**capture, "processed_factors": enriched, "ranking": ranked}.items():
        path = directory / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        files[name] = {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    missing = {name: int(processed[name].isna().sum()) for name in policy["factors"]}
    manifest = {
        "as_of_date": date,
        "captured_at": datetime.now(UTC).isoformat(),
        "model_version": version,
        "policy": policy,
        "files": files,
        "missing_factor_counts": missing,
        "complete_factor_rows": int(processed[list(policy["factors"])].notna().all(axis=1).sum()),
        "size_control_present": "total_mv" in processed,
        "industry_neutralization_enabled": False,
        "data_gate": "PASS" if not any(missing.values()) else "INCOMPLETE_FACTORS",
        "gate_scope": "AUDIT_ONLY_FOR_EXISTING_VERSION",
        "path": str(directory),
        "code_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in (
                "src/moneymore/research/sector_model.py",
                "src/moneymore/factors/preprocess.py",
                "src/moneymore/factors/builtin.py",
                "src/moneymore/portfolio_constructor.py",
            )
            if (root / relative).exists()
        },
    }
    write_diagnostic(directory / "manifest.json", manifest)
    write_diagnostic(root / "state/score-evidence" / version / "latest.json", manifest)
    return manifest
