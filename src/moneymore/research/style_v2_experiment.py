"""Explicit forward paper experiment, separate from validated-model promotion."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCOPE = "FORWARD_STYLE_REPAIR_EXPERIMENT_NOT_PROMOTION"


def execution_policy_hash(candidate):
    return hashlib.sha256(json.dumps(candidate.get("execution_policy", {}), sort_keys=True).encode()).hexdigest()


def candidate_execution_config(config, candidate):
    """Prospective per-account execution costs; never mutate shared defaults."""
    override = candidate.get("execution_policy")
    if not override:
        return config
    if candidate.get("portfolio_variant") != "factor_baseline_v2" or override.get("real_orders_enabled") is not False:
        raise ValueError("INVALID_PAPER_EXECUTION_COST_POLICY")
    slippage = float(override["slippage_bps"])
    if not np.isfinite(slippage) or slippage < 0 or not override.get("version"):
        raise ValueError("INVALID_EXECUTION_SLIPPAGE")
    return type(config).model_validate({**config.model_dump(), "slippage_bps": slippage})


def public_style_details(root, candidate):
    model = candidate.get("factor_policy", {}).get("experimental_linear_model")
    if not model:
        return {}
    path = Path(model["evidence_path"]).resolve()
    if not path.is_relative_to((root / "state/baseline-v2-style-release").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != model["evidence_sha256"]:
        return {"validation_hold": True, "style_contract_error": "STYLE_EVIDENCE_CHANGED"}
    evidence = json.loads(path.read_text(encoding="utf-8"))
    directory = Path(evidence["style_audit"]["directory"])
    eligibility = directory / "eligibility.parquet"
    if hashlib.sha256(eligibility.read_bytes()).hexdigest() != evidence["artifact_hashes"][str(eligibility)]:
        return {"validation_hold": True, "style_contract_error": "STYLE_ELIGIBILITY_CHANGED"}
    frame = pd.read_parquet(eligibility)
    return {"style_exclusions": frame.loc[frame.excluded, ["symbol", "name", "industry", "board", "exclusion_reasons"]].fillna("").to_dict("records")}


def validate_experiment(root, policy):
    model = policy.get("experimental_linear_model")
    if not model:
        return
    path = Path(model["evidence_path"]).resolve()
    if not path.is_relative_to((root / "state/baseline-v2-style-release").resolve()):
        raise ValueError("STYLE_V2_EVIDENCE_OUTSIDE_ISOLATED_RELEASE")
    if hashlib.sha256(path.read_bytes()).hexdigest() != model["evidence_sha256"]:
        raise ValueError("STYLE_V2_EVIDENCE_CHANGED")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (
        policy.get("real_orders_enabled") is not False
        or policy.get("automatic_promotion") is not False
        or model.get("scope") != SCOPE
        or evidence.get("scope") != SCOPE
        or evidence.get("coefficients") != model.get("coefficients")
        or evidence.get("real_orders_enabled") is not False
    ):
        raise ValueError("INVALID_FORWARD_STYLE_EXPERIMENT_CONTRACT")
    for artifact, checksum in evidence["artifact_hashes"].items():
        if hashlib.sha256(Path(artifact).read_bytes()).hexdigest() != checksum:
            raise ValueError("STYLE_V2_FROZEN_ARTIFACT_CHANGED")


class OverlayStore:
    """Derived-factor read view; preserve missing rows and every unverified step."""

    def __init__(self, store, path, checksum, identity):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
            raise ValueError("STYLE_V2_OVERLAY_CHANGED")
        self.store, self.identity = store, identity
        self.curated = store.curated
        frame = pd.read_parquet(
            path, columns=["ts_code", "trade_date", "adjustment_overlay_multiplier"]
        )
        if frame.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError("DUPLICATE_STYLE_OVERLAY")
        self.frame = frame
        self.last = frame.sort_values("trade_date").groupby("ts_code").tail(1).set_index("ts_code")

    def read(self, table, **kwargs):
        frame = self.store.read(table, **kwargs)
        if table != "adj_factor":
            return frame
        result = frame.copy()
        keys = pd.DataFrame(
            {
                "ts_code": result.ts_code.map(self.identity.stable),
                "trade_date": result.trade_date.astype(str),
            },
            index=result.index,
        )
        match = keys.merge(
            self.frame, on=["ts_code", "trade_date"], how="left", validate="many_to_one"
        )
        multiplier = match.adjustment_overlay_multiplier.to_numpy()
        last_dates = keys.ts_code.map(self.last.trade_date).fillna("99999999")
        future = keys.trade_date.gt(last_dates).to_numpy()
        terminal = keys.ts_code.map(self.last.adjustment_overlay_multiplier).to_numpy()
        multiplier = np.where(future & pd.isna(multiplier), terminal, multiplier)
        # Unknown securities/dates retain their source factor; no missing price/factor imputation.
        result["adj_factor"] = pd.to_numeric(
            result.adj_factor, errors="coerce"
        ).to_numpy() * np.where(pd.isna(multiplier), 1.0, multiplier)
        return result
