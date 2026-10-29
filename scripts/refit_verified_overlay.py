"""Isolated corrected-input refit; never publish or mutate existing evidence."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from validate_baseline_v2_weights import monthly_schedule

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_release import digest, read_json
from moneymore.research.baseline_v2_weights import (
    build_monthly_validation_panel,
    validate_weights,
)
from moneymore.research.security_identity import SecurityIdentity


class CorrectedStore:
    """Read-only stable-identity view; replace factors only within verified source coverage."""

    def __init__(self, store, overlay, identity):
        self.store, self.overlay, self.identity = store, overlay, identity
        self.curated = store.curated

    def read(self, table, columns=None, filters=None):
        if table == "adj_factor":
            frame = (
                self.overlay[["ts_code", "trade_date", "derived_adj_factor"]]
                .rename(columns={"derived_adj_factor": "adj_factor"})
                .copy()
            )
        else:
            expanded = []
            for key, op, value in filters or []:
                if key == "ts_code" and op == "in":
                    value = list(value)
                    for change in self.identity.changes:
                        if set(value) & {change["old"], change["new"]}:
                            value = list(set(value) | {change["old"], change["new"]})
                expanded.append((key, op, value))
            frame = self.store.read(table, columns=columns, filters=expanded or None)
            if "ts_code" in frame:
                if table in {"daily", "baseline_v2_daily_basic"}:
                    fields = [c for c in frame if c not in {"ts_code", "trade_date"}]
                    frame = self.identity.market_rows(frame, "trade_date", fields)
                else:
                    frame = self.identity.industry_rows(frame)
        for key, op, value in filters or []:
            if op == "in":
                frame = frame.loc[frame[key].isin(value)]
            elif op == ">=":
                frame = frame.loc[frame[key].ge(value)]
            elif op == "<=":
                frame = frame.loc[frame[key].le(value)]
            else:
                raise ValueError("UNSUPPORTED_RESEARCH_FILTER:" + op)
        return frame[columns].copy() if columns else frame.copy()


def main():
    root = Path(__file__).resolve().parents[1]
    overlay_report = read_json(root / "state/baseline-v2-release/adjustment-overlay.json")
    source = Path(overlay_report["output_directory"]) / "native-with-derived-factors.parquet"
    if digest(source) != overlay_report["artifact_hashes"][str(source)]:
        raise ValueError("OVERLAY_HASH_MISMATCH")
    overlay = pd.read_parquet(source)
    if overlay.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("DUPLICATE_OVERLAY_KEYS")
    run = (
        root
        / "state/baseline-v2-release/corrected-refits"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    (run / "configs").mkdir(parents=True, exist_ok=False)
    target = run / "state/baseline-v2-validation"
    target.mkdir(parents=True)
    shutil.copy2(root / "configs/baseline_v2.yaml", run / "configs/baseline_v2.yaml")
    shutil.copy2(
        root / "state/baseline-v2-validation/industry-download.json",
        target / "industry-download.json",
    )
    policy = yaml.safe_load(
        (root / "configs/baseline_v2_validation.yaml").read_text(encoding="utf-8")
    )
    # Monitoring is not fitting evidence; keep original 2020-2025 splits and exclude later data.
    policy["end"] = policy["test_end"]
    policy["as_of"] = min(str(overlay.trade_date.max()), policy["as_of"])
    changes = yaml.safe_load(
        (root / "configs/baseline_v2_market_evidence.yaml").read_text(encoding="utf-8")
    )["symbol_changes"]
    store = CorrectedStore(ParquetStore(root / "data"), overlay, SecurityIdentity(changes))
    provenance = {
        "status": "BUILDING",
        "directory": str(run),
        "overlay_sha256": digest(source),
        "overlay_verified_steps": overlay_report["verified_steps_removed"],
        "overlay_scope": overlay_report["status"],
        "production_changed": False,
        "real_orders_enabled": False,
        "protocol": policy,
    }
    pointer = root / "state/baseline-v2-release/corrected-refit.json"
    write_diagnostic(pointer, provenance)
    try:
        build_monthly_validation_panel(run, store, policy, monthly_schedule(store, policy))
        result = validate_weights(run, policy)
        # Preserve original validator output; distinguish its superseded documentary flag.
        failures = result["gate_failures"]
        result["vendor_policy_remaining_failures"] = [
            f for f in failures if f != "ORIGINAL_FINANCIAL_VINTAGES_NOT_ATTESTED"
        ]
        provenance.update(status="COMPLETED_RESEARCH_NOT_PUBLISHED", result=result)
    except Exception as exc:
        provenance.update(status="FAILED", error=type(exc).__name__ + ":" + str(exc))
        raise
    finally:
        write_diagnostic(run / "refit.json", provenance)
        write_diagnostic(pointer, provenance)
    print(json.dumps(provenance, default=str), flush=True)


if __name__ == "__main__":
    main()
