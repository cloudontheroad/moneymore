"""Map reviewed original evidence to exact frozen dependencies, not sample counts."""

import json
from pathlib import Path

import pandas as pd
import yaml

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest
from moneymore.research.disclosure_financials import derive_disclosure, vendor_value_matches

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    requirement_report = root / "state/baseline-v2-release/financial-vintage-requirements.json"
    inventory = json.loads(requirement_report.read_text(encoding="utf-8"))
    paths = [Path(p) for p in inventory["artifact_hashes"] if p.endswith("-requirements.parquet")]
    if len(paths) != 1 or digest(paths[0]) != inventory["artifact_hashes"][str(paths[0])]:
        raise ValueError("FROZEN_FINANCIAL_DEPENDENCIES_CHANGED")
    frame = pd.read_parquet(paths[0])
    reviewed_path = root / "configs/reviewed_financial_disclosures.yaml"
    records = yaml.safe_load(reviewed_path.read_text(encoding="utf-8"))["records"]
    evidence = {}
    for record in records:
        derived = derive_disclosure(record, root, "20261029")
        key = (record["symbol"], record["report_period"], record["published_date"])
        if key in evidence:
            raise ValueError("AMBIGUOUS_REVIEWED_DISCLOSURE")
        evidence[key] = derived
    statuses = []
    for row in frame.itertuples():
        derived = evidence.get((row.symbol, row.end_date, row.ann_date))
        if derived is None:
            status = "MISSING_REVIEWED_ORIGINAL"
        elif str(row.first_signal_date) < derived["available_from"]:
            status = "NOT_VISIBLE_AT_FIRST_SIGNAL"
        elif row.field not in derived or not vendor_value_matches(
            derived[row.field], row.source_value
        ):
            status = "VALUE_OR_DEFINITION_MISMATCH"
        else:
            status = "REVIEWED_ORIGINAL_DERIVATION_MATCH"
        statuses.append(status)
    frame["original_evidence_status"] = statuses
    destination = root / "state/baseline-v2-release/financial-vintage-requirements"
    path = destination / f"{digest(reviewed_path)[:20]}-original-coverage.parquet"
    frame.to_parquet(path, index=False)
    result = {
        "status": "PARTIAL_ORIGINAL_DEPENDENCY_COVERAGE_NOT_RELEASE",
        "dependency_rows": len(frame),
        "status_counts": frame.original_evidence_status.value_counts().to_dict(),
        "reviewed_reports": len(evidence),
        "artifact_hashes": {str(p): digest(p) for p in (paths[0], reviewed_path, path)},
        "original_training_input_provenance_claimed": False,
        "full_financial_vintage_gate_passed": False,
        "production_changed": False,
        "real_orders_enabled": False,
    }
    write_diagnostic(root / "state/baseline-v2-release/reviewed-financial-coverage.json", result)
    print(json.dumps(result))
