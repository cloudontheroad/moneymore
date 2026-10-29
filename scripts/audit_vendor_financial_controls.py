"""Capture vendor snapshot and run practical controls; does not activate V2."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest
from moneymore.research.vendor_financial_controls import audit_vendor_financials

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = root / "data/processed/fina_indicator_research_v2.parquet"
    directory = (
        root
        / "state/baseline-v2-release/vendor-financial-controls"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    directory.mkdir(parents=True)
    frame = pd.read_parquet(source)
    snapshot = directory / "vendor-snapshot.parquet"
    frame.to_parquet(snapshot, index=False)
    result = audit_vendor_financials(frame, "20261029")
    result["artifact_hashes"] = {
        str(p): digest(p)
        for p in (
            snapshot,
            root / "configs/baseline_v2_release_vendor.yaml",
            root / "src/moneymore/research/vendor_financial_controls.py",
            root / "src/moneymore/data/fundamental_research.py",
        )
    }
    result["output_directory"] = str(directory)
    result["whole_release_authorized"] = False
    write_diagnostic(directory / "report.json", result)
    write_diagnostic(root / "state/baseline-v2-release/vendor-financial-controls.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_hashes"}))
