"""Derive reviewed original-document financials; never certifies the whole pool."""

import json
from pathlib import Path

import pandas as pd
import yaml

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest
from moneymore.research.disclosure_financials import derive_disclosure, vendor_value_matches

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = root / "configs/reviewed_financial_disclosures.yaml"
    reviewed = yaml.safe_load(source.read_text(encoding="utf-8"))
    vendor_path = root / "data/processed/fina_indicator_research_v2.parquet"
    vendor = pd.read_parquet(vendor_path)
    rows = []
    for record in reviewed["records"]:
        result = derive_disclosure(record, root, "20261029")
        candidates = vendor.loc[
            vendor.ts_code.eq(record["symbol"])
            & vendor.end_date.eq(record["report_period"])
            & vendor.ann_date.eq(record["published_date"])
        ]
        matches = []
        for row in candidates.itertuples():
            matches.append(
                {
                    "ann_date": row.ann_date,
                    "roe": row.roe,
                    "q_sales_yoy": row.q_sales_yoy,
                    "roe_matches_disclosure_precision": vendor_value_matches(
                        result["roe"], row.roe
                    ),
                    "quarter_growth_matches_disclosure_precision": vendor_value_matches(
                        result["q_sales_yoy"], row.q_sales_yoy
                    ),
                }
            )
        rows.append({**result, "vendor_comparisons": matches})
    report = {
        "status": "PARTIAL_ORIGINAL_DOCUMENT_RECONSTRUCTION_NOT_FULL_PIT_ATTESTATION",
        "reports": rows,
        "reviewed_documents": len(rows),
        "artifact_hashes": {
            str(p): digest(p)
            for p in [
                source,
                vendor_path,
                *[root / r["document_path"] for r in reviewed["records"]],
                *[root / r["publication_index"]["path"] for r in reviewed["records"]],
            ]
        },
        "full_financial_vintage_gate_passed": False,
        "production_changed": False,
        "real_orders_enabled": False,
    }
    write_diagnostic(
        root / "state/baseline-v2-release/reviewed-financial-reconstruction.json", report
    )
    print(json.dumps(report))
