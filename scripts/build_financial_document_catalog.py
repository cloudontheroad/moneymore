"""Build a hash-checked review catalog from already downloaded public index pages."""

import json
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest
from moneymore.research.disclosure_catalog import report_candidates

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-release"
    report = json.loads((base / "financial-vintage-requirements.json").read_text(encoding="utf-8"))
    source = next(Path(p) for p in report["artifact_hashes"] if p.endswith("-requirements.parquet"))
    if digest(source) != report["artifact_hashes"][str(source)]:
        raise ValueError("DEPENDENCIES_CHANGED")
    dependencies = pd.read_parquet(source)[["symbol", "end_date", "ann_date"]].drop_duplicates()
    output, artifacts = [], {str(source): digest(source)}
    for result_path in sorted((base / "financial-documents/public-index").glob("*/result.json")):
        index = json.loads(result_path.read_text(encoding="utf-8"))
        if index["status"] != "INDEX_COMPLETE_FACTS_NOT_REVIEWED":
            continue
        announcements = []
        for name, checksum in index["artifact_hashes"].items():
            path = result_path.parent / name
            if path.name != name or digest(path) != checksum:
                raise ValueError("PUBLIC_INDEX_ARTIFACT_CHANGED")
            artifacts[str(path)] = checksum
            if name.startswith("page-"):
                envelope = json.loads(path.read_text(encoding="utf-8"))
                announcements.extend(envelope["data"]["announcements"])
        for row in dependencies.loc[dependencies.symbol.eq(index["symbol"])].itertuples():
            available_dates = {
                pd.Timestamp(a["announcementTime"], unit="ms", tz="UTC")
                .tz_convert("Asia/Shanghai")
                .strftime("%Y%m%d")
                for a in announcements
            }
            if row.ann_date not in available_dates:
                continue
            item = report_candidates(row.symbol, row.end_date, row.ann_date, announcements)
            item["index_result_path"] = str(result_path)
            output.append(item)
        artifacts[str(result_path)] = digest(result_path)
    result = {
        "status": "CATALOG_ONLY_NOT_ORIGINAL_FACT_ACCEPTANCE",
        "dependency_windows_cataloged": len(output),
        "statuses": pd.Series([r["status"] for r in output], dtype="str").value_counts().to_dict(),
        "records": output,
        "artifact_hashes": artifacts,
        "real_orders_enabled": False,
        "full_financial_vintage_gate_passed": False,
    }
    write_diagnostic(base / "financial-document-catalog.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "artifact_hashes"}}))
