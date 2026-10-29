"""One bounded read-only announcement-index permission/schema check; no retries."""

import json
from pathlib import Path

from dotenv import load_dotenv

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.baseline_v2_release import digest

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    output = root / "state/baseline-v2-release/financial-documents"
    output.mkdir(parents=True, exist_ok=True)
    try:
        frame = TushareProvider()._client.anns_d(
            ts_code="600036.SH", start_date="20231027", end_date="20231029"
        )
    except Exception as error:  # noqa: BLE001 - no provider text/token in logs
        text = str(error)
        result = {
            "status": "ANNOUNCEMENT_INDEX_UNAVAILABLE",
            "error_type": type(error).__name__,
            "permission_denied_indicated": "权限" in text or "permission" in text.lower(),
            "requests": 1,
            "retries": 0,
            "production_changed": False,
        }
    else:
        result = {
            "status": "ANNOUNCEMENT_INDEX_SCHEMA_OBSERVED_NOT_COVERAGE_ACCEPTANCE",
            "rows": len(frame),
            "columns": list(frame.columns),
            "requests": 1,
            "retries": 0,
        }
        if len(frame) > 2000:
            raise ValueError("UNEXPECTEDLY_LARGE_BOUNDED_DISCLOSURE_QUERY")
        path = output / "600036-20231027-20231029-index.parquet"
        frame.to_parquet(path, index=False)
        result["artifact_hashes"] = {str(path): digest(path)}
    write_diagnostic(output / "announcement-index-probe.json", result)
    print(json.dumps(result))
