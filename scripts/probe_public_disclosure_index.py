"""Two ordinary public CNINFO requests, no credentials/bypass/retry or bulk crawl."""

import json
import time
from pathlib import Path

import requests

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    directory = (
        root
        / "state/baseline-v2-release/financial-documents/cninfo-index/600036-20231027-20231029/v2-official-js"
    )
    directory.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "MoneyMore/1.0 personal-disclosure-research"
    calls = 0
    try:
        # Route and params observed directly in the public official common.js.
        response = session.post(
            "https://www.cninfo.com.cn/new/information/topSearch/query",
            params={"keyWord": "600036", "maxNum": 10},
            timeout=20,
        )
        calls += 1
        response.raise_for_status()
        if len(response.content) > 10000000:
            raise ValueError("PUBLIC_INDEX_RESPONSE_TOO_LARGE")
        stocks = response.json()
        matches = [s for s in stocks if s.get("code") == "600036"]
        if len(matches) != 1 or not matches[0].get("orgId"):
            raise ValueError("PUBLIC_STOCK_IDENTITY_NOT_UNIQUE")
        write_diagnostic(directory / "stock-reference.json", matches[0])
        time.sleep(2)
        payload = {
            "stock": "600036," + matches[0]["orgId"],
            "tabName": "fulltext",
            "pageSize": 30,
            "pageNum": 1,
            "column": "sse",
            "category": "category_sjdbg_szsh",
            "seDate": "2023-10-27~2023-10-29",
            "searchkey": "",
            "isHLtitle": "false",
        }
        response = session.post(
            "https://www.cninfo.com.cn/new/hisAnnouncement/query", data=payload, timeout=20
        )
        calls += 1
        response.raise_for_status()
        data = response.json()
        announcements = data.get("announcements")
        if not isinstance(announcements, list) or len(announcements) > 30:
            raise ValueError("PUBLIC_DISCLOSURE_INDEX_SCHEMA_OR_CAP")
        for item in announcements:
            if item.get("secCode") != "600036":
                raise ValueError("PUBLIC_DISCLOSURE_SYMBOL_FILTER_NOT_HONORED")
        path = directory / "response.json"
        write_diagnostic(path, {"request": payload, "response": data})
        result = {
            "status": "PUBLIC_INDEX_READABLE_NOT_BULK_ACCEPTANCE",
            "requests": calls,
            "rows": len(announcements),
            "total": data.get("totalRecordNum"),
            "artifact_hashes": {str(path): digest(path)},
        }
    except Exception as error:  # noqa: BLE001 - no retry or access-control workaround
        result = {
            "status": "PUBLIC_INDEX_UNAVAILABLE_NO_BYPASS",
            "requests": calls,
            "error_type": type(error).__name__,
            "http_status": response.status_code if "response" in globals() else None,
        }
    write_diagnostic(directory / "probe.json", result)
    print(json.dumps(result))
