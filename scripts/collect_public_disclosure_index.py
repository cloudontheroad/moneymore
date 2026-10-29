"""Bounded ordinary public disclosure lookup, resumable via immutable page cache.

Only collects metadata; never certifies extracted financial facts or PIT coverage.
No credentials, login, retry, access-control workaround, or real trading.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest


def collect(root, symbol, start, end, *, session=None, pause=time.sleep):
    if not re.fullmatch(r"\d{6}\.(SH|SZ)", symbol):
        raise ValueError("UNSUPPORTED_SYMBOL")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if not 0 <= (last - first).days <= 10:
        raise ValueError("WINDOW_MUST_BE_AT_MOST_11_CALENDAR_DATES")
    code, exchange = symbol.split(".")
    directory = Path(root) / "state/baseline-v2-release/financial-documents/public-index"
    directory /= f"{symbol}-{first:%Y%m%d}-{last:%Y%m%d}"
    directory.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.headers["User-Agent"] = "MoneyMore/1.0 personal-disclosure-research"
    requests_made = 0

    def post(name, url, **kwargs):
        nonlocal requests_made
        path = directory / f"{name}.json"
        binding = {"url": url, **kwargs}
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            body = json.dumps(cached["data"], sort_keys=True, ensure_ascii=False).encode()
            if (
                cached["request"] != binding
                or hashlib.sha256(body).hexdigest() != cached["data_sha256"]
            ):
                raise ValueError("CACHE_BINDING_OR_CONTENT_CHANGED")
            return cached["data"]
        if requests_made:
            pause(2)
        requests_made += 1
        response = session.post(url, timeout=25, **kwargs)
        response.raise_for_status()
        if len(response.content) > 10000000:
            raise ValueError("RESPONSE_TOO_LARGE")
        data = response.json()
        body = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        write_diagnostic(
            path,
            {"request": binding, "data": data, "data_sha256": hashlib.sha256(body).hexdigest()},
        )
        return data

    try:
        stocks = post(
            "identity",
            "https://www.cninfo.com.cn/new/information/topSearch/query",
            params={"keyWord": code, "maxNum": 10},
        )
        if not isinstance(stocks, list):
            raise TypeError("IDENTITY_SCHEMA")
        matches = [s for s in stocks if s.get("code") == code and s.get("orgId")]
        if len(matches) != 1:
            raise ValueError("IDENTITY_NOT_UNIQUE")
        rows, seen, total = [], set(), None
        complete = False
        for page in range(1, 4):
            payload = {
                "stock": code + "," + matches[0]["orgId"],
                "tabName": "fulltext",
                "pageSize": 30,
                "pageNum": page,
                "column": "sse" if exchange == "SH" else "szse",
                "seDate": f"{first:%Y-%m-%d}~{last:%Y-%m-%d}",
                "searchkey": "",
                "isHLtitle": "false",
            }
            data = post(
                f"page-{page}", "https://www.cninfo.com.cn/new/hisAnnouncement/query", data=payload
            )
            items, count = data.get("announcements"), data.get("totalRecordNum")
            if (
                not isinstance(items, list)
                or len(items) > 30
                or type(count) is not int
                or count < 0
            ):
                raise ValueError("INDEX_SCHEMA")
            if total is not None and total != count:
                raise ValueError("INDEX_CHANGED_DURING_PAGINATION")
            total = count
            for item in items:
                identifier = str(item["announcementId"])
                published = (
                    pd.Timestamp(item["announcementTime"], unit="ms", tz="UTC")
                    .tz_convert("Asia/Shanghai")
                    .tz_localize(None)
                    .normalize()
                )
                if item["secCode"] != code or not first <= published <= last or identifier in seen:
                    raise ValueError("INDEX_IDENTITY_DATE_OR_DUPLICATE")
                seen.add(identifier)
                rows.append(item)
            if len(rows) == total and not data.get("hasMore"):
                complete = True
                break
            if not items or len(rows) > total:
                raise ValueError("INCONSISTENT_PAGINATION")
        result = {
            "status": "INDEX_COMPLETE_FACTS_NOT_REVIEWED"
            if complete
            else "PAGE_BUDGET_EXHAUSTED_INCOMPLETE",
            "symbol": symbol,
            "rows": rows,
            "total": total,
            "requests": requests_made,
            "maximum_requests": 4,
            "versions_collapsed": False,
            "financial_gate_passed": False,
            "artifact_hashes": {
                p.name: digest(p) for p in directory.glob("*.json") if p.name != "result.json"
            },
        }
    except Exception as error:  # noqa: BLE001 - stop without retries, retain evidence
        result = {
            "status": "STOPPED_NO_RETRY",
            "symbol": symbol,
            "requests": requests_made,
            "error_type": type(error).__name__,
            "financial_gate_passed": False,
        }
    write_diagnostic(directory / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    result = collect(Path(__file__).resolve().parents[1], args.symbol, args.start, args.end)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "artifact_hashes"}}))
