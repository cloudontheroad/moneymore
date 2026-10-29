"""Select report documents for review, not financial facts or PIT acceptance."""

import re

import pandas as pd


def report_candidates(symbol, period, announcement_date, announcements):
    end = pd.Timestamp(period)
    terms = {
        3: r"(?:第一|一|第1|1)季度报告",
        6: r"半年度报告",
        9: r"(?:第三|三|第3|3)季度报告",
        12: r"(?<!半)年度报告",
    }
    if end.month not in terms or end != end + pd.offsets.QuarterEnd(0):
        raise ValueError("INVALID_REPORT_PERIOD")
    candidates = []
    seen = set()
    for row in announcements:
        title = re.sub(r"\s+", "", re.sub(r"<[^>]*>", "", row["announcementTitle"]))
        if row["secCode"] != symbol.split(".")[0]:
            raise ValueError("CATALOG_SYMBOL_MISMATCH")
        if (
            str(end.year) not in title
            or re.search(terms[end.month], title) is None
            or any(word in title for word in ("摘要", "英文", "提示", "董事会", "监事会"))
            or title.endswith("更正公告")
        ):
            continue
        path = row["adjunctUrl"]
        if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/\d+\.pdf", path, re.IGNORECASE):
            raise ValueError("NONCANONICAL_PUBLIC_DOCUMENT_PATH")
        identifier = str(row["announcementId"])
        if identifier in seen:
            raise ValueError("DUPLICATE_REPORT_ANNOUNCEMENT")
        seen.add(identifier)
        published = (
            pd.Timestamp(row["announcementTime"], unit="ms", tz="UTC")
            .tz_convert("Asia/Shanghai")
            .strftime("%Y%m%d")
        )
        candidates.append(
            {
                "announcement_id": identifier,
                "title": row["announcementTitle"],
                "published_date": published,
                "matches_dependency_publication": published == announcement_date,
                "url": "https://static.cninfo.com.cn/" + path,
                "financial_facts_reviewed": False,
            }
        )
    return {
        "symbol": symbol,
        "report_period": period,
        "dependency_announcement_date": announcement_date,
        "status": "MULTIPLE_VERSIONS_REQUIRE_REVIEW"
        if len(candidates) > 1
        else "DOCUMENT_READY_FOR_FACT_REVIEW"
        if candidates
        else "REPORT_DOCUMENT_NOT_FOUND_IN_WINDOW",
        "candidates": candidates,
        "later_correction_search_complete": False,
        "full_financial_vintage_gate_passed": False,
    }
