"""Point-in-time derivation from reviewed, hashed original disclosure facts.

Canonical explicit formulas, NOT certification of a vendor's undocumented
formula or a blanket declaration that all vendor historical values are PIT.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .baseline_v2_release import digest


def verify_publication(record, root):
    """Bind reviewed facts to a saved official index entry and its actual PDF.

    A date embedded in a URL or an issuer's latest download is not sufficient.
    This proves the selected publication, not completeness of all later corrections.
    """
    evidence = record["publication_index"]
    path = Path(evidence["path"])
    path = path if path.is_absolute() else Path(root) / path
    if digest(path) != evidence["sha256"]:
        raise ValueError("PUBLICATION_INDEX_CHANGED")
    data = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("format", "observed_probe_v1") == "public_index_cache_v1":
        body = json.dumps(data["data"], sort_keys=True, ensure_ascii=False).encode()
        if (
            hashlib.sha256(body).hexdigest() != data["data_sha256"]
            or data["request"]["url"] != "https://www.cninfo.com.cn/new/hisAnnouncement/query"
            or not data["request"]["data"]["stock"].startswith(record["symbol"].split(".")[0] + ",")
        ):
            raise ValueError("PUBLICATION_INDEX_REQUEST_OR_CONTENT_MISMATCH")
        rows = data["data"]["announcements"]
    elif evidence.get("format", "observed_probe_v1") == "observed_probe_v1":
        rows = data["response"]["announcements"]
    else:
        raise ValueError("UNSUPPORTED_PUBLICATION_INDEX_FORMAT")
    selected = [r for r in rows if str(r["announcementId"]) == str(evidence["announcement_id"])]
    if len(selected) != 1:
        raise ValueError("PUBLICATION_ID_NOT_UNIQUE")
    entry = selected[0]
    url = urlparse(record["source_url"])
    published = pd.Timestamp(entry["announcementTime"], unit="ms", tz="UTC").tz_convert(
        "Asia/Shanghai"
    )
    if (
        entry["secCode"] != record["symbol"].split(".")[0]
        or published.strftime("%Y%m%d") != record["published_date"]
        or url.scheme != "https"
        or url.netloc != "static.cninfo.com.cn"
        or url.path != "/" + entry["adjunctUrl"].lstrip("/")
        or str(entry["adjunctType"]).upper() != "PDF"
        or not record.get("announcement_title")
        or entry.get("announcementTitle") != record["announcement_title"]
        or record["publication_evidence_url"]
        != "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    ):
        raise ValueError("PUBLICATION_IDENTITY_DATE_OR_DOCUMENT_MISMATCH")
    return str(entry["announcementId"])


def _amount(fact):
    value, unit, half = (float(fact[k]) for k in ("value", "unit", "half_unit"))
    if not all(math.isfinite(x) for x in (value, unit, half)) or unit <= 0 or half < 0:
        raise ValueError("INVALID_DISCLOSED_AMOUNT_OR_PRECISION")
    return value * unit, (value - half) * unit, (value + half) * unit


def _ratio(numerator, denominator):
    n, nlow, nhigh = numerator
    d, dlow, dhigh = denominator
    if dlow <= 0:
        raise ValueError("NONPOSITIVE_DENOMINATOR_NOT_COMPARABLE")
    bounds = [a / b * 100 for a, b in itertools.product((nlow, nhigh), (dlow, dhigh))]
    return {"value": n / d * 100, "low": min(bounds), "high": max(bounds)}


def derive_disclosure(record, root, as_of):
    publication_id = verify_publication(record, root)
    path = Path(record["document_path"])
    path = path if path.is_absolute() else Path(root) / path
    if digest(path) != record["document_sha256"]:
        raise ValueError("ORIGINAL_DISCLOSURE_DOCUMENT_CHANGED")
    if (
        not record.get("source_url")
        or not record.get("publication_evidence_url")
        or record.get("statement_scope") != "CONSOLIDATED"
        or record.get("roe_definition") != "PARENT_YTD_PROFIT_OVER_MEAN_PARENT_EQUITY"
        or record.get("revenue_definition") != "SINGLE_QUARTER_SAME_DISCLOSURE_COMPARABLE"
        or record.get("rounding_assumption") != "NEAREST_REPORTED_UNIT"
    ):
        raise ValueError("MISSING_OR_INCOMPATIBLE_DISCLOSURE_SEMANTICS")
    end = pd.Timestamp(record["report_period"])
    published = pd.Timestamp(record["published_date"])
    if end.month not in (3, 6, 9, 12) or end != end + pd.offsets.QuarterEnd(0) or published <= end:
        raise ValueError("INVALID_REPORT_OR_PUBLICATION_PERIOD")
    available = published + pd.offsets.Day(1)
    if pd.Timestamp(as_of) < available:
        raise ValueError("DISCLOSURE_NOT_YET_VISIBLE")
    facts = record["facts"]
    required = {
        "profit_parent_ytd",
        "equity_parent_begin",
        "equity_parent_end",
        "revenue_quarter",
        "revenue_quarter_comparable",
    }
    if set(facts) != required or {f["currency"] for f in facts.values()} != {record["currency"]}:
        raise ValueError("MISSING_FACT_OR_MIXED_CURRENCY")
    year_start = f"{end.year}0101"
    quarter_start = f"{end.year}{end.month - 2:02d}01"
    expected = {
        "profit_parent_ytd": f"{year_start}/{end.strftime('%Y%m%d')}",
        "equity_parent_begin": f"{end.year - 1}1231",
        "equity_parent_end": end.strftime("%Y%m%d"),
        "revenue_quarter": f"{quarter_start}/{end.strftime('%Y%m%d')}",
        "revenue_quarter_comparable": f"{end.year - 1}{end.month - 2:02d}01/{end.year - 1}{end.month:02d}{end.day:02d}",
    }
    for name, fact in facts.items():
        if fact["period"] != expected[name] or not fact.get("page") or int(fact["page"]) < 1:
            raise ValueError("WRONG_FISCAL_PERIOD_OR_MISSING_PAGE")
    amounts = {name: _amount(f) for name, f in facts.items()}
    a, b = amounts["equity_parent_begin"], amounts["equity_parent_end"]
    denominator = tuple((x + y) / 2 for x, y in zip(a, b, strict=True))
    roe = _ratio(amounts["profit_parent_ytd"], denominator)
    growth = _ratio(amounts["revenue_quarter"], amounts["revenue_quarter_comparable"])
    growth = {k: v - 100 for k, v in growth.items()}
    return {
        "symbol": record["symbol"],
        "report_period": record["report_period"],
        "available_from": available.strftime("%Y%m%d"),
        "roe": roe,
        "q_sales_yoy": growth,
        "source_document_sha256": record["document_sha256"],
        "publication_id": publication_id,
        "publication_index_sha256": record["publication_index"]["sha256"],
        "scope": "REVIEWED_DOCUMENT_FACTS_AND_EXPLICIT_DERIVATIONS_ONLY",
        "rounding_assumption": record["rounding_assumption"],
        "vendor_formula_universally_attested": False,
        "production_changed": False,
    }


def vendor_value_matches(interval, value, *, decimal_places=4):
    """Intersect disclosure precision with the vendor's displayed precision."""
    value = float(value)
    if not math.isfinite(value) or decimal_places not in range(9):
        return False
    half = 0.5 * 10 ** (-decimal_places)
    return value + half >= interval["low"] and value - half <= interval["high"]
