import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from moneymore.research.baseline_v2_release import digest
from moneymore.research.disclosure_financials import derive_disclosure, vendor_value_matches


@pytest.fixture
def record(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs/reviewed_financial_disclosures.yaml"
    value = deepcopy(yaml.safe_load(source.read_text(encoding="utf-8"))["records"][0])
    # This is an arithmetic/unit test fixture, not a synthetic original PDF proof.
    document = tmp_path / "synthetic-evidence.bin"
    document.write_bytes(b"SYNTHETIC_UNIT_TEST_ONLY")
    value.update(document_path=str(document), document_sha256=digest(document))
    index = tmp_path / "synthetic-index.json"
    index.write_text(
        json.dumps(
            {
                "response": {
                    "announcements": [
                        {
                            "announcementId": "1218188128",
                            "secCode": "600036",
                            "announcementTime": 1698422400000,
                            "adjunctUrl": "finalpage/2023-10-28/1218188128.PDF",
                            "adjunctType": "PDF",
                            "announcementTitle": value["announcement_title"],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    value["publication_index"].update(path=str(index), sha256=digest(index))
    return value


@pytest.mark.parametrize(
    "change",
    [
        {"published_date": "20231027"},
        {"symbol": "600000.SH"},
        {"source_url": "https://example.com/finalpage/2023-10-28/1218188128.PDF"},
        {"source_url": "https://static.cninfo.com.cn/finalpage/2023-10-28/999.PDF"},
    ],
)
def test_publication_must_match_actual_index(record, tmp_path, change):
    record.update(change)
    with pytest.raises(ValueError, match="PUBLICATION_IDENTITY"):
        derive_disclosure(record, tmp_path, "20240131")


def test_index_tamper_blocks(record, tmp_path):
    Path(record["publication_index"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="PUBLICATION_INDEX_CHANGED"):
        derive_disclosure(record, tmp_path, "20240131")


def test_bounded_collector_cache_is_supported_and_request_is_bound(record, tmp_path):
    path = Path(record["publication_index"]["path"])
    body = json.loads(path.read_text(encoding="utf-8"))["response"]
    envelope = {
        "data": body,
        "data_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "request": {
            "url": "https://www.cninfo.com.cn/new/hisAnnouncement/query",
            "data": {"stock": "600036,org"},
        },
    }
    path.write_text(json.dumps(envelope), encoding="utf-8")
    record["publication_index"].update(format="public_index_cache_v1", sha256=digest(path))
    assert derive_disclosure(record, tmp_path, "20240131")["publication_id"] == "1218188128"
    envelope["request"]["data"]["stock"] = "600000,org"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    record["publication_index"]["sha256"] = digest(path)
    with pytest.raises(ValueError, match="REQUEST_OR_CONTENT_MISMATCH"):
        derive_disclosure(record, tmp_path, "20240131")


def test_unknown_index_format_never_falls_back(record, tmp_path):
    record["publication_index"]["format"] = "latest_undated_vendor_value"
    with pytest.raises(ValueError, match="UNSUPPORTED_PUBLICATION_INDEX_FORMAT"):
        derive_disclosure(record, tmp_path, "20240131")


def test_original_facts_reproduce_both_features_not_weighted_annual_roe(record, tmp_path):
    result = derive_disclosure(record, tmp_path, "20240131")
    assert result["roe"]["value"] == pytest.approx(11.634856145204314)
    assert vendor_value_matches(result["roe"], 11.6349)
    assert vendor_value_matches(result["q_sales_yoy"], -4.5754)
    assert not vendor_value_matches(result["roe"], 17.38)
    assert result["vendor_formula_universally_attested"] is False
    assert result["available_from"] == "20231029"


def test_future_document_is_not_visible_on_announcement_day(record, tmp_path):
    with pytest.raises(ValueError, match="NOT_YET_VISIBLE"):
        derive_disclosure(record, tmp_path, "20231028")


@pytest.mark.parametrize(
    "field,change,match",
    [
        ("equity_parent_begin", {"period": "20230630"}, "WRONG_FISCAL_PERIOD"),
        ("revenue_quarter", {"period": "20230101/20230930"}, "WRONG_FISCAL_PERIOD"),
        ("revenue_quarter_comparable", {"currency": "USD"}, "MIXED_CURRENCY"),
        ("equity_parent_end", {"half_unit": -1}, "INVALID_DISCLOSED"),
        ("revenue_quarter", {"page": 0}, "MISSING_PAGE"),
    ],
)
def test_wrong_semantics_never_silently_calculate(record, tmp_path, field, change, match):
    record["facts"][field].update(change)
    with pytest.raises(ValueError, match=match):
        derive_disclosure(record, tmp_path, "20240131")


def test_unit_conversion_does_not_change_ratio(record, tmp_path):
    expected = derive_disclosure(record, tmp_path, "20240131")
    for fact in record["facts"].values():
        fact["value"] *= 1000000
        fact["half_unit"] *= 1000000
        fact["unit"] = 1
    assert derive_disclosure(record, tmp_path, "20240131")["roe"] == expected["roe"]


def test_changed_document_or_incompatible_definition_blocks(record, tmp_path):
    record["roe_definition"] = "WEIGHTED_ANNUALIZED"
    with pytest.raises(ValueError, match="INCOMPATIBLE_DISCLOSURE"):
        derive_disclosure(record, tmp_path, "20240131")
    Path(record["document_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="DOCUMENT_CHANGED"):
        derive_disclosure(record, tmp_path, "20240131")
