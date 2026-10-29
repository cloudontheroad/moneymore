import pytest

from moneymore.research.disclosure_catalog import report_candidates


def item(title="公司2023年第三季度报告", identifier="1"):
    return {
        "secCode": "600036",
        "announcementTitle": title,
        "adjunctUrl": "finalpage/2023-10-28/" + identifier + ".PDF",
        "announcementId": identifier,
        "announcementTime": 1698422400000,
    }


def test_summary_excluded_full_versions_not_collapsed():
    result = report_candidates(
        "600036.SH",
        "20230930",
        "20231028",
        [
            item(),
            item("公司2023年第三季度报告摘要", "2"),
            item("公司2023年第三季度报告（修订后）", "3"),
        ],
    )
    assert result["status"] == "MULTIPLE_VERSIONS_REQUIRE_REVIEW"
    assert len(result["candidates"]) == 2
    assert result["later_correction_search_complete"] is False


def test_wrong_year_and_quarter_not_selected():
    result = report_candidates(
        "600036.SH",
        "20230930",
        "20231028",
        [item("公司2022年第三季度报告"), item("公司2023年第一季度报告", "2")],
    )
    assert result["candidates"] == []


def test_mismatched_publication_date_is_explicit():
    result = report_candidates("600036.SH", "20230930", "20231027", [item()])
    assert result["candidates"][0]["matches_dependency_publication"] is False


def test_unsafe_document_path_rejected():
    row = item()
    row["adjunctUrl"] = "../other.PDF"
    with pytest.raises(ValueError, match="DOCUMENT_PATH"):
        report_candidates("600036.SH", "20230930", "20231028", [row])


def test_official_short_quarter_title_and_annual_half_year_distinction():
    result = report_candidates("600036.SH", "20230930", "20231028", [item("2023年三季度报告")])
    assert len(result["candidates"]) == 1
    result = report_candidates("600036.SH", "20231231", "20231028", [item("2023年半年度报告")])
    assert result["candidates"] == []
