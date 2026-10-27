import pandas as pd

from moneymore.data.financial_reconciliation import (
    derive_quarterly_revenue_growth,
    reconcile_indicator_payload,
)
from moneymore.data.fundamental_research import (
    TABLE,
    financial_features_as_of,
    research_financial_frame,
)
from moneymore.data.store import ParquetStore


def indicator(value, flag=None, capture="20261027"):
    result = {
        "ts_code": "600001.SH",
        "ann_date": "20260820",
        "end_date": "20260630",
        "roe": 10.0,
        "q_sales_yoy": value,
        "_payload_capture": capture,
    }
    if flag is not None:
        result["update_flag"] = flag
    return result


def statement(period, revenue, ann="20260820", kind="1", comp="1"):
    return {
        "ts_code": "600001.SH",
        "ann_date": ann,
        "f_ann_date": ann,
        "end_date": period,
        "revenue": revenue,
        "report_type": kind,
        "comp_type": comp,
        "update_flag": "1",
    }


def test_null_duplicate_never_erases_value_regardless_order():
    data = pd.DataFrame([indicator(7.0), indicator(None)])
    for frame in [data, data.iloc[::-1]]:
        result = reconcile_indicator_payload(frame)
        assert result.iloc[0].q_sales_yoy == 7
        assert not result.iloc[0]["conflict__q_sales_yoy"]


def test_conflicting_current_values_block():
    result = reconcile_indicator_payload(pd.DataFrame([indicator(7, 1), indicator(8, 1)]))
    assert pd.isna(result.iloc[0].q_sales_yoy)
    assert result.iloc[0]["conflict__q_sales_yoy"]


def test_explicit_current_version_and_no_older_payload_borrowing():
    data = pd.DataFrame([indicator(7, 0), indicator(8, 1)])
    assert reconcile_indicator_payload(data).iloc[0].q_sales_yoy == 8
    data = pd.DataFrame([indicator(7, capture="20261026"), indicator(None, capture="20261027")])
    assert pd.isna(reconcile_indicator_payload(data).iloc[0].q_sales_yoy)


def test_conflict_cannot_hide_behind_previous_quarter():
    old = indicator(7)
    old["end_date"] = "20260331"
    old["ann_date"] = "20260420"
    current = indicator(None)
    current["conflict__q_sales_yoy"] = True
    f = financial_features_as_of(pd.DataFrame([old, current]), "20261027")
    assert pd.isna(f.iloc[0].q_sales_yoy)


def test_cumulative_quarter_difference_and_units():
    data = pd.DataFrame(
        [
            statement("20260630", 300),
            statement("20260331", 100),
            statement("20250630", 200),
            statement("20250331", 100),
        ]
    )
    result = derive_quarterly_revenue_growth(data, "20261027")
    row = result.loc[result.end_date.eq("20260630")].iloc[0]
    assert row.q_sales_yoy == 100  # percentage points, not decimal 1.0
    assert row.source_revenues == [300, 200, 100, 100]


def test_adjusted_single_quarter_uses_disclosed_revision_only():
    data = pd.DataFrame(
        [
            statement("20260630", 200, kind="2"),
            statement("20250630", 50, ann="20250820", kind="2"),
            statement("20250630", 100, ann="20260820", kind="3"),
        ]
    )
    original = derive_quarterly_revenue_growth(data, "20261027", single_quarter=True)
    adjusted = derive_quarterly_revenue_growth(
        data, "20261027", single_quarter=True, allow_adjusted=True
    )
    assert original.loc[original.end_date.eq("20260630")].iloc[0].q_sales_yoy == 300
    assert adjusted.loc[adjusted.end_date.eq("20260630")].iloc[0].q_sales_yoy == 100
    assert adjusted.loc[adjusted.end_date.eq("20260630")].iloc[0].source_report_types == ["2", "3"]


def test_publication_delay_excludes_today_and_no_parent_company_mix():
    data = pd.DataFrame(
        [
            statement("20260630", 200, ann="20261027", kind="2"),
            statement("20250630", 100, kind="2"),
            statement("20260630", 999, kind="7"),
        ]
    )
    result = derive_quarterly_revenue_growth(data, "20261027", single_quarter=True)
    assert "20260630" not in result.end_date.tolist()
    result = derive_quarterly_revenue_growth(data, "20261028", single_quarter=True)
    assert result.loc[result.end_date.eq("20260630")].iloc[0].q_sales_yoy == 100


def test_zero_comparison_base_stays_undefined():
    data = pd.DataFrame([statement("20260630", 200, kind="2"), statement("20250630", 0, kind="2")])
    result = derive_quarterly_revenue_growth(data, "20261027", single_quarter=True)
    row = result.loc[result.end_date.eq("20260630")].iloc[0]
    assert row.status == "NONPOSITIVE_COMPARISON_BASE"
    assert pd.isna(row.q_sales_yoy)


def test_company_type_mismatch_blocked():
    data = pd.DataFrame(
        [
            statement("20260630", 200, kind="2", comp="7"),
            statement("20250630", 100, kind="2", comp="2"),
        ]
    )
    result = derive_quarterly_revenue_growth(data, "20261027", single_quarter=True)
    assert result.loc[result.end_date.eq("20260630")].iloc[0].status == "COMPANY_TYPE_MISMATCH"


def test_approved_supplement_separate_table_and_pit(tmp_path):
    store = ParquetStore(tmp_path / "data")
    raw = pd.DataFrame([indicator(None)])
    store.merge_curated(TABLE, [raw], ["ts_code", "ann_date", "end_date"])
    before = (store.curated / (TABLE + ".parquet")).read_bytes()
    supplement = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "ann_date": "20261027",
                "end_date": "20260630",
                "q_sales_yoy": 17.0,
                "status": "AGREEMENT_OF_CUMULATIVE_AND_SINGLE_QUARTER",
            }
        ]
    )
    store.merge_curated(
        "quarterly_revenue_supplement_v2", [supplement], ["ts_code", "ann_date", "end_date"]
    )
    joined = research_financial_frame(store)
    assert pd.isna(financial_features_as_of(joined, "20261027").iloc[0].q_sales_yoy)
    result = financial_features_as_of(joined, "20261028").iloc[0]
    assert result.q_sales_yoy == 17
    assert result.growth_source == "TUSHARE_INCOME_CROSS_CHECKED_ARITHMETIC"
    assert (store.curated / (TABLE + ".parquet")).read_bytes() == before


def test_later_provider_value_wins_and_supplement_cannot_mask_conflict():
    provider = indicator(12)
    provider["growth_source"] = "TUSHARE_FINA_INDICATOR"
    supplement = {
        **indicator(17),
        "ann_date": "20260901",
        "roe": float("nan"),
        "growth_source": "TUSHARE_INCOME_CROSS_CHECKED_ARITHMETIC",
        "conflict__q_sales_yoy": False,
    }
    result = financial_features_as_of(pd.DataFrame([provider, supplement]), "20261027")
    assert result.iloc[0].q_sales_yoy == 12
    provider.update(q_sales_yoy=None, conflict__q_sales_yoy=True)
    result = financial_features_as_of(pd.DataFrame([provider, supplement]), "20261027")
    assert pd.isna(result.iloc[0].q_sales_yoy)
