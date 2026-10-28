import pandas as pd
import pytest

from moneymore.research.baseline_v2_daily_audit import audit_daily_inputs, coverage_rows


def test_daily_audit_blocks_partial_schedule(tmp_path):
    folder = tmp_path / "state/baseline-v2-validation/daily-inputs"
    folder.mkdir(parents=True)
    (folder / "readiness.json").write_text('{"missing": 1, "union_complete": false}')
    with pytest.raises(ValueError, match="EXACT_DAILY_SNAPSHOTS_INCOMPLETE"):
        audit_daily_inputs(tmp_path, None)


def test_coverage_alignment_and_exclusive_priority():
    top = pd.DataFrame({"ts_code": ["D", "C", "B", "A", "E"]})
    counts = pd.Series({"A": 253, "B": 252, "C": 253, "D": 0, "E": 253})
    out = coverage_rows(
        top,
        "20240131",
        "20240130",
        counts,
        253 - counts,
        counts * 0,
        {"A", "B", "D", "E"},
        {"A", "B", "C", "D"},
    )
    assert dict(zip(out.symbol, out.reason, strict=True)) == {
        "A": "ELIGIBLE",
        "B": "INCOMPLETE_PRICE_WINDOW_NOT_PROVEN_SUSPENSION_OR_IPO",
        "C": "REQUIRED_FINANCIAL_MISSING",
        "D": "NO_VALID_PRICE_HISTORY",
        "E": "HISTORICAL_INDUSTRY_UNKNOWN",
    }
    assert len(out) == len(top)
