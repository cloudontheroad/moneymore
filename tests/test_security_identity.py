import pandas as pd
import pytest

from moneymore.research.security_identity import SecurityIdentity


def identity():
    return SecurityIdentity([{"old": "A", "new": "B", "effective": "20250217", "source": "issuer"}])


def test_boundary_and_stable_identity():
    i = identity()
    assert i.stable("B") == "A"
    assert i.tradable("B", "20250214") == "A"
    assert i.tradable("A", "20250217") == "B"
    assert i.holdings({"A": 100}, "20250217") == {"A": {"quantity": 100, "trading_symbol": "B"}}
    with pytest.raises(ValueError, match="DUPLICATE_POSITION"):
        i.holdings({"A": 100, "B": 100}, "20250217")
    target = i.execution_targets({"A": 0.15}, "20250217")
    assert target.iloc[0].to_dict() == {
        "security_id": "A",
        "trading_symbol": "B",
        "target_weight": 0.15,
    }
    assert i.execution_targets({"B": 0.15}, "20250214").trading_symbol.tolist() == ["A"]
    with pytest.raises(ValueError, match="DUPLICATE_TARGET"):
        i.execution_targets({"A": 0.1, "B": 0.1}, "20250217")


def test_backcast_alias_is_not_counted_twice_and_no_wrong_date_fallback():
    i = identity()
    f = pd.DataFrame(
        {
            "ts_code": ["A", "B", "B", "A"],
            "trade_date": ["20250214", "20250214", "20250217", "20250218"],
            "close": [10.0, 10.0, 11.0, 12.0],
        }
    )
    result = i.market_rows(f, "trade_date", ["close"])
    assert result.trade_date.tolist() == ["20250214", "20250217"]
    assert result.ts_code.tolist() == ["A", "A"]
    assert result.source_symbol.tolist() == ["A", "B"]
    f.loc[1, "close"] = 20
    with pytest.raises(ValueError, match="ALIAS_SOURCE"):
        i.market_rows(f, "trade_date", ["close"])


def test_industry_intervals_are_preserved():
    f = pd.DataFrame({"ts_code": ["B"], "in_date": ["20210730"], "out_date": ["20260820"]})
    result = identity().industry_rows(f)
    assert result.ts_code.iloc[0] == "A"
    assert result.in_date.iloc[0] == "20210730"
    assert result.out_date.iloc[0] == "20260820"


def test_ambiguous_chain_must_not_be_guessed():
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        SecurityIdentity(
            [
                {"old": "A", "new": "B", "effective": "20250101", "source": "issuer"},
                {"old": "B", "new": "C", "effective": "20260101", "source": "issuer"},
            ]
        )


def test_financial_period_and_source_provenance_are_not_collapsed():
    f = pd.DataFrame(
        {
            "ts_code": ["A", "B", "A"],
            "ann_date": ["20241024"] * 3,
            "end_date": ["20240930"] * 3,
            "growth_source": ["official", "official", "supplement"],
            "roe": [1.0, 1.0, None],
            "q_sales_yoy": [2.0, 2.0, 2.0],
        }
    )
    out = identity().market_rows(
        f, "ann_date", ["roe", "q_sales_yoy"], record_keys=["ann_date", "end_date", "growth_source"]
    )
    assert len(out) == 2
    assert set(out.growth_source) == {"official", "supplement"}
    assert set(out.ts_code) == {"A"}
