from datetime import UTC, datetime

import pandas as pd

from scripts.sync_us_top1000 import build_top_market_cap_universe


def test_top_market_cap_universe_filters_non_equity_and_is_stable():
    basic = pd.DataFrame(
        {
            "code": ["US.B", "US.A", "US.PINK", "US.BAD"],
            "name": ["B", "A", "P", "X"],
            "listing_date": ["2020-01-01"] * 4,
            "exchange_type": ["US_NYSE", "US_NASDAQ", "US_PINK", "US_NYSE"],
            "delisting": [False, False, False, True],
        }
    )
    snapshots = pd.DataFrame(
        {
            "code": basic.code,
            "total_market_val": [100, 100, 1000, 2000],
            "last_price": [10, 10, 10, 10],
            "issued_shares": [10, 10, 100, 200],
            "equity_valid": [True] * 4,
            "circular_market_val": [100] * 4,
            "sec_status": ["NORMAL"] * 4,
            "update_time": ["2026-11-05 16:00:00"] * 4,
            "name": basic.name,
        }
    )
    result = build_top_market_cap_universe(
        snapshots=snapshots, basic=basic, captured_at=datetime(2026, 11, 5, tzinfo=UTC), limit=2
    )
    assert result.code.tolist() == ["US.A", "US.B"]
    assert result["rank"].tolist() == [1, 2]
