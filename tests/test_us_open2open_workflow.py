import json
from pathlib import Path
import sys

import exchange_calendars as xcals
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_us_open2open import settled_days
from run_us_open2open import prior_universe, settle, write_record


def test_training_excludes_labels_that_settle_at_the_trade_open():
    dates = pd.date_range("2026-11-04", periods=3, freq="B", tz="UTC")
    frame = pd.DataFrame({"label_end": dates[1:]}, index=dates[:-1])
    assert settled_days(frame, dates[-1], 10).index.tolist() == [dates[0]]


def test_stock_pool_never_backfills_from_future(tmp_path):
    directory = tmp_path / "universe"
    directory.mkdir()
    snapshot = directory / "top1000-2026-11-05T100515Z.parquet"
    snapshot.touch()
    with pytest.raises(ValueError):
        prior_universe(tmp_path, pd.Timestamp("2026-11-04", tz="UTC"))
    assert prior_universe(tmp_path, pd.Timestamp("2026-11-06", tz="UTC")) == snapshot


def test_settlement_uses_dated_opens_and_is_idempotent(tmp_path):
    plan = {
        "trade_session": "2025-11-26", "exit_session": "2025-11-28", "status": "READY",
        "cost_bps": 20, "candidates": [{"symbol": "US.ABC", "rank": 1}],
    }
    plan_file = tmp_path / "open2open_v1/plans/2025-11-26.json"
    write_record(plan_file, plan)
    raw = tmp_path / "daily_yahoo_adjusted"
    raw.mkdir()
    pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-11-26 05:00Z", "2025-11-28 05:00Z", "2025-12-01 05:00Z"]),
        "open": [100.0, 103.0, 300.0], "close": [101.0, 95.0, 400.0],
        "cash_dividend": [0.0, 1.0, 0.0],
    }).to_parquet(raw / "US_ABC.parquet")
    cal = xcals.get_calendar("XNYS")
    assert settle(tmp_path, cal)["newly_settled"] == 1
    result = json.loads((tmp_path / "open2open_v1/settled/2025-11-26.json").read_text())
    assert result["top1_net_return"] == pytest.approx(0.038)
    assert result["pnl_usd"] == pytest.approx(380)
    assert result["daily_return"] == pytest.approx(0.0038)
    assert result["realized"][0]["shares"] == 100
    assert result["realized"][0]["entry_open"] == 100
    assert result["realized"][0]["exit_open"] == 103
    assert settle(tmp_path, cal)["newly_settled"] == 0
