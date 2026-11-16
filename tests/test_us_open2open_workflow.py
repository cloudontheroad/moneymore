import json
from pathlib import Path
import sys

import exchange_calendars as xcals
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_us_open2open import settled_days
from run_us_open2open import prior_universe, reprice_settled, settle, write_record
from us_open2open_premarket import rerank
import us_open2open_premarket
from moneymore.us_overnight.config import USOvernightConfig


def test_premarket_reranking_requires_coverage_and_can_replace_a_stock():
    pool = [{"symbol": f"US.TEST{i}", "score": 50 - i} for i in range(50)]
    quotes = [{"symbol": row["symbol"], "gap": i / 100} for i, row in enumerate(pool)]
    assert rerank(pool, quotes[:34], 10) == []
    ranked = rerank(pool, quotes, 10)
    assert len({row["symbol"] for row in ranked}) == 10
    # A strong gap on the first stock outside the baseline can move it in.
    quotes[10]["gap"] = 1.0
    ranked = rerank(pool, quotes, 10)
    assert "US.TEST10" in {row["symbol"] for row in ranked}


@pytest.mark.parametrize(("replace", "expected"), [(False, "UNCHANGED"), (True, "CHANGED")])
def test_premarket_review_keeps_baseline_and_only_opens_second_book_on_replacement(
    tmp_path, monkeypatch, replace, expected
):
    pool = [{"symbol": f"US.TEST{i}", "score": 50 - i, "rank": i + 1} for i in range(50)]
    filename = "2026-11-10.json"
    baseline = tmp_path / "open2open_v1/plans" / filename
    write_record(baseline, {
        "status": "READY", "trade_session": "2026-11-10", "exit_session": "2026-11-11",
        "generated_at": "2026-11-09T22:00:00Z", "prior_close_at": "2026-11-09T21:00:00Z",
        "candidates": pool[:10], "review_pool": pool, "cost_bps": 20,
    })
    original = baseline.read_bytes()
    monkeypatch.setattr(us_open2open_premarket, "capture_events", lambda *_: {})
    monkeypatch.setattr(us_open2open_premarket, "capture_stock", lambda row, *_: {
        "symbol": row["symbol"], "gap": 100 if replace and row["rank"] == 11 else row["score"]
    })
    now = lambda **_: pd.Timestamp("2026-11-10T14:21:00Z")
    result = us_open2open_premarket.review(tmp_path, USOvernightConfig(), xcals.get_calendar("XNYS"), clock=now)
    assert baseline.read_bytes() == original
    assert result["review_status"] == expected
    assert (result["status"] == "READY") is replace
    assert len(result["candidates"]) == 10
    assert result["added"] == (["US.TEST10"] if replace else [])


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
    assert result["fee_model"] == "futu_hk_us_fixed"
    assert result["realized"][0]["buy_fee_usd"] == 2.29
    assert result["realized"][0]["sell_fee_usd"] == 2.52
    assert result["cost_usd"] == 4.81
    assert result["top1_net_return"] == pytest.approx(0.039519)
    assert result["pnl_usd"] == pytest.approx(395.19)
    assert result["daily_return"] == pytest.approx(0.0039519)
    assert result["realized"][0]["shares"] == 100
    assert result["realized"][0]["entry_open"] == 100
    assert result["realized"][0]["exit_open"] == 103
    assert settle(tmp_path, cal)["newly_settled"] == 0
    comparison = tmp_path / "open2open_v1/premarket"
    write_record(comparison / "plans" / plan_file.name, plan | {"review_status": "CHANGED"})
    write_record(comparison / "plans/2025-11-28.json", {
        "status": "REVIEW_ONLY", "review_status": "UNCHANGED",
    })
    summary = settle(tmp_path, cal)
    assert summary["newly_settled"] == 0
    assert summary["premarket_newly_settled"] == 1
    assert (comparison / "settled" / plan_file.name).is_file()
    assert not (comparison / "settled/2025-11-28.json").exists()

    legacy = {key: value for key, value in result.items() if key != "fee_model"}
    legacy["pnl_usd"] = 380
    comparison_record = comparison / "settled" / plan_file.name
    write_record(comparison_record, legacy)
    assert len(reprice_settled(tmp_path)["repriced"]) == 1
    assert json.loads(comparison_record.read_text()) == result
    assert reprice_settled(tmp_path)["repriced"] == []
