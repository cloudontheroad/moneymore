import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.candidate_lifecycle import (
    lifecycle_rows,
    review_candidate,
    run_lifecycle_review,
    trading_allowed,
)

POLICY = {
    "enabled": True,
    "effective_date": "20261026",
    "checkpoints": [20, 40, 60],
    "formal_review_days": 60,
    "review_expiry_trading_days": 10,
}
CANDIDATE = {
    "candidate_tag": "20260812",
    "account_id": "qlib_candidate_20260812",
    "research_gate_passed": True,
}


def histories(days=60, loss=0):
    dates = pd.bdate_range("2026-07-01", periods=days).strftime("%Y%m%d")
    returns = 0.002 + 0.001 * np.sin(np.arange(days))

    def frame(offset):
        return pd.DataFrame({"trade_date": dates, "equity": 1e6 * np.cumprod(1 + returns + offset)})

    return {"baseline": frame(0), "challenger": frame(0), "candidate": frame(loss)}


def review(frames, **kwargs):
    return review_candidate(
        CANDIDATE, frames, as_of="20261026", trustworthy_start="20260701", policy=POLICY, **kwargs
    )


def test_no_short_term_ranking_elimination():
    result = review(histories(19, -0.01))
    assert result["status"] == "OBSERVING"
    assert result["checkpoint"] == 0


def test_fixed_checkpoint_rejects_only_significant_dual_underperformance():
    result = review(histories(40, -0.01))
    assert result["status"] == "EARLY_REJECTED"
    assert result["checkpoint"] == 40
    assert set(result["comparisons"]) == {"baseline", "challenger"}
    assert result["comparisons"]["baseline"]["trial_count"] == 6


def test_one_bad_comparison_does_not_reject():
    frames = histories(20, -0.01)
    frames["challenger"] = frames["candidate"].copy()
    assert review(frames)["status"] == "CONTINUE_TO_60D"


def test_future_rows_excluded_and_common_dates_not_report_count():
    frames = histories(80)
    frames["candidate"] = frames["candidate"].iloc[::4]
    result = review(frames)
    assert result["common_observation_days"] == 20
    assert result["checkpoint"] == 20
    frames["candidate"]["trade_date"] = "20990101"
    assert review(frames)["common_observation_days"] == 0


def test_missing_trust_boundary_is_hold_not_rejection():
    result = review_candidate(
        CANDIDATE, histories(), as_of="20261026", trustworthy_start=None, policy=POLICY
    )
    assert result["status"] == "REVIEW_HOLD"


def test_formal_review_never_auto_promotes_and_expires():
    previous = review(histories(60))
    assert previous["status"] == "FORMAL_REVIEW_READY"
    assert previous["promotion_automatic"] is False
    frames = histories(100)
    previous["formal_review_date"] = frames["baseline"].iloc[60].trade_date
    assert review(frames, previous=previous)["status"] == "REVIEW_EXPIRED"


def test_terminal_and_newer_decisions_cannot_be_resurrected():
    previous = review(histories(20, -0.01))
    assert review(histories(60), previous=previous) == previous
    previous = {"status": "OBSERVING", "review_date": "20261027"}
    assert review(histories(), previous=previous) == previous


def test_persisted_journal_idempotent_readonly_catalog_and_execution_gate(tmp_path: Path):
    class Store:
        def read(self, table):
            frame = histories(20, -0.01)[
                {
                    "multi_sector_account_daily": "baseline",
                    "qlib_challenger_account_daily": "challenger",
                    "qlib_candidate_account_daily": "candidate",
                }[table]
            ].copy()
            frame["account_id"] = CANDIDATE["account_id"]
            return frame

    boundary = tmp_path / "state" / "point-in-time" / "latest.json"
    boundary.parent.mkdir(parents=True)
    boundary.write_text(
        json.dumps({"historical_replay_gate": {"earliest_trustworthy_date": "20260701"}})
    )
    assert lifecycle_rows(tmp_path) == {}
    assert trading_allowed(tmp_path, CANDIDATE["candidate_tag"], "20261026")
    first = run_lifecycle_review(tmp_path, Store(), [CANDIDATE], POLICY, "20261026")
    assert first == run_lifecycle_review(tmp_path, Store(), [CANDIDATE], POLICY, "20261026")
    assert not trading_allowed(tmp_path, CANDIDATE["candidate_tag"], "20261026")
    assert not trading_allowed(tmp_path, CANDIDATE["candidate_tag"], "20260815")


def test_before_effective_date_does_not_write_state(tmp_path):
    assert run_lifecycle_review(tmp_path, object(), [CANDIDATE], POLICY, "20261023") == {}
    assert not (tmp_path / "state" / "candidate-lifecycle.sqlite3").exists()


def test_research_gate_failure_is_terminal_even_without_market_data():
    result = review_candidate(
        {**CANDIDATE, "research_gate_passed": False},
        {},
        as_of="20261026",
        trustworthy_start=None,
        policy=POLICY,
    )
    assert result["status"] == "GATE_REJECTED"


def test_broker_blocks_submit_execution_and_restore_after_retirement(tmp_path):
    import sqlite3

    from moneymore.execution.paper import ExecutionBar, PaperBroker
    from moneymore.execution.risk import OrderIntent, RiskResult
    from moneymore.models import Side

    broker = PaperBroker(tmp_path / "state/paper_orders.sqlite3")
    account = CANDIDATE["account_id"]
    broker.initialize_account(100000, account)
    order = RiskResult(
        accepted=True,
        intent=OrderIntent(
            idempotency_key="retired-buy",
            strategy_id="test",
            symbol="600036.SH",
            side=Side.BUY,
            quantity=100,
            signal_date="20261023",
            reason_code="TEST",
        ),
        rejection_code=None,
    )
    assert broker.submit(order, account) == "PENDING"
    with sqlite3.connect(tmp_path / "state/candidate-lifecycle.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE candidate_state(candidate_tag TEXT PRIMARY KEY, payload TEXT)"
        )
        connection.execute(
            "INSERT INTO candidate_state VALUES (?, ?)",
            (CANDIDATE["candidate_tag"], json.dumps({"status": "GATE_REJECTED"})),
        )
    assert broker.submit(order, account) == "REJECTED_CANDIDATE_FROZEN"
    assert (
        broker.execute_pending(ExecutionBar("600036.SH", "20261026", 40, 40), object(), account)
        == []
    )
    assert broker.orders()[0]["status"] == "CANCELLED"
    assert broker.restore_cancelled(account, "20261023", "RECOVERY") == 0
    assert broker.fills(account) == []
    assert broker.reconcile(account).matched
