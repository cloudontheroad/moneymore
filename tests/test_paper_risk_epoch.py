import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.execution.paper import PaperBroker
from moneymore.execution_replay import run_execution_replay
from moneymore.research.baseline_v2_continuous_replay import EMPTY
from moneymore.research.baseline_v2_release import digest
from moneymore.research.paper_risk_epoch import EpochAwareRiskPlans, PaperRiskEpochs


def synthetic_contract(root, account):
    directory = root / "state/baseline-v2-release/runs/SYNTHETIC_TEST_ONLY"
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(
        __import__("moneymore.research.paper_risk_epoch", fromlist=["__file__"]).__file__
    ).resolve()
    stages = [
        "attribution",
        "full_union_economics",
        "financial_vintages",
        "robustness",
        "execution_stress",
        "parity",
        "forward_observation",
    ]
    manifest = {
        "input_hashes": {str(source): digest(source)},
        "protocol": {
            "real_orders_enabled": False,
            "required_stages": stages,
            "forward_review_days": 60,
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    manifest_hash = digest(directory / "manifest.json")
    contract = directory / "provisional-paper-contract.json"
    contract.write_text(
        json.dumps(
            {
                "account_id": account,
                "manifest_sha256": manifest_hash,
                "paper_activation_authorized": True,
                "real_orders_enabled": False,
                "orders_enabled": False,
            }
        )
    )
    for stage in stages[:-1]:
        (directory / (stage + ".json")).write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "manifest_sha256": manifest_hash,
                    "artifact_hashes": {str(contract): digest(contract)}
                    if stage == "parity"
                    else {},
                }
            )
        )
    return {"release_directory": str(directory), "contract_sha256": digest(contract)}


def setup_account(tmp_path):
    database = tmp_path / "state/baseline-v2-release/paper-epochs/test.sqlite3"
    broker = PaperBroker(database)
    account = "baseline_v2:test"
    broker.initialize_account(100000, account)
    state = broker.update_risk_state(account, "SELL_ONLY", "DRAWDOWN_LIMIT", "20261029")
    now = datetime(2026, 10, 29, 2, tzinfo=UTC)
    request = {
        "approval_id": "approval-1",
        "account_id": account,
        "confirm_new_risk_budget": True,
        "real_orders_enabled": False,
        "observation_contract_authorized": True,
        **synthetic_contract(tmp_path, account),
        "operator": "LOCAL_TEST_USER",
        "reason": "Explicit new paper risk budget",
        "observed_at": now.isoformat(),
        "data_health": "PASS",
        "expected_risk_updated_at": state["updated_at"],
        "expected_cash": 100000,
        "expected_epoch": 0,
        "lifetime_peak_before_recovery": 120000,
    }
    return broker, PaperRiskEpochs(tmp_path, database), request, now


def test_explicit_recovery_starts_new_epoch_not_new_lifetime_history(tmp_path):
    broker, epochs, request, now = setup_account(tmp_path)
    before = broker.account_snapshot({}, request["account_id"])
    receipt = epochs.approve(request, now=now)
    assert receipt["epoch"] == 1
    assert receipt["lifetime_drawdown_preserved"] == pytest.approx(-1 / 6)
    assert receipt["history_rewritten"] is False
    assert broker.account_snapshot({}, request["account_id"]) == before
    assert broker.risk_state(request["account_id"])["effective_state"] == "NORMAL"
    assert epochs.drawdown(request["account_id"], 100000) == 0
    assert epochs.drawdown(request["account_id"], 110000) == 0
    assert epochs.drawdown(request["account_id"], 99000) == pytest.approx(-0.1)
    restarted = PaperRiskEpochs(tmp_path, broker.database)
    assert restarted.drawdown(request["account_id"], 99000) == pytest.approx(-0.1)
    assert epochs.approve(request, now=now + timedelta(hours=1)) == receipt
    with pytest.raises(ValueError, match="APPROVAL_ID_REUSED"):
        epochs.approve({**request, "reason": "changed"}, now=now)


def test_lifetime_peak_survives_restart_and_new_risk_epoch(tmp_path):
    broker, epochs, request, now = setup_account(tmp_path)
    account = request["account_id"]
    assert epochs.lifetime_mark(account, 120000)["peak"] == 120000
    restarted = PaperRiskEpochs(tmp_path, broker.database)
    assert restarted.lifetime_mark(account, 100000)["drawdown"] == pytest.approx(-1 / 6)
    restarted.approve(request, now=now)
    assert restarted.lifetime_mark(account, 100000)["peak"] == 120000
    assert restarted.drawdown(account, 100000) == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"confirm_new_risk_budget": False}, "EXPLICIT_AUTHORIZED"),
        ({"observation_contract_authorized": False}, "EXPLICIT_AUTHORIZED"),
        ({"real_orders_enabled": True}, "EXPLICIT_AUTHORIZED"),
        ({"account_id": "old_baseline"}, "EXPLICIT_AUTHORIZED"),
        ({"observed_at": "2026-10-28T02:00:00+00:00"}, "STALE_OR_FUTURE"),
        ({"data_health": "BLOCKED"}, "RECOVERY_DATA"),
        ({"expected_risk_updated_at": "old"}, "RISK_STATE_CHANGED"),
        ({"expected_epoch": 5}, "RISK_EPOCH_CHANGED"),
        ({"expected_cash": 99999}, "ACCOUNT_VALUES"),
    ],
)
def test_failed_checks_do_not_unlock(tmp_path, change, match):
    broker, epochs, request, now = setup_account(tmp_path)
    with pytest.raises(ValueError, match=match):
        epochs.approve({**request, **change}, now=now)
    assert broker.risk_state(request["account_id"])["effective_state"] == "SELL_ONLY"
    assert epochs.drawdown(request["account_id"], 100000) is None


def test_pending_order_or_operational_suspension_cannot_be_bypassed(tmp_path):
    broker, epochs, request, now = setup_account(tmp_path)
    with sqlite3.connect(broker.database) as conn:
        conn.execute(
            "INSERT INTO orders VALUES ('pending', 'now', 'PENDING', 'test', 'A', 'BUY', 100, '20261028', 'TEST', ?)",
            (request["account_id"],),
        )
    with pytest.raises(ValueError, match="NO_PENDING_ORDERS"):
        epochs.approve(request, now=now)
    state = broker.update_risk_state(
        request["account_id"], "SUSPENDED", "DATA_QUALITY_BLOCKED", "20261029"
    )
    with pytest.raises(ValueError, match="ONLY_DRAWDOWN_STOP"):
        epochs.approve({**request, "expected_risk_updated_at": state["updated_at"]}, now=now)


def test_existing_production_database_cannot_be_targeted(tmp_path):
    database = tmp_path / "production.sqlite3"
    database.touch()
    with pytest.raises(ValueError, match="ONLY_EXISTING_ISOLATED"):
        PaperRiskEpochs(tmp_path, database)


def test_claimed_authorization_cannot_bypass_real_release_failure(tmp_path):
    broker, epochs, request, now = setup_account(tmp_path)
    stage = Path(request["release_directory"]) / "financial_vintages.json"
    payload = json.loads(stage.read_text())
    payload["status"] = "BLOCKED_ORIGINAL_VINTAGES_NOT_ATTESTED"
    stage.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="HARD_GATES_NOT_PASSED"):
        epochs.approve(request, now=now)
    assert broker.risk_state(request["account_id"])["effective_state"] == "SELL_ONLY"


def test_changed_contract_cannot_be_reauthorized_by_replacing_request_hash(tmp_path):
    broker, epochs, request, now = setup_account(tmp_path)
    path = Path(request["release_directory"]) / "provisional-paper-contract.json"
    payload = json.loads(path.read_text())
    payload["account_id"] = "baseline_v2:other"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="STAGE_EVIDENCE_CHANGED"):
        epochs.approve({**request, "contract_sha256": digest(path)}, now=now)
    assert broker.risk_state(request["account_id"])["effective_state"] == "SELL_ONLY"


def test_recovery_transaction_rolls_back_and_can_retry_after_storage_failure(tmp_path):
    broker, epochs, request, now = setup_account(tmp_path)
    with sqlite3.connect(broker.database) as conn:
        conn.execute("""CREATE TRIGGER fail_recovery BEFORE INSERT ON risk_state_transitions
                        WHEN NEW.transition_type='MANUAL_NEW_RISK_EPOCH'
                        BEGIN SELECT RAISE(ABORT, 'synthetic storage failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        epochs.approve(request, now=now)
    assert broker.risk_state(request["account_id"])["effective_state"] == "SELL_ONLY"
    assert epochs.drawdown(request["account_id"], 100000) is None
    with sqlite3.connect(broker.database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM v2_risk_epochs").fetchone()[0] == 0
        conn.execute("DROP TRIGGER fail_recovery")
    assert epochs.approve(request, now=now)["epoch"] == 1


def test_planner_consumes_approval_but_preserves_lifetime_risk_and_operational_halt(tmp_path):
    _broker, epochs, request, now = setup_account(tmp_path)
    planner = object.__new__(EpochAwareRiskPlans)
    planner.epoch_store = epochs
    planner.account_id = request["account_id"]
    planner.config = SimpleNamespace(max_drawdown=0.15, max_gross_exposure=1)
    planner.peak = 120000
    planner.input_hold = None
    snapshot = {"equity": 100000, "cash": 100000, "market_value": 0}
    locked = planner.evaluate_opening_risk("20261029", snapshot)
    assert locked["effective_state"] == "SELL_ONLY"
    assert locked["explicit_epoch_drawdown"] is None
    request["expected_risk_updated_at"] = locked["updated_at"]
    epochs.approve(request, now=now)
    recovered = planner.evaluate_opening_risk("20261029", snapshot)
    assert recovered["effective_state"] == "NORMAL"
    assert recovered["lifetime_drawdown_preserved"] == pytest.approx(-1 / 6)
    assert recovered["explicit_epoch_drawdown"] == 0
    planner.input_hold = {"fault": "missing_input"}
    assert planner.evaluate_opening_risk("20261029", snapshot)["effective_state"] == "SUSPENDED"


def test_synthetic_paper_fills_resume_only_after_explicit_approval(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=6)
    keys = dates.strftime("%Y%m%d").tolist()
    account = "baseline_v2:synthetic"
    db = tmp_path / "state/baseline-v2-release/paper-epochs/replay.sqlite3"
    prices = pd.DataFrame({"A": [10, 10, 8, 8, 8, 8], "B": [10] * 6}, index=dates)
    scores = pd.DataFrame(
        [{"date": d, "symbol": s, "score": 2 if s == "A" else 1} for d in dates for s in ("A", "B")]
    )
    pool = pd.DataFrame([{"signal_date": d, "security_id": s} for d in keys for s in ("A", "B")])
    policy = {
        "portfolio": {
            "top_k": 1,
            "exit_rank": 1,
            "max_replacements": 1,
            "gross_exposure": 1,
            "minimum_weight": 1,
            "maximum_weight": 1,
            "correlation_lookback": 40,
            "cluster_correlation_threshold": 0.99,
            "maximum_cluster_members": 1,
            "weighting_method": "rank_linear",
        }
    }
    config = BacktestConfig(
        initial_cash=100000,
        max_position_weight=1,
        max_drawdown=0.15,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=0,
        lot_size=100,
        max_gross_exposure=1,
    )
    runtime = {}

    def opening(date, snapshot, quotes):
        if not runtime:
            # Engine creates its fresh ledger before invoking the first open.
            runtime["broker"] = PaperBroker(db)
            runtime["epochs"] = PaperRiskEpochs(tmp_path, db)
            runtime["planner"] = EpochAwareRiskPlans(
                scores,
                pool,
                prices,
                policy,
                config,
                {"turnover_l1": 2, "solver_time_limit": 10, "solver_threads": None},
                keys,
                keys[-1],
                {"A", "B"},
                risk_database=db,
                account_id=account,
                epoch_store=runtime["epochs"],
            )
        return runtime["planner"].opening(date, snapshot, quotes)

    receipts = []

    def closing(date, snapshot):
        planner, epochs, broker = runtime["planner"], runtime["epochs"], runtime["broker"]
        target = planner.closing(date, snapshot)
        if date == keys[3]:
            # A scripted human approval is a test event, NOT an automatic policy.
            now = datetime.now(UTC)
            receipts.append(
                epochs.approve(
                    {
                        "approval_id": "synthetic-human-1",
                        "account_id": account,
                        "confirm_new_risk_budget": True,
                        "real_orders_enabled": False,
                        "observation_contract_authorized": True,
                        **synthetic_contract(tmp_path, account),
                        "operator": "SYNTHETIC_TEST_ONLY",
                        "reason": "Explicit human test event",
                        "observed_at": now.isoformat(),
                        "data_health": "PASS",
                        "expected_risk_updated_at": broker.risk_state(account)["updated_at"],
                        "expected_cash": snapshot["cash"],
                        "expected_epoch": 0,
                        "lifetime_peak_before_recovery": planner.peak,
                    },
                    now=now,
                )
            )
        return target

    bars = pd.DataFrame(
        [
            {
                "date": d,
                "symbol": s,
                "open": prices.loc[d, s],
                "close": prices.loc[d, s],
                "can_buy": True,
                "can_sell": True,
            }
            for d in dates
            for s in ("A", "B")
        ]
    )
    result = run_execution_replay(
        bars=bars,
        targets=EMPTY,
        config=config,
        database=db,
        strategy_id="synthetic",
        account_id=account,
        closing_target_provider=closing,
        opening_plan_provider=opening,
    )
    assert [(f["side"], f["trade_date"]) for f in result.fills] == [
        ("BUY", keys[1]),
        ("SELL", keys[2]),
        ("BUY", keys[4]),
    ]
    assert len(receipts) == 1
    assert receipts[0]["lifetime_drawdown_preserved"] < -0.15
    assert result.equity.iloc[-1].equity < config.initial_cash
