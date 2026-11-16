from pathlib import Path

from moneymore.config import BacktestConfig
from moneymore.execution.paper import PaperBroker
from moneymore.execution.risk_state import (
    constrained_target_weight,
    evaluate_account_risk_state,
)


def _config() -> BacktestConfig:
    return BacktestConfig.from_yaml("configs/default.yaml")


def test_historical_outage_cannot_refreeze_latest_account(tmp_path):
    broker = PaperBroker(tmp_path / 'paper.sqlite3')
    broker.initialize_account(1_000_000,'a')
    broker.update_risk_state('a','NORMAL','RISK_CHECKS_PASS','20261110')
    old = broker.update_risk_state('a','SUSPENDED','DATA_QUALITY_BLOCKED','20261103')
    assert old['effective_state']=='NORMAL' and not old['recovery_required']
    current = broker.update_risk_state('a','SUSPENDED','DATA_QUALITY_BLOCKED','20261110')
    assert current['effective_state']=='SUSPENDED'
    broker.update_risk_state('a','NORMAL','RISK_CHECKS_PASS','20261111')
    assert broker.risk_state('a')['recovery_required']


def test_risk_state_rules_and_target_permissions() -> None:
    normal = evaluate_account_risk_state(
        _config(), 1_000_000, 500_000, 0.5, -0.02, "HEALTHY", True
    )
    warning = evaluate_account_risk_state(
        _config(), 1_000_000, 50_000, 0.75, -0.02, "HEALTHY", True
    )
    sell_only = evaluate_account_risk_state(
        _config(), 850_000, 200_000, 0.76, -0.15, "HEALTHY", True
    )
    suspended = evaluate_account_risk_state(
        _config(), 1_000_000, 500_000, 0.5, 0, "BLOCKED", True
    )

    assert normal.proposed_state == "NORMAL"
    assert warning.proposed_state == "NORMAL"
    assert sell_only.proposed_state == "SELL_ONLY"
    assert suspended.proposed_state == "SUSPENDED"
    assert constrained_target_weight("NORMAL", 0.08, 0.03) == 0.08
    assert constrained_target_weight("REDUCE_ONLY", 0.08, 0.03) == 0.03
    assert constrained_target_weight("SELL_ONLY", 0.08, 0.03) == 0
    assert constrained_target_weight("SUSPENDED", 0.08, 0.03) is None


def test_risk_state_escalates_automatically_and_recovers_manually(
    tmp_path: Path,
) -> None:
    broker = PaperBroker(tmp_path / "paper.sqlite3")
    broker.initialize_account(1_000_000, "multi")

    suspended = broker.update_risk_state(
        "multi", "SUSPENDED", "DATA_QUALITY_BLOCKED", "20260729"
    )
    pending_recovery = broker.update_risk_state(
        "multi", "NORMAL", "RISK_CHECKS_PASS", "20260730"
    )

    assert suspended["effective_state"] == "SUSPENDED"
    assert pending_recovery["effective_state"] == "SUSPENDED"
    assert pending_recovery["proposed_state"] == "NORMAL"
    assert pending_recovery["recovery_required"]
    recovered = broker.approve_risk_recovery("multi")
    assert recovered["effective_state"] == "NORMAL"
    assert not recovered["recovery_required"]
    assert broker.risk_transitions("multi")[0]["transition_type"] == "MANUAL_RECOVERY"
