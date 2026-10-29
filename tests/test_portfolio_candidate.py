from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution.paper import PaperBroker
from moneymore.portfolio_candidate import (
    execute_portfolio_candidate_open,
    prepare_portfolio_candidate,
)


@pytest.fixture
def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "moneymore.portfolio_candidate.trading_dates", lambda *a: ["20261027", "20261028"]
    )
    broker = PaperBroker(tmp_path / "state/paper_orders.sqlite3")
    account = "qlib_candidate_20261027"
    broker.initialize_account(10000, account)
    config = BacktestConfig(
        initial_cash=10000,
        commission_rate=0.00015,
        minimum_commission=5,
        stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_bps=2,
        lot_size=100,
        max_position_weight=1.0,
        max_gross_exposure=1.0,
        max_drawdown=0.2,
    )
    plan = {
        "signal_date": "20261027",
        "strategy_id": "test_forward",
        "target_weights": {"A": 0.5, "B": 0.5},
        "styles": {"A": {"style": -1.0}, "B": {"style": 1.0}},
        "style_limits": {"style": 2.0},
        "bootstrap": True,
    }
    write_diagnostic(tmp_path / "state/qlib-portfolio-plans/20261027/20261027.json", plan)
    now = datetime(2026, 10, 28, 9, 31, tzinfo=ZoneInfo("Asia/Shanghai"))

    class Provider:
        name = "fake"
        ready = True
        calls = 0

        def intraday_quotes(self, symbols):
            self.calls += 1
            return pd.DataFrame(
                [
                    {
                        "symbol": s,
                        "last": 10.0,
                        "timestamp": int(now.timestamp() * 1000),
                        "up_limit": 11.0,
                        "down_limit": 9.0,
                        "execution_ready": self.ready,
                    }
                    for s in symbols
                ]
            )

    provider = Provider()
    return dict(
        root=tmp_path,
        store=ParquetStore(tmp_path / "data"),
        broker=broker,
        config=config,
        candidate={"candidate_tag": "20261027", "account_id": account},
        provider=provider,
        trade_date="20261028",
        now=now,
    )


def test_forward_open_executes_and_freezes_budget_once(setup_case):
    args = setup_case
    result = execute_portfolio_candidate_open(**args)
    assert result["status"] == "COMPLETED"
    assert result["reconciliation"]["matched"]
    assert len(args["broker"].fills(args["candidate"]["account_id"])) == 2
    budget = args["root"] / "state/qlib-portfolio-budgets/20261027.json"
    content = budget.read_bytes()
    assert execute_portfolio_candidate_open(**args)["status"] == "COMPLETED"
    assert budget.read_bytes() == content
    assert args["provider"].calls == 1
    assert len(args["broker"].fills(args["candidate"]["account_id"])) == 2


def test_quote_quality_blocks_without_orders(setup_case):
    setup_case["provider"].ready = False
    assert execute_portfolio_candidate_open(**setup_case)["status"] == "QUOTE_QUALITY_BLOCKED"
    assert setup_case["broker"].orders() == []
    setup_case["provider"].ready = True
    assert execute_portfolio_candidate_open(**setup_case)["status"] == "COMPLETED"


def test_prepared_crash_receipt_never_blindly_repeats(setup_case):
    write_diagnostic(
        setup_case["root"] / "state/qlib-portfolio-execution/20261027/20261028.json",
        {"status": "PREPARED", "account_id": "qlib_candidate_20261027"},
    )
    assert execute_portfolio_candidate_open(**setup_case)["status"] == "PREPARED"
    assert setup_case["provider"].calls == 0
    assert setup_case["broker"].orders() == []


def test_off_session_does_not_request_quotes(setup_case):
    setup_case["now"] = datetime(2026, 10, 28, 18, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert execute_portfolio_candidate_open(**setup_case)["status"] == "OUTSIDE_EXECUTION_SESSION"
    assert setup_case["provider"].calls == 0


def test_factor_v2_uses_same_paper_executor_without_freezing_style_budget(setup_case):
    import json

    args = setup_case
    args["candidate"]["portfolio_variant"] = "factor_baseline_v2"
    path = args["root"] / "state/qlib-portfolio-plans/20261027/20261027.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan.update(freeze_budget=False, reason_code="FACTOR_BASELINE_V2_FORWARD_PAPER", created_at=args["now"].isoformat())
    write_diagnostic(path, plan)
    result = execute_portfolio_candidate_open(**args)
    assert result["status"] == "COMPLETED"
    assert result["reconciliation"]["matched"]
    assert not (args["root"] / "state/qlib-portfolio-budgets/20261027.json").exists()
    assert {o["reason_code"] for o in args["broker"].orders()} == {"FACTOR_BASELINE_V2_FORWARD_PAPER"}
    assert execute_portfolio_candidate_open(**args)["status"] == "COMPLETED"
    assert len(args["broker"].fills(args["candidate"]["account_id"])) == 2


def test_style_v2_frozen_contract_runs_paper_fill_and_reconciliation(setup_case):
    import hashlib
    import json
    from test_style_v2_experiment import contract

    args = setup_case
    policy = contract(args["root"])
    args["candidate"].update(portfolio_variant="factor_baseline_v2", factor_policy=policy)
    path = args["root"] / "state/qlib-portfolio-plans/20261027/20261027.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan.update(created_at=args["now"].isoformat(), real_orders_enabled=False, freeze_budget=False,
                policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest())
    write_diagnostic(path, plan)
    result = execute_portfolio_candidate_open(**args)
    assert result["status"] == "COMPLETED"
    assert result["reconciliation"]["matched"]
    assert len(args["broker"].fills(args["candidate"]["account_id"])) == 2
    assert execute_portfolio_candidate_open(**args)["status"] == "COMPLETED"
    assert len(args["broker"].fills(args["candidate"]["account_id"])) == 2


def test_style_v2_wrong_policy_plan_cannot_submit(setup_case):
    import json
    from test_style_v2_experiment import contract

    args = setup_case
    args["candidate"].update(portfolio_variant="factor_baseline_v2", factor_policy=contract(args["root"]))
    path = args["root"] / "state/qlib-portfolio-plans/20261027/20261027.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan.update(created_at=args["now"].isoformat(), real_orders_enabled=False, policy_sha256="wrong")
    write_diagnostic(path, plan)
    assert execute_portfolio_candidate_open(**args)["status"] == "EXPERIMENT_POLICY_OR_PAPER_SCOPE_CHANGED"
    assert args["broker"].orders() == []
    assert args["provider"].calls == 0


def test_new_v2_uses_10bps_in_optimizer_and_actual_paper_fills(setup_case):
    import json
    from moneymore.research.style_v2_experiment import execution_policy_hash

    args = setup_case
    original = args["config"].slippage_bps
    args["candidate"].update(portfolio_variant="factor_baseline_v2", execution_policy={"version": "test_10bps", "slippage_bps": 10, "real_orders_enabled": False})
    path = args["root"] / "state/qlib-portfolio-plans/20261027/20261027.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan.update(created_at=args["now"].isoformat(), freeze_budget=False,
                execution_slippage_bps=10, execution_policy_sha256=execution_policy_hash(args["candidate"]))
    write_diagnostic(path, plan)
    result = execute_portfolio_candidate_open(**args)
    assert result["status"] == "COMPLETED"
    assert result["execution_slippage_bps"] == 10
    assert result["reconciliation"]["matched"]
    assert all(f["price"] == pytest.approx(10.01) for f in args["broker"].fills(args["candidate"]["account_id"]))
    assert args["config"].slippage_bps == original


def test_new_v2_blocks_stale_2bps_plan_before_querying_quotes(setup_case):
    args = setup_case
    args["candidate"].update(portfolio_variant="factor_baseline_v2", execution_policy={"version": "test_10bps", "slippage_bps": 10, "real_orders_enabled": False})
    assert execute_portfolio_candidate_open(**args)["status"] == "EXECUTION_POLICY_PLAN_MISMATCH"
    assert args["provider"].calls == 0
    assert args["broker"].orders() == []


def test_factor_v2_respects_account_freeze_before_submitting(setup_case):
    import json
    args = setup_case
    args["candidate"]["portfolio_variant"] = "factor_baseline_v2"
    path = args["root"] / "state/qlib-portfolio-plans/20261027/20261027.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["created_at"] = args["now"].isoformat()
    write_diagnostic(path, plan)
    args["broker"].update_risk_state(args["candidate"]["account_id"], "SUSPENDED", "TEST", "20261028")
    result = execute_portfolio_candidate_open(**args)
    assert result["status"] == "ACCOUNT_RISK_BLOCKED"
    assert args["broker"].orders() == []


def test_completed_legacy_accounts_do_not_mask_new_derived_account(tmp_path):
    from moneymore.portfolio_candidate import derived_execution_complete
    candidate = {"candidate_tag": "20261028", "portfolio_variant": "factor_baseline_v2"}
    assert not derived_execution_complete(tmp_path, [candidate], "20261028")
    write_diagnostic(tmp_path / "state/qlib-portfolio-execution/20261028/20261028.json", {"status": "COMPLETED"})
    assert derived_execution_complete(tmp_path, [candidate], "20261028")
    assert derived_execution_complete(tmp_path, [candidate], "20261027")


def test_factor_v2_rejects_plan_without_real_creation_provenance(setup_case):
    setup_case["candidate"]["portfolio_variant"] = "factor_baseline_v2"
    assert execute_portfolio_candidate_open(**setup_case)["status"] == "PLAN_PROVENANCE_BLOCKED"
    assert setup_case["provider"].calls == 0


def test_weight_validation_hold_never_requests_quotes_or_submits(setup_case):
    setup_case["candidate"]["validation_hold"]=True
    assert execute_portfolio_candidate_open(**setup_case)["status"]=="WEIGHT_VALIDATION_HOLD"
    assert setup_case["provider"].calls==0
    assert setup_case["broker"].orders()==[]


def test_bootstrap_uses_fresh_risk_eligible_scores_not_stale_parent_holdings(
    setup_case, monkeypatch
):
    import json

    import numpy as np

    args = setup_case
    research = args["root"] / "research.json"
    write_diagnostic(
        research,
        {
            "protocol": {
                "portfolio_policy": {
                    "top_k": 2,
                    "exit_rank": 2,
                    "max_replacements": 2,
                    "minimum_weight": 0.4,
                    "maximum_weight": 0.6,
                    "correlation_penalty": 0.0,
                    "cluster_correlation_threshold": 0.9,
                    "maximum_cluster_members": 2,
                }
            }
        },
    )
    styles = pd.DataFrame({"style": [-1.0, 1.0]}, index=["A", "B"])
    risk = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (120, 2)), columns=["A", "B"])
    monkeypatch.setattr(
        "moneymore.portfolio_candidate.risk_snapshot",
        lambda *a: (risk, pd.Series({"A": 1.0, "B": 1.0}), styles),
    )
    result = prepare_portfolio_candidate(
        root=args["root"],
        store=args["store"],
        broker=args["broker"],
        config=args["config"],
        candidate={
            **args["candidate"],
            "research_path": str(research),
            "display_name": "test",
            "source_candidate_tag": "20261010",
        },
        trade_date="20261027",
        score_frame=pd.DataFrame(
            [
                {"instrument": "STALE", "score": 3.0},
                {"instrument": "A", "score": 2.0},
                {"instrument": "B", "score": 1.0},
            ]
        ),
        bootstrap_weights={"STALE": 1.0},
        marks={"A": 10.0, "B": 10.0},
        strategy_id="test",
        model_id="test",
    )
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert set(report["target_weights"]) == {"A", "B"}
    assert report["optimizer"]["ignored_stale_source_targets"] == ["STALE"]
    assert report["planned_trade_date"] == "20261028"
    assert args["broker"].orders() == []
    assert args["broker"].fills(args["candidate"]["account_id"]) == []
