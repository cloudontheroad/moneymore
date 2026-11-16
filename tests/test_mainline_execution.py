import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moneymore.config import BacktestConfig
from moneymore.research.mainline_execution import (
    physical_open_plan,
    risk_window_with_halt_seed,
    run_native,
    validate_calibration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_month_start_calibration_can_use_previous_close_labels():
    cal = {"cutoff_exclusive": "20260101", "last_signal": "20251201", "last_label_end": "20251231"}
    validate_calibration(cal, "20251231", "20260105")
    with pytest.raises(ValueError, match="CALIBRATION"):
        validate_calibration({**cal, "last_label_end": "20260102"}, "20251231", "20260105")
    with pytest.raises(ValueError, match="CALIBRATION"):
        validate_calibration(cal, "20260202", "20260203")


def test_native_cost_reduces_unnecessary_turnover(tmp_path, monkeypatch):
    if not (ROOT / ".runtime/ashare-cvxportfolio-v1/Scripts/python.exe").exists():
        pytest.skip("isolated cvxportfolio environment not installed")
    request = {
        "symbols": ["A", "B"],
        "mu": [0.0003, 0.0],
        "weights": [0.4975, 0.4975, 0.005],
        "benchmark": [0.4975, 0.4975, 0.005],
        "industry_exposure": [[1.0], [1.0]],
        "industry_limits": [1.0],
        "risk_aversion": 1.0,
        "stock_exposure": 0.995,
        "upper_weights": [1.0, 1.0],
        "lower_weights": [0.0, 0.0],
        "turnover_limit": 0.5,
        "equity": 1e7,
        "signal_date": "20260105",
        "buy_rate": 0.00116,
        "sell_rate": 0.00166,
    }
    cache = tmp_path / "old/native/example/20260105"
    risk = run_native({**request, "include_cost": False}, np.eye(2) * 0.01, cache, ROOT)
    cost = run_native({**request, "include_cost": True}, np.eye(2) * 0.01, tmp_path / "cost", ROOT)
    assert risk["status"] == cost["status"] == "CONTINUOUS_PLAN_VERIFIED_NOT_EXECUTED"
    assert cost["one_way_turnover"] < 1e-4
    assert risk["one_way_turnover"] > cost["one_way_turnover"] + 0.001
    decision = tmp_path / "old/decisions/example/20260105.json"
    decision.parent.mkdir(parents=True)
    decision.write_text(json.dumps({"continuous": risk}), encoding="utf-8")

    def must_recompute(*args, **kwargs):
        raise AssertionError("cache_miss_recomputes")

    monkeypatch.setattr("moneymore.research.mainline_execution.subprocess.run", must_recompute)
    reused = run_native(
        {**request, "include_cost": False},
        np.eye(2) * 0.01,
        tmp_path / "reuse",
        ROOT,
        verified_cache_directory=cache,
    )
    assert reused["cache_hit"] and reused["weights"] == risk["weights"]
    with pytest.raises(AssertionError, match="cache_miss_recomputes"):
        run_native(
            {**request, "include_cost": False},
            np.eye(2) * 0.02,
            tmp_path / "changed",
            ROOT,
            verified_cache_directory=cache,
        )


def test_physical_plan_preserves_t1_and_star_minimum():
    config = BacktestConfig.from_yaml(ROOT / "configs/default.yaml").model_copy(
        update={"initial_cash": 100000, "max_position_weight": 1.0}
    )
    snapshot = {
        "positions": [{"symbol": "A", "quantity": 200, "available_quantity": 100}],
        "cash": 10000.0,
        "equity": 12000.0,
    }
    desired, cash, clipped = physical_open_plan(
        {"688001.SH": 100},
        snapshot,
        {"A": 10.0, "688001.SH": 10.0},
        {"A": True, "688001.SH": True},
        {"A": True},
        config,
    )
    assert desired == {"A": 100} and cash > 10000 and clipped


def test_untradable_order_never_becomes_a_fill():
    config = BacktestConfig.from_yaml(ROOT / "configs/default.yaml").model_copy(
        update={"initial_cash": 100000}
    )
    snapshot = {
        "positions": [{"symbol": "A", "quantity": 100, "available_quantity": 100}],
        "cash": 1000.0,
        "equity": 2000.0,
    }
    desired, cash, _ = physical_open_plan({}, snapshot, {"A": 10.0}, {}, {"A": False}, config)
    assert desired == {"A": 100} and cash == 1000.0


def test_integer_plan_matches_actual_paper_broker(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import research_ashare_paper_routes as engine
    from report_ashare_mainline_replay import audit_ledger

    from moneymore.execution.paper import PaperBroker

    config = BacktestConfig.from_yaml(ROOT / "configs/default.yaml").model_copy(
        update={"initial_cash": 100000, "slippage_bps": 20, "max_position_weight": 0.4}
    )
    broker = PaperBroker(tmp_path / "paper.sqlite3")
    broker.initialize_account(config.initial_cash, "isolated")
    opened = {"600001.SH": 15.0, "000001.SZ": 22.0, "688001.SH": 10.0}
    allow = dict.fromkeys(opened, True)
    desired = {"600001.SH": 2000, "000001.SZ": 2000, "688001.SH": 100}
    exported = []
    for day, target in (("20260105", desired), ("20260106", {"688001.SH": 3000})):
        broker.settle_available_positions("isolated", day)
        snap = broker.account_snapshot(opened, "isolated")
        plan, cash, _ = physical_open_plan(target, snap, opened, allow, allow, config)
        trades, _ = engine.execute(
            broker, "test", day, "20251231", plan, opened, allow, allow, config
        )
        exported.extend(trades)
        after = broker.account_snapshot(opened, "isolated")
        actual = {p["symbol"]: p["quantity"] for p in after["positions"] if p["quantity"] > 0}
        assert actual == plan
        assert after["cash"] == pytest.approx(cash, abs=0.001)
        assert broker.reconcile("isolated").matched
    frame = pd.DataFrame(exported)
    assert audit_ledger(broker.database, frame)["matched"]
    frame.loc[0, "fee"] += 100
    with pytest.raises(ValueError, match="FILL_EXPORT_VALUE_MISMATCH"):
        audit_ledger(broker.database, frame)


def test_calibration_never_reads_future_label_values():
    spec = importlib.util.spec_from_file_location(
        "mainline_summary", ROOT / "scripts/summarize_ashare_mainline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    days = pd.bdate_range("2024-01-01", periods=100)
    records = [
        {
            "date": day.strftime("%Y%m%d"),
            "symbol": str(i),
            "score": i,
            "label_20": i / 10000,
            "label_end_20": (day + pd.offsets.BDay(20)).strftime("%Y%m%d"),
        }
        for day in days
        for i in range(110)
    ]
    frame = pd.DataFrame(records)
    cutoff = days[90].strftime("%Y%m%d")
    a, _ = module.causal_calibration(frame, cutoff)
    frame.loc[frame.label_end_20.ge(cutoff), "label_20"] = 1e6
    b, _ = module.causal_calibration(frame, cutoff)
    assert a == b and a["last_label_end"] < cutoff


@pytest.mark.parametrize("allocator", ["topk", "cvx_cost"])
def test_missing_held_score_freezes_only_that_stock(tmp_path, monkeypatch, allocator):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import replay_ashare_mainline as runner

    planner = runner.Planner.__new__(runner.Planner)
    planner.output = tmp_path
    planner.solver_cache = None
    planner.contract = json.loads((ROOT / "configs/ashare_mainline_n3.json").read_bytes())
    planner.scores = {("ridge_full", "20260130"): pd.Series({"B": 1.0, "C": -1.0})}
    planner.calibration = {
        ("ridge_full", "202602"): {
            "cutoff_exclusive": "20260201",
            "last_label_end": "20260130",
            "last_signal": "20260101",
            "beta_conservative": 0.001,
        }
    }
    planner.industry_history = {"A": pd.DataFrame({"date": ["20260129"], "industry": ["bank"]})}
    attrs = pd.DataFrame(
        {"total_mv": [100.0, 100.0], "industry": ["bank", "industry"]}, index=["B", "C"]
    )
    covariance = pd.DataFrame(np.eye(3) * 0.01, index=list("ABC"), columns=list("ABC"))
    planner.risk = lambda signal: (covariance, attrs, {})
    requests = []

    def native(request, *unused, **kwargs):
        requests.append(request)
        # Verify the actual equality constraint; native feasibility is covered separately.
        i = request["symbols"].index("A")
        assert request["lower_weights"][i] == request["upper_weights"][i] == 0.01
        return {
            "status": "CONTINUOUS_PLAN_VERIFIED_NOT_EXECUTED",
            "weights": {"A": 0.01, "B": 0.04, "C": 0.04},
        }

    monkeypatch.setattr(runner, "run_native", native)
    snapshot = {
        "cash": 99000.0,
        "equity": 100000.0,
        "unsettled_stock_rights": {},
        "receivable_cash": 0.0,
        "positions": [{"symbol": "A", "quantity": 100, "available_quantity": 100}],
    }
    context = {
        "kind": "ridge_full--" + allocator,
        "index": 20,
        "day": "20260202",
        "signal": "20260130",
        "held": {"A": 100},
        "night_holdings": {"A": 100},
        "members": set("ABC"),
        "night": snapshot,
        "snapshot": snapshot,
        "last": dict.fromkeys("ABC", 10.0),
        "opened": dict.fromkeys("ABC", 10.0),
        "buy": {"B": True, "C": True},
        "sell": {},
        "actions": pd.DataFrame(columns=["symbol", "ex_date"]),
        "config": BacktestConfig.from_yaml(ROOT / "configs/default.yaml").model_copy(
            update={"max_position_weight": 0.05}
        ),
    }
    quantities, reason = planner(**context)
    assert reason is None and quantities["A"] == 100 and quantities["B"] > 0
    decision = json.loads((tmp_path / "decisions" / context["kind"] / "20260202.json").read_bytes())
    assert decision["frozen_symbols"] == ["A"]
    # Genuine missing covariance still blocks optional trades, not a fake exit.
    planner.risk = lambda signal: (covariance.loc[["B", "C"], ["B", "C"]], attrs, {})
    quantities, reason = planner(**context)
    assert reason == "HELD_RISK_HISTORY_MISSING" and quantities == {"A": 100}


def test_rolling_window_retains_causally_known_halt_seed():
    prices = pd.DataFrame(
        {"A": [10.0, np.nan, np.nan, 11.0, 99.0]},
        index=["20260128", "20260129", "20260130", "20260202", "20260203"],
    )
    proof = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "trade_date": ["20260129", "20260130"],
            "suspend_type": ["S", "S"],
            "suspend_timing": [None, None],
        }
    )
    window, records = risk_window_with_halt_seed(prices, proof, ["20260130", "20260202"])
    assert window.A.tolist() == [10.0, 11.0]
    assert [r["trade_date"] for r in records] == ["20260130"]
    prices.loc["20260203", "A"] = 1e6
    other, _ = risk_window_with_halt_seed(prices, proof, ["20260130", "20260202"])
    pd.testing.assert_frame_equal(window, other)
    unknown, _ = risk_window_with_halt_seed(prices, proof.iloc[1:], ["20260130", "20260202"])
    assert pd.isna(unknown.loc["20260130", "A"])


def test_pending_exit_never_reoptimizes_other_positions_off_schedule(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import replay_ashare_mainline as runner

    planner = runner.Planner.__new__(runner.Planner)
    planner.output = tmp_path
    planner.contract = {"rebalance_sessions": 20}

    def forbidden(signal):
        raise AssertionError("off_schedule_reoptimization")

    planner.risk = forbidden
    snapshot = {
        "cash": 98000.0,
        "equity": 100000.0,
        "positions": [{"symbol": s, "quantity": 100, "available_quantity": 100} for s in "AB"],
    }
    context = {
        "kind": "ridge_full--cvx_cost",
        "index": 21,
        "day": "20260203",
        "signal": "20260202",
        "held": {"A": 100, "B": 100},
        "night_holdings": {"A": 100, "B": 100},
        "members": {"B", "C"},
        "snapshot": snapshot,
        "opened": dict.fromkeys("ABC", 10.0),
        "buy": {"B": True, "C": True},
        "sell": {},
        "config": BacktestConfig.from_yaml(ROOT / "configs/default.yaml"),
    }
    plan, reason = planner(**context)
    assert plan == {"A": 100, "B": 100} and reason is None
    context.update(index=22, day="20260204", signal="20260203", sell={"A": True})
    plan, reason = planner(**context)
    assert plan == {"B": 100} and reason is None
    decision = json.loads((tmp_path / "decisions/ridge_full--cvx_cost/20260204.json").read_bytes())
    assert decision["status"] == "MANDATORY_EXITS_ONLY" and not decision["retry_next_session"]
