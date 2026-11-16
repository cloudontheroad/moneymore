"""Isolated reproduction contracts; no production stores or live API calls."""
import ast
import collections
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_hist_reproduction import performance
from train_hist_reproduction import DailyData, average_params


def test_compound_and_arithmetic_are_not_confused():
    result = performance(pd.Series([.1, -.1], index=pd.to_datetime(["2020-01-02", "2020-01-03"])))
    assert result["cumulative_return"] == pytest.approx(-.01)
    assert result["arithmetic_return_sum_not_compounded"] == pytest.approx(0.)
    assert result["maximum_drawdown"] == pytest.approx(-.1)


def test_first_day_drawdown_uses_initial_capital():
    result = performance(pd.Series([-.1, .02], index=pd.to_datetime(["2020-01-02", "2020-01-03"])))
    assert result["maximum_drawdown"] == pytest.approx(-.1)


def test_invalid_returns_rejected():
    for values in ([np.nan], [-1.], [np.inf], []):
        with pytest.raises(ValueError):
            performance(pd.Series(values, dtype=float))


def test_averaging_is_author_equivalent():
    source = ROOT / "external_research/HIST-upstream/learn.py"
    if not source.exists():
        pytest.skip("Upstream reproduction checkout not installed")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "average_params")
    namespace = {"collections": collections}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 -- single inspected pure function from pinned upstream
    for count in [1, 2, 5]:
        states = deque([{"weight": torch.tensor([float(i), i+.3])} for i in range(count)], maxlen=5)
        assert torch.equal(average_params(states)["weight"], namespace["average_params"](states)["weight"])


def test_loader_has_identical_day_boundaries_and_values(tmp_path, monkeypatch):
    import train_hist_reproduction as module
    monkeypatch.setattr(module, "OUT", tmp_path)
    folder = tmp_path / "train"
    folder.mkdir()
    rows = pd.DataFrame({"datetime": pd.to_datetime(["2010-01-04"]*2 + ["2010-01-05"]*3),
                         "instrument": ["A", "B", "A", "B", "C"]})
    rows.to_parquet(folder / "index.parquet", index=False)
    arrays = {"features": np.arange(1800, dtype=np.float32).reshape(5, 360),
              "labels": np.arange(5, dtype=np.float32), "market_value": np.ones(5, dtype=np.float32),
              "stock_index": np.array([1, 2, 1, 2, 3]), "counts": np.array([2, 3])}
    for name, values in arrays.items():
        np.save(folder / f"{name}.npy", values)
    data = DailyData("train", "cpu")
    assert data.slices == [slice(0, 2), slice(2, 5)]
    assert np.array_equal(data.x.numpy(), arrays["features"])
    assert data.sid.dtype == torch.long
    assert data.index.names == ["datetime", "instrument"]
    original = ROOT / "external_research/HIST-upstream/dataloader.py"
    if original.exists():
        namespace = {}
        # Upstream has invalid mixed indentation. expandtabs(8) only; no logic edits.
        exec(compile(original.read_text().expandtabs(8), str(original), "exec"), namespace)  # noqa: S102 -- inspected pinned loader
        reference = namespace["DataLoader"](
            pd.DataFrame(arrays["features"], index=data.index),
            pd.DataFrame(arrays["labels"], index=data.index), arrays["market_value"],
            arrays["stock_index"], batch_size=-1, pin_memory=True, device="cpu")
        assert [s for _, s in reference.iter_daily()] == data.slices
        for slc in data.slices:
            expected = reference.get(slc)
            for actual, target in zip((data.x[slc], data.y[slc], data.mv[slc], data.sid[slc]), expected[:4], strict=True):
                assert torch.equal(actual, target)


def test_protocol_retains_author_executable_hyperparameters():
    from prepare_hist_reproduction import CFG
    assert CFG["train"] == ["2007-01-01", "2014-12-31"]
    assert CFG["valid"] == ["2015-01-01", "2016-12-31"]
    assert CFG["test"] == ["2017-01-01", "2020-12-31"]
    assert CFG["hidden_size_actual"] == 64 and CFG["K"] == 1
    assert CFG["early_stop"] == 30 and CFG["smooth_steps"] == 5
    assert CFG["learning_rate"] == .0002 and CFG["epochs_max"] == 200
    assert CFG["label"] == "Ref($close, -1) / $close - 1"
    assert CFG["topk"] == 30 and CFG["buy_cost"] == .0005 and CFG["sell_cost"] == .0015


def test_execution_timings_are_predeclared_not_test_selected():
    from replay_hist_reproduction import EXECUTION
    assert EXECUTION["frozen_before_test_predictions"] is True
    assert EXECUTION["paper_execution_identity_certified"] is False
    assert EXECUTION["lanes"] == ["next_open_top30_targets", "paper_top30_targets", "qlib_top30_dropout30"]


def test_cash_guard_reproduces_and_fixes_seed1_negative_cash():
    from hist_reproduction_exchange import cash_safe_buy
    factor = 0.057189665734767914
    cash, price = 3138048.6713427603, 1.1655253171920776
    raw_affordable = cash / 1.0005 / price * factor
    original = (raw_affordable + .1) // 100 * 100 / factor
    assert cash - original*price*1.0005 == pytest.approx(-1.3798908, abs=1e-6)
    qty, value, fee = cash_safe_buy(original, price, cash, .0005, 5., 100/factor)
    assert cash >= value+fee
    assert (original-qty)*factor == pytest.approx(100.)
    assert fee == pytest.approx(value*.0005)


@pytest.mark.parametrize("cash,amount,price,rate,minimum,lot", [
    (1005., 100., 10., .0005, 5., 100.),
    (1004.99, 100., 10., .0005, 5., 100.),
    (4.99, 100., 10., .0005, 5., 100.),
    (1000000., 100., 10., .0005, 5., 100.),
    (999.9, 100., 10., .0005, 5., None),
])
def test_cash_guard_retains_fees_lots_and_affordable_orders(cash, amount, price, rate, minimum, lot):
    from hist_reproduction_exchange import cash_safe_buy
    qty, value, fee = cash_safe_buy(amount, price, cash, rate, minimum, lot)
    assert 0 <= qty <= amount and value+fee <= cash
    assert fee == (max(minimum, value*rate) if qty else 0.)
    if lot:
        assert qty/lot == pytest.approx(round(qty/lot))
    if amount*price+max(minimum, amount*price*rate) <= cash:
        assert qty == amount


def test_cash_guard_budget_boundaries_randomized():
    from hist_reproduction_exchange import cash_safe_buy
    rng = np.random.default_rng(0)
    for _ in range(1000):
        factor, price = 10**rng.uniform(-2, 1), 10**rng.uniform(-1, 3)
        lots = int(rng.integers(1, 10000))
        amount = lots*100/factor
        value = amount*price
        cash = max(0., value+max(5., value*.0005)+rng.uniform(-10, 10))
        qty, value, fee = cash_safe_buy(amount, price, cash, .0005, 5., 100/factor)
        assert qty <= amount+1e-7 and value+fee <= cash
        assert qty*factor/100 == pytest.approx(round(qty*factor/100))


def test_queue_does_not_call_failed_work_complete():
    from prepare_hist_reproduction import CFG
    from run_hist_reproduction import terminal_status
    results = [{"seed": seed, "status": "complete"} for seed in CFG["seeds"]]
    assert terminal_status(results, []) == "complete"
    results[-1]["status"] = "failed"
    assert terminal_status(results, []) == "failed_requires_attention"
    assert terminal_status(results[:2], [2]) == "running"
    assert terminal_status(results[:2], []) == "failed_requires_attention"


def test_retries_are_bounded_to_transient_failures():
    from run_hist_reproduction import retryable_failure
    assert retryable_failure("PermissionError: [WinError 32]")
    assert retryable_failure("TimeoutError: temporary busy")
    assert not retryable_failure("AssertionError: cash below zero")
    assert not retryable_failure("PermissionError in history; ValueError: Cash solvency failed")
    assert not retryable_failure("Unknown deterministic error")


def test_completed_seed_skips_all_process_launches(tmp_path, monkeypatch):
    import run_hist_reproduction as module
    monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(module, "training_complete", lambda folder, seed: True)
    monkeypatch.setattr(module, "replay_complete", lambda folder: True)

    def unexpected(*args, **kwargs):
        pytest.fail("Completed seed launched a worker")

    monkeypatch.setattr(module.subprocess, "Popen", unexpected)
    assert module.seed_job(1) == {"seed": 1, "status": "complete"}


def test_exchange_guard_operates_before_ledger_update():
    from types import SimpleNamespace

    from hist_reproduction_exchange import CashSafeExchange
    from qlib.backtest.decision import Order
    from qlib.backtest.exchange import Exchange

    cash = 3138048.6713427603
    exchange = CashSafeExchange.__new__(CashSafeExchange)
    exchange.impact_cost, exchange.open_cost, exchange.close_cost = 0., .0005, .0015
    exchange.min_cost, exchange.trade_unit, exchange.trade_w_adj_price = 5., 100, False
    exchange.logger = SimpleNamespace(debug=lambda *args: None)
    exchange.get_deal_price = lambda *args, **kwargs: 1.1655253171920776
    exchange.get_volume = lambda *args: 1e8
    exchange.get_factor = lambda *args: .057189665734767914
    exchange._clip_amount_by_volume = lambda *args: None
    position = SimpleNamespace(get_cash=lambda: cash)
    day = pd.Timestamp("2017-04-18")
    order = Order("SH601877", 3049501.9993441785, Order.BUY, day, day)
    _, value, fee = Exchange._calc_trade_info_by_order(exchange, order, position, {})
    assert value+fee > cash
    _, value, fee = exchange._calc_trade_info_by_order(order, position, {})
    assert value+fee <= cash
    assert order.cash_guard_overspend == pytest.approx(1.3798908, abs=1e-6)
    assert position.get_cash() == cash  # Guard sizes a fill; it does not alter the ledger.


def test_worker_retries_transient_failure_at_most_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import run_hist_reproduction as module
    monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(module, "training_complete", lambda folder, seed: True)
    monkeypatch.setattr(module, "replay_complete", lambda folder: False)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    launched = []

    def temporary_failure(*args, **kwargs):
        launched.append(args)
        kwargs["stdout"].write("PermissionError: [WinError 32]\n")
        return SimpleNamespace(pid=123, wait=lambda: 1)

    monkeypatch.setattr(module.subprocess, "Popen", temporary_failure)
    result = module.seed_job(2)
    assert len(launched) == 2
    assert result["status"] == "failed" and result["attempts"] == 2


def test_concurrent_reports_share_lock_and_aliases(tmp_path, monkeypatch):
    import json
    from concurrent.futures import ThreadPoolExecutor

    import replay_hist_reproduction as module
    monkeypatch.setattr(module, "OUT", tmp_path)
    monkeypatch.setattr(module, "replay_complete", lambda folder: False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(lambda _: module.report_results(), range(8)))
    assert all(row["status"] == "partial_not_final" for row in reports)
    assert (tmp_path / "results.json").read_bytes() == (tmp_path / "live-results.json").read_bytes()
    assert json.loads((tmp_path / "results.json").read_text())["completed_backtests"] == 0
    assert "0 / 10" in (tmp_path / "HIST_RESULTS.md").read_text(encoding="utf-8")
