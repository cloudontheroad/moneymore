"""Repair a documented merger, replay its affected tail, and audit final HIST results.

Original model outputs, original replay journals and the 2017-2020 study stay immutable.
The merger is a security conversion, never a fabricated sell or cash adjustment.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from build_hist_extension import init_extended
from prepare_hist_extension import CONFIG, OUT
from prepare_hist_reproduction import CFG, ROOT, digest, write_json
from replay_hist_reproduction import EXECUTION, performance
from run_hist_extension import fingerprint

FINAL = OUT / "final"
EFFECTIVE = pd.Timestamp("2025-03-17")
PRIOR = pd.Timestamp("2025-03-14")
OLD, NEW = "SH600837", "SH601211"
ACTION = {
    "source": OLD, "destination": NEW, "actual_share_ratio": .62,
    "delisted": "2025-03-04", "tradable_from": "2025-03-17",
    "source_document": "https://static.cninfo.com.cn/finalpage/2025-03-14/1222789136.PDF",
    "source_document_published": "2025-03-14", "source_pages": [5, 27, 34],
    "rounding": "floor fractional new actual share; allocation lottery not simulated; <1 share conservative difference",
    "interim_mark": "untradeable merger entitlement carried at last pre-suspension mark until listing",
}


def convert(positions, old_factor, new_factor):
    """Convert actual shares, retaining total book value until the first market mark."""
    result = copy.deepcopy(positions)
    if OLD not in result:
        return result, None
    old = result.pop(OLD)
    actual = old["amount"] * old_factor
    rounded = round(actual)
    if abs(actual-rounded) > .1:
        raise ValueError(f"Unexpected fractional pre-conversion actual shares: {actual}")
    exact = rounded * ACTION["actual_share_ratio"]
    new_actual = int(np.floor(exact+1e-7))
    new_amount = new_actual / new_factor
    value = old["amount"] * old["price"]
    existing = result.get(NEW, {"amount": 0., "price": 0.})
    total_amount = existing["amount"]+new_amount
    book_value = existing["amount"]*existing["price"]+value
    result[NEW] = {"amount": total_amount, "price": book_value/total_amount,
                   "count_day": max(1, existing.get("count_day", 1)), "weight": 0.}
    event = {**ACTION, "old_adjusted_shares": old["amount"], "old_factor": old_factor,
             "old_actual_shares": rounded, "new_actual_shares": new_actual,
             "new_adjusted_shares": new_amount, "new_factor": new_factor,
             "fractional_share_not_allocated": exact-new_actual, "book_value_yuan": value,
             "cash_change": 0., "commission": 0.}
    return result, event


def audit_tail(daily, orders, holdings, account, event, bars):
    cash = account["cash"]
    quantity = {s: v["amount"] for s, v in account.items() if s != "cash"}
    if event:
        # Independently apply the published ratio, not the replay's saved new quantity.
        actual = round(quantity.pop(OLD)*event["old_factor"])
        quantity[NEW] = quantity.get(NEW, 0.) + np.floor(actual*.62+1e-7)/event["new_factor"]
    closes = bars["$close"].unstack("instrument").sort_index().ffill()
    grouped = {d: g for d, g in orders.groupby("date")}
    snapshots = {d: g.set_index("instrument").adjusted_shares for d, g in holdings.groupby("date")}
    equity_error = cash_error = holding_error = 0.
    fills = 0
    for date, row in daily.iterrows():
        for order in grouped.get(date, pd.DataFrame()).itertuples(index=False):
            if not order.value:
                assert order.filled_adjusted_shares == order.cost == 0
                continue
            buy = order.direction == 1
            assert order.cost == max(CFG["minimum_cost"], order.value*(CFG["buy_cost"] if buy else CFG["sell_cost"]))
            quote = bars.loc[(order.instrument, date)]
            assert quote["$volume"] > 0
            field = "$open" if order.lane.startswith("next_open") else "$close"
            assert abs(order.price_adjusted-quote[field]) < 1e-6
            assert abs(order.value-order.price_adjusted*order.filled_adjusted_shares) < .01
            quantity[order.instrument] = quantity.get(order.instrument, 0.) + order.filled_adjusted_shares*(1 if buy else -1)
            cash += (-order.value if buy else order.value)-order.cost
            assert cash >= -.01 and min(quantity.values()) > -1e-5
            fills += 1
        held = pd.Series({s: q for s, q in quantity.items() if abs(q) > 1e-5})
        marks = closes.loc[date].reindex(held.index)
        assert marks.notna().all()
        equity_error = max(equity_error, abs(cash+float((held*marks).sum())-row.account))
        cash_error = max(cash_error, abs(cash-row.cash))
        saved = snapshots[date]
        aligned = held.subtract(saved, fill_value=0.)
        holding_error = max(holding_error, float(aligned.abs().max()))
    assert equity_error < .1 and cash_error < .1 and holding_error < 1e-4
    return {"passed": True, "days": len(daily), "fills": fills,
            "max_equity_error_yuan": equity_error, "max_cash_error_yuan": cash_error,
            "max_holding_error_adjusted_shares": holding_error}


def replay_tail(seed, lane, predictions, bars):
    from hist_reproduction_exchange import CashSafeExchange
    from qlib.backtest import get_strategy_executor
    from qlib.backtest.backtest import backtest_loop
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase

    class Targets(WeightStrategyBase):
        def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
            if isinstance(score, pd.DataFrame):
                score = score.iloc[:, 0]
            chosen = score.dropna().sort_values(ascending=False, kind="stable").iloc[:30]
            return {str(s): 1./len(chosen) for s in chosen.index} if len(chosen) else {}

    class Journal(SimulatorExecutor):
        def __init__(self):
            super().__init__(time_per_step="day", generate_portfolio_metrics=True)
            self.journal = []

        def _collect_data(self, trade_decision, level=0):
            result, info = super()._collect_data(trade_decision, level)
            for order, value, cost, price in result:
                self.journal.append({"date": order.start_time, "instrument": order.stock_id,
                    "direction": int(order.direction), "requested_adjusted_shares": order.amount,
                    "filled_adjusted_shares": order.deal_amount, "price_adjusted": price,
                    "value": value, "cost": cost,
                    "cash_guard_overspend_prevented": getattr(order, "cash_guard_overspend", 0.)})
            return result, info

    original = OUT/"runs"/f"seed-{seed:02d}"/"replay"
    dest = FINAL/f"seed-{seed:02d}"/"replay"
    dest.mkdir(parents=True, exist_ok=True)
    old_daily = pd.read_parquet(original/f"{lane}-daily.parquet")
    old_holdings = pd.read_parquet(original/f"{lane}-holdings.parquet")
    old_orders = pd.read_parquet(original/f"{lane}-orders.parquet")
    prior = old_daily.loc[PRIOR]
    positions = {r.instrument: {"amount": r.adjusted_shares, "price": r.price, "count_day": 1, "weight": 0.}
                 for r in old_holdings.loc[old_holdings.date.eq(PRIOR)].itertuples(index=False)}
    account = {"cash": float(prior.cash), **copy.deepcopy(positions)}
    old_factor = float(bars.loc[OLD, "$factor"].loc[:PRIOR].dropna().iloc[-1])
    new_factor = float(bars.loc[(NEW, EFFECTIVE), "$factor"])
    positions, event = convert(positions, old_factor, new_factor)
    assert abs(account["cash"]+sum(v["amount"]*v["price"] for v in positions.values())-prior.account) < .1
    strategy = (Targets(signal=predictions, risk_degree=1.) if lane.endswith("top30_targets") else
                TopkDropoutStrategy(signal=predictions, topk=30, n_drop=30, risk_degree=.95))
    price = "open" if lane.startswith("next_open") else "close"
    limits = ("$open / Ref($close, 1) >= 1.095", "$open / Ref($close, 1) <= 0.905") if price == "open" else .095
    exchange = CashSafeExchange(freq="day", start_time="2025-03-13", end_time="2026-11-13",
        codes=sorted(set(predictions.index.get_level_values("instrument")) | set(positions)),
        deal_price=price, open_cost=CFG["buy_cost"], close_cost=CFG["sell_cost"], min_cost=5, limit_threshold=limits)
    executor = Journal()
    strategy, executor = get_strategy_executor(start_time=EFFECTIVE, end_time="2026-11-13",
        strategy=strategy, executor=executor, account={"cash": account["cash"], **positions},
        benchmark="SH000300", exchange_kwargs={"exchange": exchange})
    # Qlib's dictionary account otherwise treats cash alone as opening equity.
    executor.trade_account.init_cash = float(prior.account)
    portfolio, _ = backtest_loop(start_time=EFFECTIVE, end_time="2026-11-13", trade_strategy=strategy, trade_executor=executor)
    tail, held = portfolio["1day"]
    tail["net_return"] = tail["return"]-tail.cost
    assert np.allclose((1+tail.net_return).cumprod(), tail.account/prior.account, atol=1e-9)
    journal = pd.DataFrame(executor.journal)
    holding_rows = [{"date": d, "instrument": s, "adjusted_shares": p.get_stock_amount(s), "price": p.get_stock_price(s)}
                    for d, p in held.items() for s in p.get_stock_list()]
    tail_holdings = pd.DataFrame(holding_rows)
    check = audit_tail(tail, journal.assign(lane=lane), tail_holdings, account, event, bars)
    unchanged_error = None
    if event is None:
        unchanged_error = float((tail.account-old_daily.loc[tail.index, "account"]).abs().max())
        assert unchanged_error < .1, f"Restart altered unaffected account: {unchanged_error}"
    tail["total_cost"] += prior.total_cost
    tail["total_turnover"] += prior.total_turnover
    daily = pd.concat([old_daily.loc[old_daily.index < EFFECTIVE], tail])
    orders = pd.concat([old_orders.loc[old_orders.date < EFFECTIVE], journal], ignore_index=True)
    holdings = pd.concat([old_holdings.loc[old_holdings.date < EFFECTIVE], tail_holdings], ignore_index=True)
    assert len(daily) == 2394 and daily.index.is_unique
    assert np.allclose((1+daily.net_return).cumprod(), daily.account/CFG["initial_capital"], atol=1e-9)
    assert daily.cash.min() >= -1e-5
    assert abs(orders.cost.sum()-daily.total_cost.iloc[-1]) < .01
    retired = pd.read_parquet(OUT/"data/official/stock_basic-delisted-20261113.parquet")
    dates = {s.split(".")[1]+s[:6]: pd.Timestamp(d) for s, d in zip(retired.ts_code, retired.delist_date)
             if isinstance(d, str) and len(d) == 8}
    r = holdings.loc[holdings.instrument.isin(dates)].copy()
    r = r.loc[r.date.ge(r.instrument.map(dates))]
    pending = r.instrument.eq(OLD) & r.date.lt(EFFECTIVE)
    assert r.loc[~pending].empty, "Unhandled post-delisting holding"
    daily.to_parquet(dest/f"{lane}-daily.parquet")
    orders.to_parquet(dest/f"{lane}-orders.parquet", index=False)
    holdings.to_parquet(dest/f"{lane}-holdings.parquet", index=False)
    prefix_audit = json.loads((original/"independent-audit.json").read_text())[lane]
    result = {"full_2017_now": performance(daily.net_return),
              "since2020": performance(daily.loc["2020-01-01":, "net_return"]),
              "post2020": performance(daily.loc["2021-01-01":, "net_return"]),
              "year_metrics": {str(y): performance(g.net_return) for y, g in daily.groupby(daily.index.year)},
              "audit": {"prefix": prefix_audit, "corrected_tail": check, "passed": True},
              "cash_min": float(daily.cash.min()), "total_cost_yuan": float(orders.cost.sum()),
              "turnover_sum": float(daily.turnover.sum()), "event": event,
              "unchanged_restart_max_error_yuan": unchanged_error,
              "unhandled_retirement_rows": 0, "pending_merger_entitlement_rows": int(pending.sum()),
              "replay_daily_sha256": digest(dest/f"{lane}-daily.parquet")}
    write_json(dest/f"{lane}-result.json", result)
    print("CORRECTED", seed, lane, result["post2020"], flush=True)
    return result


def run_seed(seed):
    from qlib.data import D
    init_extended()
    raw = OUT/"runs"/f"seed-{seed:02d}"
    proof = json.loads((raw/"prediction.json").read_text())
    assert proof["execution_fingerprint"] == fingerprint()
    assert proof["prediction_sha256"] == digest(raw/"pred-full.parquet")
    predictions = pd.read_parquet(raw/"pred-full.parquet")
    symbols = sorted(set(predictions.index.get_level_values("instrument")) | {OLD, NEW})
    bars = D.features(symbols, ["$close", "$open", "$factor", "$volume"],
        start_time="2025-01-01", end_time="2026-11-13").sort_index()
    result = {lane: replay_tail(seed, lane, predictions, bars) for lane in EXECUTION["lanes"]}
    write_json(FINAL/f"seed-{seed:02d}"/"complete.json", {
        "runs": result, "frozen_execution_fingerprint": fingerprint(),
        "correction_sha256": digest(ROOT/"scripts/finalize_hist_extension.py"), "action": ACTION})


def report():
    raw = json.loads((OUT/"results.json").read_text())
    for path in (OUT/"benchmarks").glob("*.parquet"):
        b = pd.read_parquet(path).sort_values("trade_date")
        returns = pd.Series(b.close.to_numpy(), index=pd.to_datetime(b.trade_date)).pct_change(fill_method=None).iloc[1:]
        raw["benchmarks"][path.stem]["since2020"] = performance(returns.loc["2020-01-01":])
    runs = {}
    for seed in range(10):
        p = FINAL/f"seed-{seed:02d}"/"complete.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        assert data["frozen_execution_fingerprint"] == fingerprint()
        assert data["correction_sha256"] == digest(ROOT/"scripts/finalize_hist_extension.py")
        runs[str(seed)] = data["runs"]
    result = {"status": "complete" if len(runs) == 10 else "partial_not_final", "completed": len(runs),
              "cutoff": CONFIG["requested_end"], "runs": runs, "benchmarks": raw["benchmarks"],
              "windows": {}, "action": ACTION, "updated_at": datetime.now(UTC).isoformat()}
    for window in ["full_2017_now", "since2020", "post2020"]:
        result["windows"][window] = {}
        for lane in EXECUTION["lanes"]:
            rows = [r[lane][window] for r in runs.values()]
            if not rows:
                continue
            stats = {m: {"mean": float(np.mean([r[m] for r in rows])), "min": float(min(r[m] for r in rows)),
                         "max": float(max(r[m] for r in rows))} for m in ["cumulative_return", "annualized_return_252",
                    "annualized_volatility_252", "sharpe_rf0_252", "maximum_drawdown"]}
            stats["beating_ChiNext_TR"] = sum(r["cumulative_return"] > raw["benchmarks"]["ChiNext_total_return"][window]["cumulative_return"] for r in rows)
            result["windows"][window][lane] = stats
    write_json(FINAL/"results.json", result)
    return result


def workflow():
    FINAL.mkdir(exist_ok=True)
    def job(seed):
        dest = FINAL/f"seed-{seed:02d}"
        dest.mkdir(exist_ok=True)
        with (dest/"run.log").open("w", encoding="utf-8") as log:
            p = subprocess.Popen([sys.executable, "-u", __file__, "seed", "--seed", str(seed)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            return {"seed": seed, "exit_code": p.wait()}
    completed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(job, seed) for seed in range(10)]
        for future in as_completed(futures):
            completed.append(future.result())
            summary = report()
            write_json(FINAL/"workflow.json", {"status": "running", "completed": completed,
                "verified": summary["completed"], "updated_at": datetime.now(UTC).isoformat()})
            print(completed[-1], flush=True)
    summary = report()
    ok = summary["status"] == "complete" and all(x["exit_code"] == 0 for x in completed)
    write_json(FINAL/"workflow.json", {"status": "complete" if ok else "failed", "completed": completed,
        "verified": summary["completed"], "updated_at": datetime.now(UTC).isoformat()})
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["seed", "workflow", "report"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "seed":
        run_seed(args.seed)
    elif args.phase == "workflow":
        raise SystemExit(workflow())
    else:
        report()
