"""Causal Top30 reconstruction. Upstream HIST contains no trading engine/code."""
import argparse
import json
import time

import numpy as np
import pandas as pd
from prepare_hist_reproduction import CFG, OUT, ROOT, digest, init_qlib, write_json

EXECUTION = json.loads((ROOT / "configs/hist_reproduction_execution_v1.json").read_text(encoding="utf-8"))
REPLAY_DIR = "replay-cash-safe-v2"


def execution_fingerprint():
    return {name: digest(ROOT / name) for name in (
        "configs/hist_reproduction_execution_v1.json", "scripts/replay_hist_reproduction.py",
        "scripts/hist_reproduction_exchange.py", "scripts/audit_hist_reproduction_replay.py")}


def replay_complete(folder):
    path = folder / REPLAY_DIR / "replay.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = execution_fingerprint()
    return set(result) == set(EXECUTION["lanes"]) and all(
        row.get("execution_fingerprint") == fingerprint and row.get("independent_ledger_audit", {}).get("passed")
        for row in result.values())


def performance(returns):
    returns = returns.astype(float)
    if returns.empty or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Invalid return stream")
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.)
    vol = returns.std(ddof=1) * np.sqrt(252)
    return {"days": len(returns), "cumulative_return": float(equity.iloc[-1]-1),
            "annualized_return_252": float(equity.iloc[-1]**(252/len(returns))-1),
            "annualized_volatility_252": float(vol),
            "sharpe_rf0_252": float(returns.mean()*252/vol) if vol else None,
            "maximum_drawdown": float((equity/peak-1).min()),
            "arithmetic_return_sum_not_compounded": float(returns.sum()),
            "yearly_return": {str(y): float((1+r).prod()-1) for y, r in returns.groupby(returns.index.year)}}


def fetch_benchmarks():
    from dotenv import load_dotenv

    from moneymore.data.tushare_provider import TushareProvider
    dest = OUT / "benchmarks"
    dest.mkdir(exist_ok=True)
    codes = {"000300.SH": "CSI300_price", "H00300.CSI": "CSI300_total_return",
             "399006.SZ": "ChiNext_price", "399606.SZ": "ChiNext_total_return"}
    client = None
    for code, name in codes.items():
        path = dest / f"{name}.parquet"
        if path.exists():
            continue
        if client is None:
            load_dotenv(ROOT / ".env", override=False)
            client = TushareProvider()._client
        # Exactly one bounded historical request per series. No credentials in reports.
        try:
            frame = client.index_daily(ts_code=code, start_date="20161230", end_date="20201231")
        except Exception as error:  # noqa: BLE001 -- redact provider exception bodies/credentials
            raise RuntimeError(f"Benchmark {code}: {type(error).__name__}") from None
        if frame is None or frame.empty:
            raise ValueError(f"Empty benchmark {code}; no substitution")
        frame = frame.sort_values("trade_date")
        assert frame.ts_code.eq(code).all() and not frame.trade_date.duplicated().any()
        assert str(frame.trade_date.iloc[0]) == "20161230" and str(frame.trade_date.iloc[-1]) == "20201231"
        frame.to_parquet(path, index=False)
        print("BENCHMARK", name, len(frame), flush=True)
        time.sleep(1.1)
    write_json(dest / "manifest.json", {"source": "Tushare index_daily", "symbols": codes,
                                        "sha256": {p.name: digest(p) for p in sorted(dest.glob("*.parquet"))}})


def replay(predictions, directory, start="2017-01-01", end="2020-12-31", smoke=False):
    from hist_reproduction_exchange import CashSafeExchange
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy, WeightStrategyBase

    class PaperTop30Targets(WeightStrategyBase):
        def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
            if isinstance(score, pd.DataFrame):
                score = score.iloc[:, 0]
            chosen = score.dropna().sort_values(ascending=False, kind="stable").iloc[:CFG["topk"]]
            return {str(code): 1./len(chosen) for code in chosen.index} if len(chosen) else {}

    class JournalExecutor(SimulatorExecutor):
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

    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for lane in EXECUTION["lanes"]:
        executor = JournalExecutor()
        if lane.endswith("top30_targets"):
            strategy = PaperTop30Targets(signal=predictions[["score"]], risk_degree=1.)
        else:
            strategy = TopkDropoutStrategy(signal=predictions[["score"]], topk=30, n_drop=30, risk_degree=.95)
        deal_price = "open" if lane.startswith("next_open") else "close"
        limits = ("$open / Ref($close, 1) >= 1.095", "$open / Ref($close, 1) <= 0.905") if deal_price == "open" else .095
        exchange = CashSafeExchange(freq="day", start_time="2016-12-30", end_time=end,
                codes=sorted(predictions.index.get_level_values("instrument").unique()),
                deal_price=deal_price, open_cost=CFG["buy_cost"], close_cost=CFG["sell_cost"],
                min_cost=CFG["minimum_cost"], limit_threshold=limits)
        portfolio, _ = backtest(start_time=start, end_time=end, strategy=strategy, executor=executor,
            account=CFG["initial_capital"], benchmark="SH000300", exchange_kwargs={"exchange": exchange})
        report, positions = portfolio["1day"]
        if not smoke:
            calendar = pd.to_datetime((OUT / "provider/calendars/day.txt").read_text().splitlines())
            expected = calendar[(calendar >= start) & (calendar <= end)]
            assert report.index.equals(expected), "Missing backtest valuation dates"
        net = report["return"] - report["cost"]
        assert np.allclose((1+net).cumprod().values, report.account.values/CFG["initial_capital"], atol=1e-9)
        journal = pd.DataFrame(executor.journal)
        if report.cash.min() < -1e-5:
            diagnostics = directory / "failed-diagnostics" / lane
            diagnostics.mkdir(parents=True, exist_ok=True)
            report.to_parquet(diagnostics / "daily.parquet")
            journal.to_parquet(diagnostics / "orders.parquet", index=False)
            detail = {"reason": "negative_cash", "minimum_cash": float(report.cash.min()),
                      "date": str(report.cash.idxmin()), "negative_days": int(report.cash.lt(-1e-5).sum())}
            write_json(diagnostics / "failure.json", detail)
            raise ValueError(f"Cash solvency failed: {detail}")
        assert not journal.empty
        assert abs(journal.cost.sum() - report.total_cost.iloc[-1]) < .01
        assert abs(journal.value.sum() - report.total_turnover.iloc[-1]) < .01
        traded = journal.loc[journal.value.gt(0)]
        first_signal = predictions.index.get_level_values("datetime").min()
        assert traded.date.min() > first_signal, "Same-day prediction leakage"
        holding_rows = []
        for day, position in positions.items():
            for code in position.get_stock_list():
                holding_rows.append({"date": day, "instrument": code,
                                     "adjusted_shares": position.get_stock_amount(code),
                                     "price": position.get_stock_price(code)})
        pd.DataFrame(holding_rows).to_parquet(directory / f"{lane}-holdings.parquet", index=False)
        report["net_return"] = net
        report.to_parquet(directory / f"{lane}-daily.parquet")
        journal.to_parquet(directory / f"{lane}-orders.parquet", index=False)
        result = performance(net)
        result.update({"cash_min": float(report.cash.min()), "total_cost_yuan": float(report.total_cost.iloc[-1]),
            "buy_sell_turnover_sum": float(report.turnover.sum()), "executed_orders": len(traded),
            "zero_fill_orders": len(journal)-len(traded), "average_invested_fraction": float((report.value/report.account).mean()),
            "equity_and_cost_reconciled": True, "first_fill": str(traded.date.min()),
            "smoke_test_not_model_result": smoke, "execution_price": deal_price,
            "factor_rounding_enabled": not executor.trade_exchange.trade_w_adj_price,
            "cash_guard_clipped_orders": int(journal.cash_guard_overspend_prevented.gt(0).sum()),
            "max_prevented_overspend_yuan": float(journal.cash_guard_overspend_prevented.max()),
            "execution_fingerprint": execution_fingerprint()})
        results[lane] = result
        print("REPLAY", lane, result, flush=True)
    from audit_hist_reproduction_replay import audit
    checks = audit(directory)
    for lane, check in checks.items():
        results[lane]["independent_ledger_audit"] = check
    write_json(directory / "replay.json", results)
    if not smoke:
        report_results()
    return results


def report_results(output_name="results.json"):
    from filelock import FileLock
    # One lock for child workers and coordinator, including the Markdown replace.
    with FileLock(str(OUT / "live-report.lock"), timeout=60):
        return _report_results(output_name)


def _report_results(output_name):
    benchmark = {}
    for path in (OUT / "benchmarks").glob("*.parquet"):
        frame = pd.read_parquet(path).sort_values("trade_date")
        close = pd.Series(frame.close.to_numpy(), index=pd.to_datetime(frame.trade_date))
        ret = close.pct_change(fill_method=None).iloc[1:]
        benchmark[path.stem] = performance(ret)
    runs = {}
    for seed in CFG["seeds"]:
        folder = OUT / "runs" / f"seed-{seed:02d}"
        if replay_complete(folder):
            runs[folder.name] = json.loads((folder / REPLAY_DIR / "replay.json").read_text(encoding="utf-8"))
    summary = {"completed_backtests": len(runs), "required_seeds": len(CFG["seeds"]),
               "status": "complete" if len(runs)==len(CFG["seeds"]) else "partial_not_final",
               "benchmarks": benchmark, "runs": runs, "source_model_only_exact": True,
               "backtest_reconstruction_not_author_released_code": True, "execution_protocol": EXECUTION,
               "replay_directory": REPLAY_DIR, "execution_fingerprint": execution_fingerprint()}
    for lane in EXECUTION["lanes"]:
        rows = [run[lane] for run in runs.values()]
        if rows:
            stats = {}
            for metric in ("cumulative_return", "annualized_return_252", "sharpe_rf0_252", "maximum_drawdown"):
                values = np.array([row[metric] for row in rows], dtype=float)
                stats[metric] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                 "min": float(values.min()), "max": float(values.max()), "std": float(values.std())}
            if "ChiNext_total_return" in benchmark:
                stats["seeds_beating_ChiNext_total_return"] = sum(row["cumulative_return"] > benchmark["ChiNext_total_return"]["cumulative_return"] for row in rows)
            summary[lane] = stats
    write_json(OUT / output_name, summary)
    for alias in ("results.json", "live-results.json"):
        if alias != output_name:
            write_json(OUT / alias, summary)
    lines = ["# HIST 公开实现复现：阶段结果", "",
             f"已完成并对账：{len(runs)} / {len(CFG['seeds'])} 个固定种子。状态：{summary['status']}。", "",
             "测试窗口2017—2020，训练2007—2014、验证2015—2016。作者模型源码未改，交易回放为独立重建；不是论文原始投资代码的逐字节复现。", "",
             "|策略/口径|累计净收益|年化收益|夏普（rf=0）|最大回撤|", "|---|---:|---:|---:|---:|"]
    names = {"next_open_top30_targets": "HIST次日开盘等权目标", "paper_top30_targets": "HIST次日收盘等权目标",
             "qlib_top30_dropout30": "HIST Qlib原生Top30/Drop30"}
    for lane in EXECUTION["lanes"]:
        if lane in summary:
            s = summary[lane]
            lines.append(f"|{names[lane]}（已完成种子均值）|{s['cumulative_return']['mean']:.2%}|{s['annualized_return_252']['mean']:.2%}|{s['sharpe_rf0_252']['mean']:.3f}|{s['maximum_drawdown']['mean']:.2%}|")
    for name, row in benchmark.items():
        lines.append(f"|{name}|{row['cumulative_return']:.2%}|{row['annualized_return_252']:.2%}|{row['sharpe_rf0_252']:.3f}|{row['maximum_drawdown']:.2%}|")
    lines += ["", "种子均值不是实际组合账户；各种子的完整指标、分年收益、费用、换手和独立账本校验见live-results.json。", "",
              "## 必须保留的边界", "",
              "- 镜像行情包缺VWAP；作者Fillna把60个对应特征置零。没有以其他价格冒充。",
              "- 作者静态概念矩阵的历史时点无法认证；DK_L按未来标签是否可用过滤样本。不能据此声称无穿越、实盘可部署。",
              "- 三种成交口径全部报告；不事后挑选收益最高者。交易信号晚于当日收盘数据形成，最早次日开盘执行。",
              "- 统一使用研究专用现金保护：修正Qlib整手取整0.1股容差造成的透支，只在无法支付成交额加费用时向下取整；不修改模型、目标仓位、费率或现金账本。旧回放保留，最终汇总不混用旧版本。",
              "- 开盘采用开盘涨跌幅保护，收盘沿用Qlib 9.5%保护；未穷尽历史ST/20%规则、冲击成本等真实执行细节。",
              "- 论文所称120%以上不是本机结果。当前研究也不证明2023—2026或未来能跑赢创业板。", ""]
    tmp = OUT / "HIST_RESULTS.md.tmp"
    tmp.write_text("\n".join(lines), encoding="utf-8")
    import os
    os.replace(tmp, OUT / "HIST_RESULTS.md")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--benchmarks", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    if args.benchmarks:
        fetch_benchmarks()
    elif args.smoke:
        init_qlib()
        from qlib.data import D
        sample = D.features(D.instruments("csi300"), ["$close"], start_time="2017-01-03", end_time="2017-01-13").swaplevel().sort_index()
        sample["score"] = np.arange(len(sample), dtype=float) % 300
        replay(sample, OUT / "smoke-not-HIST-results", start="2017-01-03", end="2017-01-13", smoke=True)
    else:
        init_qlib()
        seeds = [args.seed] if args.seed is not None else CFG["seeds"]
        for seed in seeds:
            directory = OUT / "runs" / f"seed-{seed:02d}"
            if not (directory / "complete.json").exists():
                continue
            if not replay_complete(directory):
                replay(pd.read_parquet(directory / "pred-test.parquet"), directory / REPLAY_DIR)
    if not args.no_summary:
        report_results()
