"""Audit-only v2 for completed rolling HIST fits; all frozen sources stay immutable."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import finalize_hist_extension as correction
import numpy as np
import pandas as pd
import run_hist_rolling as rolling
from filelock import FileLock
from prepare_hist_reproduction import CFG, ROOT, digest, write_json
from prepare_hist_rolling import CONFIG, EXT, OUT
from replay_hist_reproduction import EXECUTION

FINAL = OUT / "final-v2"


def audit_tail(daily, orders, holdings, account, event, bars):
    """Empty rows represent cash only iff an independent fill ledger also has no stock."""
    assert daily.index.is_unique and daily.index.is_monotonic_increasing
    assert np.isfinite(daily[["account", "cash"]].to_numpy()).all()
    cash = float(account["cash"])
    quantity = {s: float(v["amount"]) for s, v in account.items() if s != "cash"}
    if event:
        actual = round(quantity.pop(correction.OLD)*event["old_factor"])
        quantity[correction.NEW] = quantity.get(correction.NEW, 0.) + np.floor(actual*.62+1e-7)/event["new_factor"]
    closes = bars["$close"].unstack("instrument").sort_index().ffill()
    if not orders.empty:
        assert orders.date.isin(daily.index).all(), "Order outside audit dates"
    if not holdings.empty:
        assert holdings.date.isin(daily.index).all(), "Holdings outside audit dates"
        assert not holdings.duplicated(["date", "instrument"]).any(), "Duplicate holding"
        assert np.isfinite(holdings.adjusted_shares).all()
        assert holdings.adjusted_shares.ge(0).all()
    grouped = {} if orders.empty else dict(tuple(orders.groupby("date")))
    snapshots = {} if holdings.empty else {
        d: g.set_index("instrument").adjusted_shares for d, g in holdings.groupby("date")}
    equity_error = cash_error = holding_error = 0.
    fills = empty_days = 0
    for date, row in daily.iterrows():
        for order in grouped.get(date, pd.DataFrame()).itertuples(index=False):
            if not order.value:
                assert order.filled_adjusted_shares == order.cost == 0
                continue
            assert order.direction in (0, 1)
            assert order.value > 0 and order.filled_adjusted_shares > 0
            buy = order.direction == 1
            assert order.cost == max(CFG["minimum_cost"], order.value*(CFG["buy_cost"] if buy else CFG["sell_cost"]))
            quote = bars.loc[(order.instrument, date)]
            assert quote["$volume"] > 0
            field = "$open" if order.lane.startswith("next_open") else "$close"
            assert abs(order.price_adjusted-quote[field]) < 1e-6
            assert abs(order.value-order.price_adjusted*order.filled_adjusted_shares) < .01
            quantity[order.instrument] = quantity.get(order.instrument, 0.) + order.filled_adjusted_shares*(1 if buy else -1)
            cash += (-order.value if buy else order.value)-order.cost
            fills += 1
        assert np.isfinite(cash) and cash >= -.01
        assert all(np.isfinite(q) and q > -1e-5 for q in quantity.values())
        held = pd.Series({s: q for s, q in quantity.items() if abs(q) > 1e-5}, dtype=float)
        marks = closes.loc[date].reindex(held.index)
        assert np.isfinite(marks).all(), "Missing held-position mark"
        equity_error = max(equity_error, abs(cash+float((held*marks).sum())-row.account))
        cash_error = max(cash_error, abs(cash-row.cash))
        if date not in snapshots:
            assert held.empty, f"Missing nonempty holding snapshot: {date}"
            saved = pd.Series(dtype=float)
        else:
            saved = snapshots[date]
        aligned = held.subtract(saved, fill_value=0.)
        error = 0. if aligned.empty else float(aligned.abs().max())
        assert np.isfinite(error)
        holding_error = max(holding_error, error)
        if held.empty:
            assert abs(row.account-row.cash) < .1, "Cash-only equity mismatch"
            empty_days += 1
    assert equity_error < .1 and cash_error < .1 and holding_error < 1e-4
    return {"passed": True, "days": len(daily), "fills": fills, "cash_only_days": empty_days,
            "max_equity_error_yuan": equity_error, "max_cash_error_yuan": cash_error,
            "max_holding_error_adjusted_shares": holding_error, "audit_revision": "cash-only-v2"}


def revision():
    return {"frozen_execution_fingerprint": rolling.execution_fingerprint(),
            "audit_adapter_sha256": digest(Path(__file__))}


def immutable_snapshot():
    paths = []
    for name in ("model.pt", "pred-train.parquet", "pred-valid.parquet", "pred-test.parquet", "pred-frozen.parquet"):
        paths += list((OUT/"folds").glob(f"*/runs/seed-*/{name}"))
    assert len(paths) == 350, "All 70 original fits and controls must exist"
    paths += list((OUT/"folds").glob("*/normalizer.npz"))
    paths += list((OUT/"comparisons").glob("*/runs/seed-*/pred-full.parquet"))
    paths += list((OUT/"comparisons").glob("*/final/seed-*/replay/*.parquet"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def compare_replay_tables(old, new, kind):
    """Ignore row enumeration, not trades; allow only absolute summation roundoff."""
    if kind in ("holdings", "orders"):
        keys = ["date", "instrument"] + (["direction"] if kind == "orders" else [])
        left = old.sort_values(keys, kind="stable").reset_index(drop=True)
        right = new.sort_values(keys, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        return {"canonical_exact": True}
    assert old.index.equals(new.index) and old.columns.equals(new.columns)
    rates = {"return", "net_return", "turnover", "cost", "bench"}
    errors = {}
    for column in old:
        tolerance = 1e-12 if column in rates else 1e-6
        if column == "total_turnover":
            # Lifetime yuan volume can exceed 1e11: one float64 ULP is already 3e-5 yuan.
            tolerance = np.maximum(tolerance, 8*np.spacing(np.maximum(old[column].abs(), new[column].abs())))
            assert np.max(tolerance) <= .001, "Unexpected turnover magnitude"
            assert np.isfinite(old[column]).all() and np.isfinite(new[column]).all()
            assert ((old[column]-new[column]).abs() <= tolerance).all()
        else:
            np.testing.assert_allclose(old[column], new[column], rtol=0, atol=tolerance, equal_nan=False)
        errors[column] = float((old[column]-new[column]).abs().max())
    return {"absolute_errors": errors, "money_tolerance_yuan": 1e-6, "rate_tolerance": 1e-12,
            "lifetime_turnover_tolerance": "max(1e-6 yuan, 8 float64 ULP), capped at .001 yuan"}


def finalize_one(seed, variant):
    import replay_hist_reproduction as base
    from build_hist_extension import init_extended
    from qlib.data import D

    stamp = revision()
    dest = FINAL/variant/f"seed-{seed:02d}"
    if (dest/"complete.json").exists():
        done = json.loads((dest/"complete.json").read_text())
        assert done["revision"] == stamp
        for name, value in done["outputs"].items():
            assert digest(dest/"replay"/name) == value
        return
    branch = OUT/"comparisons"/variant
    source = branch/"runs"/f"seed-{seed:02d}"
    proof = json.loads((source/"prediction.json").read_text())
    assert proof["input_fingerprint"] == rolling.verify_inputs()
    assert proof["prediction_sha256"] == digest(source/"pred-full.parquet")
    # These two small reference files are normally staged by the original runner.
    import shutil
    for name in ("provider/calendars/day.txt", "data/official/stock_basic-delisted-20261113.parquet"):
        target = branch/name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(EXT/name, target)
    init_extended()
    predictions = pd.read_parquet(source/"pred-full.parquet")
    base.OUT = branch
    base.execution_fingerprint = rolling.execution_fingerprint
    base.report_results = lambda *a, **kw: None
    raw = source/"replay/replay.json"
    if raw.exists():
        assert all(v["execution_fingerprint"] == rolling.execution_fingerprint() for v in json.loads(raw.read_text()).values())
    else:
        base.replay(predictions, source/"replay", start="2017-01-01", end=CONFIG["cutoff"])
    correction.OUT, correction.FINAL = branch, FINAL/variant
    correction.audit_tail = audit_tail
    symbols = sorted(set(predictions.index.get_level_values("instrument")) | {correction.OLD, correction.NEW})
    bars = D.features(symbols, ["$close", "$open", "$factor", "$volume"],
                      start_time="2025-01-01", end_time=CONFIG["cutoff"]).sort_index()
    rows = {lane: correction.replay_tail(seed, lane, predictions, bars) for lane in EXECUTION["lanes"]}
    continuity, parity = {}, {}
    for lane in EXECUTION["lanes"]:
        actual = pd.read_parquet(dest/"replay"/f"{lane}-daily.parquet").loc[:"2019-12-31", "account"]
        old = pd.read_parquet(EXT/"final"/f"seed-{seed:02d}"/"replay"/f"{lane}-daily.parquet").loc[:"2019-12-31", "account"]
        assert actual.index.equals(old.index)
        continuity[lane] = float((actual-old).abs().max())
        assert continuity[lane] < .1
        # Recheck successful v1 outputs rather than silently changing their trading results.
        for suffix in ("daily", "holdings", "orders"):
            name = f"{lane}-{suffix}.parquet"
            prior = branch/"final"/f"seed-{seed:02d}"/"replay"/name
            if prior.exists():
                parity[name] = compare_replay_tables(pd.read_parquet(prior), pd.read_parquet(dest/"replay"/name), suffix)
    outputs = {p.name: digest(p) for p in (dest/"replay").glob("*.parquet")}
    assert len(outputs) == 9
    write_json(dest/"complete.json", {"runs": rows, "revision": stamp, "outputs": outputs,
        "prediction_sha256": proof["prediction_sha256"], "pre_2020_equity_error_yuan": continuity,
        "unchanged_existing_v1_tables": parity})


def report():
    stamp = revision()
    runs = {v: {} for v in ("rolling", "frozen")}
    for variant, variant_runs in runs.items():
        for seed in CONFIG["seeds"]:
            p = FINAL/variant/f"seed-{seed:02d}"/"complete.json"
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            assert data["revision"] == stamp
            for name, value in data["outputs"].items():
                assert digest(p.parent/"replay"/name) == value
            assert all(r["audit"]["passed"] and r["unhandled_retirement_rows"] == 0 for r in data["runs"].values())
            variant_runs[str(seed)] = data["runs"]
    paired = sorted(set(runs["rolling"]) & set(runs["frozen"]))
    benchmark = json.loads((EXT/"final/results.json").read_text())["benchmarks"]
    metrics = ("cumulative_return", "annualized_return_252", "annualized_volatility_252", "sharpe_rf0_252", "maximum_drawdown")
    result = {"status": "complete" if len(paired) == 10 else "partial_not_final", "paired_seeds": paired,
              "runs": runs, "benchmarks": benchmark, "revision": stamp, "comparison": {},
              "authoritative_result": str((FINAL/"results.json").relative_to(ROOT)),
              "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for window in ("since2020", "post2020"):
        result["comparison"][window] = {}
        for lane in EXECUTION["lanes"]:
            if not paired:
                continue
            rows = {}
            for variant in runs:
                values = [runs[variant][s][lane][window] for s in paired]
                rows[variant] = {m: {"mean": float(np.mean([v[m] for v in values])),
                    "min": float(min(v[m] for v in values)), "max": float(max(v[m] for v in values))} for m in metrics}
                rows[variant]["seeds_beating_ChiNext_TR"] = sum(v["cumulative_return"] > benchmark["ChiNext_total_return"][window]["cumulative_return"] for v in values)
            rows["rolling_minus_frozen_mean"] = {m: rows["rolling"][m]["mean"]-rows["frozen"][m]["mean"] for m in metrics}
            result["comparison"][window][lane] = rows
    write_json(FINAL/"results.json", result)
    write_report(result)
    return result


def write_report(result):
    paired, runs = result["paired_seeds"], result["runs"]
    names = dict(zip(EXECUTION["lanes"], ["次日开盘等权", "次日收盘等权", "Qlib Top30/Drop30"]))
    lines = ["# HIST年度滚动重训：完整对照结果", "",
        f"状态：{result['status']}；完整成对种子 {len(paired)}/10。更新：{result['updated_at']}。",
        "", "70次训练已完成；所有组都必须通过三路账本核验才计入成对汇总，不排除亏损种子。",
        "", "2020-01-02—2026-11-13，1663个交易日；年化按252日，夏普无风险利率0。表内为十个独立种子的指标均值，不是十账户合并净值。",
        "", "每年最近三年训练、末季度验证；同一可观察股票池、同一手续费、同一2019年末账户起点。未修改标签、特征、网络或调仓规则。",
        "", "## 2020年至今", "",
        "|执行|版本|累计收益|年化收益|波动|夏普|最大回撤|超过创业板全收益|",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for lane, rows in result["comparison"]["since2020"].items():
        for v, label in (("frozen", "旧冻结"), ("rolling", "滚动重训")):
            s = rows[v]
            lines.append(f"|{names[lane]}|{label}|{s['cumulative_return']['mean']:.2%}|{s['annualized_return_252']['mean']:.2%}|{s['annualized_volatility_252']['mean']:.2%}|{s['sharpe_rf0_252']['mean']:.3f}|{abs(s['maximum_drawdown']['mean']):.2%}|{s['seeds_beating_ChiNext_TR']}/{len(paired)}|")
    lines += ["", "## 同期指数", "", "|指数|累计收益|年化收益|波动|夏普|最大回撤|", "|---|---:|---:|---:|---:|---:|"]
    bnames = {"ChiNext_price": "创业板价格", "ChiNext_total_return": "创业板全收益", "CSI300_price": "沪深300价格", "CSI300_total_return": "沪深300全收益"}
    for name, b in result["benchmarks"].items():
        s = b["since2020"]
        lines.append(f"|{bnames[name]}|{s['cumulative_return']:.2%}|{s['annualized_return_252']:.2%}|{s['annualized_volatility_252']:.2%}|{s['sharpe_rf0_252']:.3f}|{abs(s['maximum_drawdown']):.2%}|")
    if paired:
        lane = "next_open_top30_targets"
        lines += ["", "## 开盘组逐年收益（种子均值）", "", "|年份|旧冻结|滚动重训|差额（百分点）|创业板全收益|沪深300全收益|", "|---|---:|---:|---:|---:|---:|"]
        for year in range(2020, 2027):
            a, b = [float(np.mean([runs[v][s][lane]["year_metrics"][str(year)]["cumulative_return"] for s in paired])) for v in ("frozen", "rolling")]
            indices = [result["benchmarks"][name]["since2020"]["yearly_return"][str(year)] for name in ("ChiNext_total_return", "CSI300_total_return")]
            lines.append(f"|{year}{'（截至11月13日）' if year == 2026 else ''}|{a:.2%}|{b:.2%}|{(b-a)*100:+.2f}|{indices[0]:.2%}|{indices[1]:.2%}|")
        lines += ["", "## 开盘组逐种子", "", "|种子|旧冻结累计|滚动累计|差额（百分点）|滚动夏普|滚动回撤|", "|---|---:|---:|---:|---:|---:|"]
        for s in paired:
            a, b = [runs[v][s][lane]["since2020"] for v in ("frozen", "rolling")]
            lines.append(f"|{s}|{a['cumulative_return']:.2%}|{b['cumulative_return']:.2%}|{(b['cumulative_return']-a['cumulative_return'])*100:+.2f}|{b['sharpe_rf0_252']:.3f}|{abs(b['maximum_drawdown']):.2%}|")
        lines += ["", "## 2020年起成本与换手（种子均值）", "", "|执行|版本|交易成本（元）|累计换手（账户权益倍数）|成交笔数|全现金日数|", "|---|---|---:|---:|---:|---:|"]
        for lane in EXECUTION["lanes"]:
            for variant in ("frozen", "rolling"):
                values = []
                for seed in paired:
                    folder = FINAL/variant/f"seed-{int(seed):02d}"/"replay"
                    d = pd.read_parquet(folder/f"{lane}-daily.parquet").loc["2020-01-01":]
                    o = pd.read_parquet(folder/f"{lane}-orders.parquet")
                    o = o.loc[o.date.ge(pd.Timestamp("2020-01-01"))]
                    values.append([o.cost.sum(), d.turnover.sum(), o.value.gt(0).sum(), (d.account-d.cash).abs().lt(.1).sum()])
                a, b, c, d = np.mean(values, axis=0)
                lines.append(f"|{names[lane]}|{variant}|{a:,.2f}|{b:.2f}|{c:.0f}|{d:.1f}|")
    lines += ["", "## 审计与边界", "",
        "- 修复仅针对全现金日的空持仓快照：独立成交账本确实无股、现金与权益同时一致才通过；真正持仓缺失仍报错。",
        "- 全部组统一使用cash-only-v2；原模型、年度预测、旧回放文件保留，已有回放的成交与持仓按日期/证券精确比对；权益最多允许1e-6元、比例最多1e-12浮点误差，千亿元级累计交易额允许8个浮点末位且不超过0.001元。修复不改变交易决策。",
        "- 原账户自2017年连续积累，2020年分叉；延续原1亿元初始资本，买5bps/卖15bps、最低5元、整手与现金保护。",
        "- 静态概念时点、VWAP零通道、月度成分快照、固定9.5%涨跌限制代理及无冲击成本等局限保留，不是全面点时认证或生产发布。",
        "- 旧冻结对照在相同推理行集重算，不能直接拿此前不同池的十组均值作成对改善。",
        "- 全部明细及2021年起指标见同目录results.json；这里不按收益选择种子或隐藏收盘组。"]
    (FINAL/"HIST_ROLLING_RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def workflow():
    FINAL.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT/"workflow.lock"), timeout=0):
        revision()
        assert all(rolling.training_ready(y, s) for y in CONFIG["years"] for s in CONFIG["seeds"])
        snapshot = immutable_snapshot()
        receipt = FINAL/"immutable-inputs.json"
        if receipt.exists():
            assert json.loads(receipt.read_text()) == snapshot
        else:
            write_json(receipt, snapshot)
        # Preserve the earlier partial report and failed coordinator state as evidence.
        import shutil
        for name in ("workflow.json", "results.json", "HIST_ROLLING_RESULTS.md"):
            backup = FINAL/"before-repair"/name
            if not backup.exists() and (OUT/name).exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(OUT/name, backup)

        def job(seed, variant):
            folder = FINAL/variant/f"seed-{seed:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/"run.log").open("a", encoding="utf-8") as log:
                p = subprocess.Popen([sys.executable, "-u", __file__, "seed", "--seed", str(seed), "--variant", variant],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                return {"seed": seed, "variant": variant, "exit_code": p.wait()}

        completed = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(job, s, v): (s, v) for s in CONFIG["seeds"] for v in ("rolling", "frozen")}
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    completed.append(future.result())
                    print("JOB", completed[-1], flush=True)
                state = {"status": "finalizing", "pid": os.getpid(), "models_completed": 70,
                         "completed": completed, "pending": list(pending.values()),
                         "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "resume_entry": "scripts/finalize_hist_rolling.py workflow"}
                write_json(FINAL/"workflow.json", state)
                write_json(OUT/"workflow.json", state)
        assert immutable_snapshot() == snapshot, "Changed original models, predictions or existing replay tables"
        summary = report()
        ok = summary["status"] == "complete" and all(x["exit_code"] == 0 for x in completed)
        write_json(FINAL/"verification.json", {"passed": ok, "immutable_files_unchanged": len(snapshot),
                   "trained_models": 70, "paired_seeds": summary["paired_seeds"], "revision": revision()})
        state.update(status="complete" if ok else "failed_requires_attention", immutable_files_unchanged=len(snapshot))
        write_json(FINAL/"workflow.json", state)
        write_json(OUT/"workflow.json", state)
        if ok:
            write_json(OUT/"results.json", summary)
            shutil.copy2(FINAL/"HIST_ROLLING_RESULTS.md", OUT/"HIST_ROLLING_RESULTS.md")
        return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["seed", "workflow", "report"])
    parser.add_argument("--seed", type=int, choices=range(10))
    parser.add_argument("--variant", choices=["rolling", "frozen"])
    args = parser.parse_args()
    if args.phase == "seed":
        finalize_one(args.seed, args.variant)
    elif args.phase == "report":
        report()
    else:
        raise SystemExit(workflow())
