"""Frozen HIST inference, full continuous replay, and same-window reporting."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from filelock import FileLock
from prepare_hist_extension import CONFIG, OUT, SOURCE
from prepare_hist_reproduction import CFG, ROOT, UPSTREAM, digest, write_json
from replay_hist_reproduction import EXECUTION, performance


def fingerprint():
    import replay_hist_reproduction as base
    original = {name: digest(ROOT/name) for name in (
        "configs/hist_reproduction_execution_v1.json", "scripts/replay_hist_reproduction.py",
        "scripts/hist_reproduction_exchange.py", "scripts/audit_hist_reproduction_replay.py")}
    original.update({name: digest(ROOT/name) for name in (
        "configs/hist_extension_20261114.json", "scripts/prepare_hist_extension.py",
        "scripts/build_hist_extension.py", "scripts/run_hist_extension.py")})
    original.update({"inputs:"+name: digest(OUT/name) for name in (
        "data/complete.json", "data/quote-audit.json", "provider-complete.json", "normalizer-check.json")})
    assert base.CFG == CFG
    return original


def predict_all():
    import torch
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(UPSTREAM))
    from model import HIST
    assert torch.cuda.is_available()
    assert digest(UPSTREAM/"model.py") == json.loads((SOURCE/"runs/seed-00/complete.json").read_text())["source_model_sha256"]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    concepts = torch.tensor(np.load(SOURCE/"concepts.npy"), device="cuda")
    models = {}
    for seed in CFG["seeds"]:
        model = HIST(d_feat=6, num_layers=CFG["num_layers"], K=CFG["K"]).cuda().eval()
        model.load_state_dict(torch.load(SOURCE/"runs"/f"seed-{seed:02d}"/"model.pt", map_location="cuda", weights_only=True))
        models[seed] = model
    with threadpool_limits(limits=2), torch.no_grad():
        for year in range(2021, pd.Timestamp(CONFIG["requested_end"]).year+1):
            folder = OUT/"features"/str(year)
            deadline = time.monotonic()+1800
            while not (folder/"complete.json").exists():
                write_json(OUT/"inference-progress.json", {"status": "waiting_for_features", "year": year,
                    "updated_at": datetime.now(UTC).isoformat()})
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Feature preparation did not complete for {year}")
                time.sleep(5)
            index = pd.MultiIndex.from_frame(pd.read_parquet(folder/"index.parquet"))
            x = torch.tensor(np.load(folder/"features.npy"), device="cuda")
            mv = torch.tensor(np.load(folder/"market_value.npy"), device="cuda")
            sid = torch.tensor(np.load(folder/"stock_index.npy"), device="cuda")
            counts = np.load(folder/"counts.npy")
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            for seed, model in models.items():
                dest = OUT/"runs"/f"seed-{seed:02d}"
                dest.mkdir(parents=True, exist_ok=True)
                path = dest/f"pred-{year}.parquet"
                if path.exists():
                    continue
                predictions = []
                for first, count in zip(starts, counts):
                    slc = slice(int(first), int(first+count))
                    predictions.append(model(x[slc], concepts[sid[slc]], mv[slc]).cpu().numpy())
                values = np.concatenate(predictions)
                assert np.isfinite(values).all()
                pd.DataFrame({"score": values}, index=index).to_parquet(path)
                write_json(OUT/"inference-progress.json", {"status": "predicting", "year": year,
                    "seed": seed, "rows": len(values), "updated_at": datetime.now(UTC).isoformat()})
                print("PREDICT", seed, year, len(values), flush=True)
            del x, mv, sid
    for seed in CFG["seeds"]:
        dest = OUT/"runs"/f"seed-{seed:02d}"
        pieces = [pd.read_parquet(SOURCE/"runs"/dest.name/"pred-test.parquet")[["score"]]]
        pieces += [pd.read_parquet(dest/f"pred-{year}.parquet") for year in range(2021, 2027)]
        combined = pd.concat(pieces).sort_index()
        assert combined.index.is_unique
        combined.to_parquet(dest/"pred-full.parquet")
        write_json(dest/"prediction.json", {"seed": seed, "rows": len(combined),
            "model_sha256": digest(SOURCE/"runs"/dest.name/"model.pt"), "prediction_sha256": digest(dest/"pred-full.parquet"),
            "normalizer_sha256": digest(OUT/"normalizer.npz"), "execution_fingerprint": fingerprint()})
    write_json(OUT/"inference-progress.json", {"status": "complete", "seeds": 10,
        "updated_at": datetime.now(UTC).isoformat()})


def replay_seed(seed):
    import replay_hist_reproduction as base
    from build_hist_extension import init_extended
    directory = OUT/"runs"/f"seed-{seed:02d}"
    if (directory/"replay/replay.json").exists():
        rows = json.loads((directory/"replay/replay.json").read_text())
        assert all(x["execution_fingerprint"] == fingerprint() for x in rows.values())
        return
    proof = json.loads((directory/"prediction.json").read_text())
    assert proof["execution_fingerprint"] == fingerprint()
    assert proof["model_sha256"] == digest(SOURCE/"runs"/directory.name/"model.pt")
    assert proof["prediction_sha256"] == digest(directory/"pred-full.parquet")
    init_extended()
    base.OUT = OUT
    base.execution_fingerprint = fingerprint
    base.report_results = lambda *args, **kwargs: None
    pred = pd.read_parquet(directory/"pred-full.parquet")
    base.replay(pred, directory/"replay", start="2017-01-01", end=str(pd.Timestamp(CONFIG["requested_end"]).date()))
    # The full replay must exactly retain the first study before the splice.
    comparisons = {}
    retirements = pd.read_parquet(OUT/"data/official/stock_basic-delisted-20261113.parquet")
    retired = {s.split(".")[1]+s[:6]: pd.Timestamp(d) for s, d in zip(retirements.ts_code, retirements.delist_date)
               if isinstance(d, str) and len(d) == 8}
    retirement_audit = {}
    for lane in EXECUTION["lanes"]:
        new = pd.read_parquet(directory/"replay"/f"{lane}-daily.parquet")
        old = pd.read_parquet(SOURCE/"runs"/directory.name/"replay-cash-safe-v2"/f"{lane}-daily.parquet")
        before = new.reindex(old.index)
        error = float((before.account-old.account).abs().max())
        assert error < .1, f"Pre-2021 account changed: {lane} {error}"
        comparisons[lane] = {"days": len(old), "max_equity_error_yuan": error}
        holdings = pd.read_parquet(directory/"replay"/f"{lane}-holdings.parquet")
        held_retired = holdings.loc[holdings.instrument.isin(retired)].copy()
        if not held_retired.empty:
            held_retired["delist_date"] = held_retired.instrument.map(retired)
            held_retired = held_retired.loc[held_retired.date.ge(held_retired.delist_date)]
        held_retired.to_parquet(directory/"replay"/f"{lane}-retirement-holdings.parquet", index=False)
        retirement_audit[lane] = {"post_delisting_holding_rows": len(held_retired),
                                 "symbols": sorted(held_retired.instrument.unique())}
    write_json(directory/"continuity-audit.json", comparisons)
    write_json(directory/"retirement-audit.json", retirement_audit)
    print("SEED_COMPLETE", seed, flush=True)


def report():
    windows = {"full_2017_now": "2017-01-01", "post2020": "2021-01-01"}
    runs = {}
    for seed in CFG["seeds"]:
        directory = OUT/"runs"/f"seed-{seed:02d}"
        if not (directory/"continuity-audit.json").exists():
            continue
        raw = json.loads((directory/"replay/replay.json").read_text())
        assert all(x["execution_fingerprint"] == fingerprint() for x in raw.values())
        lanes = {}
        for lane in EXECUTION["lanes"]:
            daily = pd.read_parquet(directory/"replay"/f"{lane}-daily.parquet")
            lanes[lane] = {name: performance(daily.loc[daily.index >= start, "net_return"]) for name, start in windows.items()}
            lanes[lane]["year_metrics"] = {str(y): performance(g.net_return) for y, g in daily.groupby(daily.index.year)}
            lanes[lane]["audit"] = raw[lane]["independent_ledger_audit"]
            lanes[lane]["cash_min"] = raw[lane]["cash_min"]
            lanes[lane]["retirement_audit"] = json.loads((directory/"retirement-audit.json").read_text())[lane]
        runs[str(seed)] = lanes
    benchmarks = {}
    for path in sorted((OUT/"benchmarks").glob("*.parquet")):
        b = pd.read_parquet(path).sort_values("trade_date")
        returns = pd.Series(b.close.to_numpy(), index=pd.to_datetime(b.trade_date)).pct_change(fill_method=None).iloc[1:]
        benchmarks[path.stem] = {name: performance(returns.loc[returns.index >= start]) for name, start in windows.items()}
        benchmarks[path.stem]["year_metrics"] = {str(y): performance(g) for y, g in returns.groupby(returns.index.year)}
    summary = {"status": "complete" if len(runs) == 10 else "partial_not_final", "completed": len(runs),
        "cutoff": CONFIG["requested_end"], "runs": runs, "benchmarks": benchmarks,
        "windows": {}, "execution_fingerprint": fingerprint(), "post2020_account": "inherited continuous positions/equity, not fresh cash restart"}
    metrics = ("cumulative_return", "annualized_return_252", "annualized_volatility_252", "sharpe_rf0_252", "maximum_drawdown")
    for window in windows:
        summary["windows"][window] = {}
        for lane in EXECUTION["lanes"]:
            rows = [r[lane][window] for r in runs.values()]
            if rows:
                result = {m: {"mean": float(np.mean([r[m] for r in rows])), "min": float(min(r[m] for r in rows)),
                              "max": float(max(r[m] for r in rows))} for m in metrics}
                result["beating_ChiNext_TR"] = sum(r["cumulative_return"] > benchmarks["ChiNext_total_return"][window]["cumulative_return"] for r in rows)
                summary["windows"][window][lane] = result
    write_json(OUT/"results.json", summary)
    lines = ["# HIST 延长区间结果", "", f"完成 {len(runs)}/10；截止 {CONFIG['requested_end']}；状态 {summary['status']}。", "",
             "固定旧模型、固定三种成交口径。各行是十个种子指标的均值，不是组合账户。2021年以后为连续账户子区间，并非2021年重置现金。", ""]
    for window, results in summary["windows"].items():
        lines += [f"## {window}", "", "|口径|累计收益|年化收益|年化波动|夏普|最大回撤|", "|---|---:|---:|---:|---:|---:|"]
        for name, s in results.items():
            lines.append(f"|{name}|{s['cumulative_return']['mean']:.2%}|{s['annualized_return_252']['mean']:.2%}|{s['annualized_volatility_252']['mean']:.2%}|{s['sharpe_rf0_252']['mean']:.3f}|{s['maximum_drawdown']['mean']:.2%}|")
        for name, b in benchmarks.items():
            s = b[window]
            lines.append(f"|{name}|{s['cumulative_return']:.2%}|{s['annualized_return_252']:.2%}|{s['annualized_volatility_252']:.2%}|{s['sharpe_rf0_252']:.3f}|{s['maximum_drawdown']:.2%}|")
        lines.append("")
    lines += ["静态概念时点未认证、未知概念零行、VWAP仍为旧模型零通道、月度已知成员快照、执行限价代理等局限保留；不是可直接实盘的证明。"]
    (OUT/"HIST_EXTENSION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def workflow():
    with FileLock(str(OUT/"workflow.lock"), timeout=0):
        results = []

        def job(seed):
            path = OUT/"runs"/f"seed-{seed:02d}"/"replay.log"
            with path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-u", str(ROOT/"scripts/run_hist_extension.py"), "replay", "--seed", str(seed)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                code = process.wait()
            return {"seed": seed, "status": "complete" if code == 0 else "failed", "exit_code": code}

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(job, seed): seed for seed in CFG["seeds"]}
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    seed = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 -- retain failure status, no credential-bearing body
                        result = {"seed": seed, "status": "failed", "exception_type": type(exc).__name__}
                    results.append(result)
                    print(result, flush=True)
                    report()
                write_json(OUT/"workflow.json", {"pid": os.getpid(), "status": "running" if pending else "checking",
                    "completed": results, "pending": sorted(pending.values()), "updated_at": datetime.now(UTC).isoformat()})
        summary = report()
        ok = summary["status"] == "complete" and all(r["status"] == "complete" for r in results)
        write_json(OUT/"workflow.json", {"status": "complete" if ok else "failed_requires_attention",
            "completed": results, "pending": [], "updated_at": datetime.now(UTC).isoformat()})
        print("FINAL", summary["status"], summary["completed"], flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["all", "predict", "replay", "workflow", "report"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "all":
        predict_all()
        raise SystemExit(workflow())
    elif args.phase == "predict":
        predict_all()
    elif args.phase == "replay":
        replay_seed(args.seed)
    elif args.phase == "report":
        report()
    else:
        raise SystemExit(workflow())
