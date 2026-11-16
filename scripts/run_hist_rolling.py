"""Resumable isolated 7-year x 10-seed HIST rolling study, then matched replays."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
from filelock import FileLock
from prepare_hist_reproduction import ROOT, UPSTREAM, digest, write_json
from prepare_hist_rolling import CONFIG, EXT, ORIGINAL, OUT, frozen_inputs
from replay_hist_reproduction import EXECUTION


def verify_inputs():
    saved = json.loads((OUT/"input-manifest.json").read_text())
    assert saved == frozen_inputs(), "Changed input/source after protocol freeze"
    return saved


def training_ready(year, seed):
    import train_hist_reproduction as trainer
    folder = OUT/"folds"/str(year)
    path = folder/"runs"/f"seed-{seed:02d}"/"complete.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    assert record["status"] == "training_and_prediction_complete" and record["seed"] == seed
    assert record["reload_exact_match"] and record["source_model_sha256"] == digest(UPSTREAM/"model.py")
    assert all((path.parent/n).exists() for n in ("model.pt", "pred-train.parquet", "pred-valid.parquet", "pred-test.parquet"))
    manifest = json.loads((folder/"manifest.json").read_text())
    assert manifest["input_fingerprint"] == verify_inputs()
    assert trainer.average_params is not None  # pinned training module is reused, not rewritten
    return True


def fit(year, seed, max_new_epochs=None):
    import train_hist_reproduction as trainer
    verify_inputs()
    folder = OUT/"folds"/str(year)
    manifest = json.loads((folder/"manifest.json").read_text())
    trainer.OUT = folder
    trainer.CFG = manifest["config"]
    trainer.train([seed], max_new_epochs=max_new_epochs)


def control_prediction(year, seed):
    """Original checkpoint, original normalizer, same observable test rows as rolling."""
    import torch
    import train_hist_reproduction as trainer
    from threadpoolctl import threadpool_limits
    folder = OUT/"folds"/str(year)
    dest = folder/"runs"/f"seed-{seed:02d}"
    marker = dest/"frozen-control.json"
    source = ORIGINAL/"runs"/dest.name/"model.pt"
    if marker.exists():
        data = json.loads(marker.read_text())
        assert data["model_sha256"] == digest(source) and data["pred_sha256"] == digest(dest/"pred-frozen.parquet")
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(UPSTREAM))
    from model import HIST
    trainer.OUT = folder
    with threadpool_limits(limits=2):
        data = trainer.DailyData("test", "cuda")
        data.x = torch.tensor(np.load(folder/"test/frozen_features.npy"), device="cuda")
        model = HIST(d_feat=6, num_layers=2, K=1).cuda().eval()
        model.load_state_dict(torch.load(source, map_location="cuda", weights_only=True))
        concepts = torch.tensor(np.load(ORIGINAL/"concepts.npy"), device="cuda")
        frame = trainer.predict(model, data, concepts)
    frame.to_parquet(dest/"pred-frozen.parquet")
    write_json(marker, {"model_sha256": digest(source), "pred_sha256": digest(dest/"pred-frozen.parquet"),
                       "rows": len(frame), "test_row_sha256": digest(folder/"test/index.parquet")})


def assemble(seed):
    verify_inputs()
    prefix = pd.read_parquet(ORIGINAL/"runs"/f"seed-{seed:02d}"/"pred-test.parquet")[["score"]]
    first = json.loads((OUT/"folds/2020/manifest.json").read_text())["plan"]["test"][0]
    prefix = prefix.loc[prefix.index.get_level_values("datetime") < first]
    row_index = None
    for variant, name in (("rolling", "pred-test.parquet"), ("frozen", "pred-frozen.parquet")):
        pieces, hashes = [prefix], {}
        for year in CONFIG["years"]:
            if not training_ready(year, seed):
                raise ValueError("Incomplete training fold")
            folder = OUT/"folds"/str(year)
            path = folder/"runs"/f"seed-{seed:02d}"/name
            part = pd.read_parquet(path)[["score"]]
            wanted = pd.MultiIndex.from_frame(pd.read_parquet(folder/"test/index.parquet"))
            assert part.index.equals(wanted) and np.isfinite(part.score).all()
            pieces.append(part)
            hashes[str(path.relative_to(OUT))] = digest(path)
        frame = pd.concat(pieces).sort_index()
        assert frame.index.is_unique
        if row_index is None:
            row_index = frame.index
        else:
            assert row_index.equals(frame.index), "Unmatched control prediction pool"
        dest = OUT/"comparisons"/variant/"runs"/f"seed-{seed:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dest/"pred-full.parquet")
        write_json(dest/"prediction.json", {"seed": seed, "variant": variant, "rows": len(frame),
            "prediction_sha256": digest(dest/"pred-full.parquet"), "fold_prediction_hashes": hashes,
            "input_fingerprint": verify_inputs(), "same_inference_pool": True})


def execution_fingerprint():
    return {**verify_inputs(), "rolling_protocol": digest(ROOT/"configs/hist_rolling_20261114.json")}


def replay(seed, variant):
    import finalize_hist_extension as correction
    import replay_hist_reproduction as base
    from build_hist_extension import init_extended
    from qlib.data import D
    branch = OUT/"comparisons"/variant
    dest = branch/"runs"/f"seed-{seed:02d}"
    proof = json.loads((dest/"prediction.json").read_text())
    assert proof["input_fingerprint"] == verify_inputs()
    assert proof["prediction_sha256"] == digest(dest/"pred-full.parquet")
    final = branch/"final"/f"seed-{seed:02d}"
    if (final/"complete.json").exists():
        record = json.loads((final/"complete.json").read_text())
        assert record["execution_fingerprint"] == execution_fingerprint()
        assert record["prediction_sha256"] == proof["prediction_sha256"]
        for lane, row in record["runs"].items():
            assert row["audit"]["passed"] and row["unhandled_retirement_rows"] == 0
            assert row["replay_daily_sha256"] == digest(final/"replay"/f"{lane}-daily.parquet")
        return
    for relative in ("provider/calendars/day.txt", "data/official/stock_basic-delisted-20261113.parquet"):
        target = branch/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(EXT/relative, target)
    init_extended()
    predictions = pd.read_parquet(dest/"pred-full.parquet")
    available = {line.split("\t")[0] for line in (EXT/"provider/instruments/all.txt").read_text().splitlines()}
    assert set(predictions.index.get_level_values("instrument")).issubset(available), "Missing replay instruments"
    base.OUT = branch
    base.execution_fingerprint = execution_fingerprint
    base.report_results = lambda *a, **kw: None
    raw = dest/"replay/replay.json"
    if raw.exists():
        assert all(v["execution_fingerprint"] == execution_fingerprint() for v in json.loads(raw.read_text()).values())
    else:
        base.replay(predictions, dest/"replay", start="2017-01-01", end=CONFIG["cutoff"])
    correction.OUT, correction.FINAL = branch, branch/"final"
    symbols = sorted(set(predictions.index.get_level_values("instrument")) | {correction.OLD, correction.NEW})
    bars = D.features(symbols, ["$close", "$open", "$factor", "$volume"],
        start_time="2025-01-01", end_time=CONFIG["cutoff"]).sort_index()
    rows = {lane: correction.replay_tail(seed, lane, predictions, bars) for lane in EXECUTION["lanes"]}
    # Both variants inherit identical previously audited pre-2020 positions/equity.
    continuity = {}
    for lane in EXECUTION["lanes"]:
        actual = pd.read_parquet(final/"replay"/f"{lane}-daily.parquet").loc[:"2019-12-31", "account"]
        old = pd.read_parquet(EXT/"final"/f"seed-{seed:02d}"/"replay"/f"{lane}-daily.parquet").loc[:"2019-12-31", "account"]
        error = float((actual-old).abs().max())
        assert error < .1
        continuity[lane] = error
    write_json(final/"complete.json", {"runs": rows, "execution_fingerprint": execution_fingerprint(),
        "prediction_sha256": proof["prediction_sha256"], "pre_2020_equity_error_yuan": continuity})


def report():
    benchmark = json.loads((EXT/"final/results.json").read_text())["benchmarks"]
    runs = {}
    for variant in ("rolling", "frozen"):
        runs[variant] = {}
        for seed in CONFIG["seeds"]:
            path = OUT/"comparisons"/variant/"final"/f"seed-{seed:02d}"/"complete.json"
            if path.exists():
                data = json.loads(path.read_text())
                assert data["execution_fingerprint"] == execution_fingerprint()
                runs[variant][str(seed)] = data["runs"]
    paired = sorted(set(runs["rolling"]) & set(runs["frozen"]))
    complete = len(paired) == len(CONFIG["seeds"])
    result = {"status": "complete" if complete else "partial_not_final", "paired_seeds": paired,
        "benchmarks": benchmark, "runs": runs, "comparison": {}, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    metrics = ["cumulative_return", "annualized_return_252", "annualized_volatility_252", "sharpe_rf0_252", "maximum_drawdown"]
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
    write_json(OUT/"results.json", result)
    lines = ["# HIST年度滚动训练：实时结果", "", f"状态：{result['status']}；完整成对种子 {len(paired)}/10。", "",
        "只展示已全部完成七年训练、三路回放和账本核验的成对账户；部分结果不能当最终结论。", "",
        "年度3年训练+末季度验证；当年数据只作前推。旧2014模型用相同推理股票集重算作为对照。", ""]
    for window, lanes in result["comparison"].items():
        lines += [f"## {window}", "", "|口径|版本|累计收益|年化收益|波动|夏普|最大回撤|", "|---|---|---:|---:|---:|---:|---:|"]
        for lane, values in lanes.items():
            for variant in ("rolling", "frozen"):
                s = values[variant]
                lines.append(f"|{lane}|{variant}|{s['cumulative_return']['mean']:.2%}|{s['annualized_return_252']['mean']:.2%}|{s['annualized_volatility_252']['mean']:.2%}|{s['sharpe_rf0_252']['mean']:.3f}|{abs(s['maximum_drawdown']['mean']):.2%}|")
    lines += ["", "静态概念矩阵、原VWAP零通道、月度成分快照及交易重建限制保留；不是实盘收益承诺。"]
    (OUT/"HIST_ROLLING_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def stage(seed, name, arguments):
    from run_hist_reproduction import retryable_failure
    folder = OUT/"workers"/f"seed-{seed:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, CONFIG["retry_attempts"]+1):
        path = folder/f"{name}-attempt-{attempt}.log"
        with path.open("a", encoding="utf-8") as log:
            p = subprocess.Popen([sys.executable, "-u", str(ROOT/"scripts/run_hist_rolling.py"), *arguments],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            write_json(folder/"worker.json", {"stage": name, "status": "running", "pid": p.pid,
                "attempt": attempt, "log": str(path.relative_to(ROOT)), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            code = p.wait()
        if code == 0:
            return
        tail = path.read_bytes()[-6000:].decode("utf-8", errors="replace")
        if attempt == CONFIG["retry_attempts"] or not retryable_failure(tail):
            write_json(folder/"worker.json", {"stage": name, "status": "failed", "exit_code": code, "log": str(path.relative_to(ROOT))})
            raise RuntimeError(f"Seed {seed} stage {name} failed; inspect {path.relative_to(ROOT)}")
        time.sleep(2)


def seed_job(seed):
    for year in CONFIG["years"]:
        if not training_ready(year, seed):
            stage(seed, f"train-{year}", ["train", "--seed", str(seed), "--year", str(year)])
            if not training_ready(year, seed):
                raise ValueError("Training exited without verified completion")
        stage(seed, f"control-{year}", ["control", "--seed", str(seed), "--year", str(year)])
    assemble(seed)
    for variant in ("rolling", "frozen"):
        stage(seed, f"replay-{variant}", ["replay", "--seed", str(seed), "--variant", variant])
    write_json(OUT/"workers"/f"seed-{seed:02d}"/"worker.json", {"status": "complete", "seed": seed})
    return {"seed": seed, "status": "complete"}


def workflow():
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT/"workflow.lock"), timeout=0):
        write_json(OUT/"workflow.json", {"status": "preparing", "pid": os.getpid(), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        if not (OUT/"preparation-progress.json").exists() or json.loads((OUT/"preparation-progress.json").read_text())["status"] != "complete":
            with (OUT/"preparation.log").open("a", encoding="utf-8") as log:
                p = subprocess.Popen([sys.executable, "-u", str(ROOT/"scripts/prepare_hist_rolling.py")], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                while p.poll() is None:
                    write_json(OUT/"workflow.json", {"status": "preparing", "pid": os.getpid(), "worker_pid": p.pid,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                    time.sleep(5)
                if p.returncode != 0:
                    write_json(OUT/"workflow.json", {"status": "preparation_failed", "exit_code": p.returncode})
                    return 1
        verify_inputs()
        results = []
        with ThreadPoolExecutor(max_workers=CONFIG["maximum_training_workers"]) as pool:
            pending = {pool.submit(seed_job, seed): seed for seed in CONFIG["seeds"]}
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    seed = pending.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as exc:  # noqa: BLE001 -- keep failures explicit and other fixed seeds running
                        results.append({"seed": seed, "status": "failed", "error": str(exc)})
                    report()
                write_json(OUT/"workflow.json", {"status": "running", "pid": os.getpid(), "completed": results,
                    "pending": sorted(pending.values()), "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "models_completed": len(list((OUT/"folds").glob("*/runs/seed-*/complete.json"))), "models_required": 70})
        summary = report()
        ok = summary["status"] == "complete" and all(r["status"] == "complete" for r in results)
        write_json(OUT/"workflow.json", {"status": "complete" if ok else "failed_requires_attention", "completed": results,
            "pending": [], "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models_required": 70})
        print("FINAL", summary["status"], flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["workflow", "train", "control", "replay", "report"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--year", type=int)
    parser.add_argument("--variant", choices=["rolling", "frozen"])
    parser.add_argument("--max-new-epochs", type=int)
    args = parser.parse_args()
    if args.phase == "train":
        fit(args.year, args.seed, args.max_new_epochs)
    elif args.phase == "control":
        control_prediction(args.year, args.seed)
    elif args.phase == "replay":
        replay(args.seed, args.variant)
    elif args.phase == "report":
        report()
    else:
        raise SystemExit(workflow())
