"""Bounded local job continuation; isolated research only, no service/trading control.

Adopts explicitly identified workers and queues only the already frozen study stages.
Failures are recorded, not bypassed. This is a run-to-completion job, not a scheduled task.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import psutil
from prepare_ashare_three_year import BASE, ROOT
from prepare_four_strategy_2026 import save


def main(financial_pid, training_pids, execution_pid=None, warmup_pid=None):
    output = BASE / "workflow-v1"
    output.mkdir(exist_ok=True)
    beginning = time.monotonic()
    handles = []
    adopted = []
    needs_financial = not (BASE / "data-v1/financial-results.json").exists()
    adopted_ids = [
        *([financial_pid] if needs_financial else []),
        *training_pids,
        *([execution_pid] if execution_pid else []),
        *([warmup_pid] if warmup_pid else []),
    ]
    for pid in adopted_ids:
        proc = psutil.Process(pid)
        cmd = " ".join(proc.cmdline())
        if "ashare_three_year" not in cmd:
            raise ValueError("CANNOT_ADOPT_UNRELATED_PROCESS")
        adopted.append(
            {"pid": pid, "created": proc.create_time(), "command": cmd, "args": proc.cmdline()}
        )
    save(output / "adopted.json", adopted)
    env = {**os.environ, "PYTHONHASHSEED": "42", "TQDM_DISABLE": "1"}
    processes = {}
    launched = set()
    complete = set()
    if execution_pid or (BASE / "data-v1/execution-results.json").exists():
        launched.add("execution-data")
    if (BASE / "data-v1/execution-results.json").exists():
        complete.add("execution-data")
    if (BASE / "panel-v1/results.json").exists():
        launched.add("augmentation")
        complete.add("augmentation")
    warmup_ready_path = BASE / "warmup-v1/results.json"
    if warmup_ready_path.exists():
        complete.add("financial-warmup")

    def launch(name, script, args=()):
        stream = (output / (name + ".log")).open("a", encoding="utf-8")
        handles.append(stream)
        p = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" / script), *args],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            if os.name == "nt"
            else 0,
        )
        psutil.Process(p.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        processes[name] = p
        launched.add(name)
        print(json.dumps({"event": "STARTED", "stage": name, "pid": p.pid}), flush=True)

    def alive(pid):
        record = next((x for x in adopted if x["pid"] == pid), None)
        if record is None:
            return False
        try:
            return psutil.Process(pid).create_time() == record["created"]
        except psutil.NoSuchProcess:
            return False

    financial_groups = {
        "financial-2023-2024": ["2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4"],
        "financial-2025": ["2025Q1", "2025Q2", "2025Q3", "2025Q4"],
        "financial-2026": ["2026Q1", "2026Q2", "2026Q3", "2026Q4"],
    }
    for name, folds in financial_groups.items():
        if all(
            (BASE / "training-v1" / f / "financial_rank_cap/results.json").exists() for f in folds
        ):
            launched.add(name)
            complete.add(name)
    last_snapshot = None
    try:
        while time.monotonic() - beginning < 12 * 3600:
            if "financial-warmup" not in complete:
                if warmup_ready_path.exists():
                    complete.add("financial-warmup")
                elif not warmup_pid or not alive(warmup_pid):
                    raise RuntimeError("FINANCIAL_WARMUP_NOT_COMPLETE")
            if execution_pid and "execution-data" not in complete:
                if (BASE / "data-v1/execution-results.json").exists():
                    complete.add("execution-data")
                elif not alive(execution_pid):
                    raise RuntimeError("ADOPTED_EXECUTION_FETCH_EXITED_WITHOUT_COMPLETE_ARTIFACT")
            for name, p in list(processes.items()):
                code = p.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError("STAGE_FAILED:" + name + ":" + str(code))
                    complete.add(name)
                    del processes[name]
                    print(json.dumps({"event": "FINISHED", "stage": name}), flush=True)
            financial_ready = (BASE / "data-v1/financial-results.json").exists()
            if not financial_ready and not alive(financial_pid):
                raise RuntimeError("ADOPTED_FINANCIAL_FETCH_EXITED_WITHOUT_COMPLETE_ARTIFACT")
            if financial_ready and "execution-data" not in launched:
                launch("execution-data", "prepare_ashare_three_year.py", ["--mode", "execution"])
            free_gib = psutil.virtual_memory().available / 1024**3
            if (
                financial_ready
                and "financial-warmup" in complete
                and "augmentation" not in launched
                and free_gib >= 24
            ):
                launch("augmentation", "build_ashare_three_year.py", ["--stage", "panel"])
            active_rank = sum(
                alive(pid)
                and "--cases rank_cap" in next(x["command"] for x in adopted if x["pid"] == pid)
                for pid in training_pids
            )
            active_fin = sum(n.startswith("financial-") for n in processes)
            gpu_count = sum(
                alive(pid)
                and "--cases rnn_rank_cap" in next(x["command"] for x in adopted if x["pid"] == pid)
                for pid in training_pids
            )
            gpu_preparing = any(
                json.loads(p.read_bytes()).get("phase") == "FITTING"
                and any(
                    alive(pid) and json.loads(p.read_bytes()).get("pid") == pid
                    for pid in training_pids
                )
                for p in (BASE / "training-v1").glob("progress-rnn*.json")
            )
            if (
                "data-audit" in complete
                and active_rank + active_fin < 3 - gpu_count
                and free_gib >= 18
                and not gpu_preparing
            ):
                for name, folds in financial_groups.items():
                    if name not in launched:
                        launch(
                            name,
                            "train_ashare_three_year.py",
                            ["--cases", "financial_rank_cap", "--folds", *folds],
                        )
                        break  # Re-measure memory after the worker has loaded its input.
            if (
                {"augmentation", "execution-data"} <= complete
                and "data-audit" not in launched
                and free_gib >= 20
                and not gpu_preparing
            ):
                launch("data-audit", "audit_ashare_three_year_data.py")
            trained = list((BASE / "training-v1").glob("*/*/results.json"))
            for pid in training_pids:
                if alive(pid):
                    continue
                args = next(x["args"] for x in adopted if x["pid"] == pid)
                if "--folds" not in args or "--cases" not in args:
                    continue
                cases = args[args.index("--cases") + 1 : args.index("--folds")]
                folds = args[args.index("--folds") + 1 :]
                if any(
                    not (BASE / "training-v1" / f / c / "results.json").exists()
                    for f in folds
                    for c in cases
                ):
                    raise RuntimeError("ADOPTED_MODEL_WORKER_INCOMPLETE:" + str(pid))
            if not any(alive(pid) for pid in training_pids):
                existing_cases = [json.loads(p.read_bytes()) for p in trained]
                if sum(r["case"] != "financial_rank_cap" for r in existing_cases) != 39:
                    raise RuntimeError("ADOPTED_MODEL_WORKERS_EXITED_WITH_INCOMPLETE_MATRIX")
            if len(trained) == 52 and "data-audit" in complete and "replay-smoke" not in launched:
                launch("replay-smoke", "replay_ashare_three_year.py", ["--bps", "10", "--smoke"])
            if "replay-smoke" in complete:
                for bps in [10, 20]:
                    name = "replay-" + str(bps)
                    if name not in launched:
                        launch(name, "replay_ashare_three_year.py", ["--bps", str(bps)])
            if {"replay-10", "replay-20"} <= complete and "final-report" not in launched:
                launch("final-report", "report_ashare_three_year.py")
            snapshot = {
                "status": "COMPLETE" if "final-report" in complete else "RUNNING",
                "pid": os.getpid(),
                "utc": datetime.now(UTC).isoformat(),
                "trained_models": len(trained),
                "target_models": 52,
                "active": {n: p.pid for n, p in processes.items()},
                "adopted_alive": [pid for pid in adopted_ids if alive(pid)],
                "completed_stages": sorted(complete),
                "available_gib": round(free_gib, 2),
                "elapsed_seconds": round(time.monotonic() - beginning),
                "production_modified": False,
            }
            save(output / "progress.json", snapshot)
            meaningful = (
                snapshot["status"],
                len(trained),
                tuple(sorted(complete)),
                tuple(sorted(processes)),
            )
            if meaningful != last_snapshot:
                print(json.dumps(snapshot), flush=True)
                last_snapshot = meaningful
            if snapshot["status"] == "COMPLETE":
                return
            time.sleep(20)
        raise RuntimeError("BOUNDED_WORKFLOW_DEADLINE_REACHED")
    except Exception as exc:
        failure = {
            "status": "FAILED_NOT_ACCEPTED",
            "pid": os.getpid(),
            "type": type(exc).__name__,
            "error": str(exc),
            "utc": datetime.now(UTC).isoformat(),
            "still_running_owned_children": {
                n: p.pid for n, p in processes.items() if p.poll() is None
            },
            "production_modified": False,
        }
        save(output / "failure.json", failure)
        save(output / "progress.json", failure)
        raise
    finally:
        for stream in handles:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--financial-pid", type=int, required=True)
    parser.add_argument("--training-pids", nargs="*", type=int, default=[])
    parser.add_argument("--execution-pid", type=int)
    parser.add_argument("--warmup-pid", type=int)
    args = parser.parse_args()
    main(args.financial_pid, args.training_pids, args.execution_pid, args.warmup_pid)
