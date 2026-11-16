"""Detach this bounded research run from the interactive tool process lifetime.

Invoke once through Windows process management with a hidden window. It does not
install a service, schedule, startup item, change power/security, or touch trading.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import psutil
from prepare_ashare_three_year import BASE, ROOT
from prepare_four_strategy_2026 import save


def main():
    output = BASE / "workflow-v1"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    owned = []
    names = [
        "train_ashare_three_year.py",
        "continue_ashare_three_year.py",
        "prepare_ashare_three_year_warmup.py",
    ]
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.info["name"] != "python.exe":
            continue
        cmd = process.info["cmdline"] or []
        if any(
            any(name == os.path.basename(arg.replace("\\", "/")) for name in names) for arg in cmd
        ):
            raise RuntimeError("EXISTING_STUDY_WORKER:" + str(process.pid))
    env = {**os.environ, "PYTHONHASHSEED": "42", "TQDM_DISABLE": "1"}

    def launch(label, script, args):
        log = output / (stamp + "-" + label + ".log")
        with log.open("a", encoding="utf-8") as stream:
            p = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "scripts" / script), *args],
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS,
            )
        # The venv Windows launcher can wrap the actual interpreter; adopt its child.
        actual = psutil.Process(p.pid)
        for _ in range(20):
            if p.poll() is not None:
                raise RuntimeError("LAUNCH_EXITED:" + label + ":" + str(p.returncode))
            children = actual.children(recursive=True)
            matches = [
                c for c in children if c.name() == "python.exe" and script in " ".join(c.cmdline())
            ]
            if matches:
                actual = matches[-1]
                break
            time.sleep(0.1)
        actual.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        owned.append(
            {
                "label": label,
                "pid": actual.pid,
                "created": actual.create_time(),
                "command": actual.cmdline(),
                "log": str(log),
            }
        )
        save(
            output / "detached-launch.json",
            {"utc": stamp, "pid": os.getpid(), "workers": owned, "production_modified": False},
        )
        return actual.pid

    warmup = None
    if not (BASE / "warmup-v1/results.json").exists():
        warmup = launch("warmup", "prepare_ashare_three_year_warmup.py", [])
    groups = [["2025Q3", "2025Q4", "2026Q1"], ["2026Q2", "2026Q3", "2026Q4"]]
    training = []
    for folds in groups:
        if all(
            (BASE / "training-v1" / f / c / "results.json").exists()
            for f in folds
            for c in ["rnn_rank_cap", "tra_rank_cap"]
        ):
            continue
        pid = launch(
            "gpu-" + folds[0],
            "train_ashare_three_year.py",
            ["--cases", "rnn_rank_cap", "tra_rank_cap", "--folds", *folds],
        )
        training.append(pid)
        # Stagger initial feature allocation, but don't change fitting or prediction.
        progress = (
            BASE / "training-v1" / ("progress-rnn_rank_cap-tra_rank_cap-" + folds[0] + ".json")
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                raise RuntimeError("GPU_EXITED_DURING_PREPARATION:" + str(pid))
            if progress.exists():
                record = json.loads(progress.read_bytes())
                if record.get("pid") == pid and record.get("phase") != "FITTING":
                    break
            time.sleep(3)
    if not training:
        complete_nonfinancial = [
            p
            for p in (BASE / "training-v1").glob("*/*/results.json")
            if p.parent.name != "financial_rank_cap"
        ]
        if len(complete_nonfinancial) != 39:
            raise RuntimeError("NONFINANCIAL_MATRIX_NEEDS_EARLIER_COHORT_RECOVERY")
    args = ["--financial-pid", "0", "--training-pids", *map(str, training)]
    if warmup and not (BASE / "warmup-v1/results.json").exists():
        args += ["--warmup-pid", str(warmup)]
    launch("queue", "continue_ashare_three_year.py", args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save(
            BASE / "workflow-v1/detached-launch-failure.json",
            {
                "status": "FAILED",
                "pid": os.getpid(),
                "error": str(exc),
                "utc": datetime.now(UTC).isoformat(),
                "production_modified": False,
            },
        )
        raise
