"""Bounded isolated workflow: fixed ten seeds, at most two training processes."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from filelock import FileLock
from prepare_hist_reproduction import CFG, OUT, ROOT, UPSTREAM, digest, write_json
from replay_hist_reproduction import replay_complete, report_results


def training_complete(folder, seed):
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    row = json.loads(marker.read_text(encoding="utf-8"))
    valid = (row.get("status") == "training_and_prediction_complete" and row.get("seed") == seed
             and row.get("reload_exact_match") is True
             and row.get("source_model_sha256") == digest(UPSTREAM / "model.py")
             and all((folder / name).is_file() for name in
                     ("model.pt", "pred-train.parquet", "pred-valid.parquet", "pred-test.parquet")))
    if not valid:
        raise ValueError(f"Seed {seed} completion marker invalid; refusing to overwrite trained artifacts")
    return True


def retryable_failure(log_tail):
    # Deterministic accounting/data/validation failures must stop and retain evidence.
    if any(word in log_tail for word in ("AssertionError", "ValueError", "Cash solvency failed")):
        return False
    return any(word in log_tail for word in ("PermissionError", "TimeoutError", "BrokenPipeError", "[WinError 32]"))


def terminal_status(results, pending):
    if pending:
        return "running"
    return "complete" if len(results) == len(CFG["seeds"]) and all(
        row["status"] == "complete" for row in results) else "failed_requires_attention"


def input_snapshot():
    snapshot = OUT / "recovery-inputs-20261114.json"
    if snapshot.exists():
        record = json.loads(snapshot.read_text(encoding="utf-8"))
    else:
        paths = [OUT / "manifest.json", UPSTREAM / "model.py"]
        for seed in CFG["seeds"]:
            folder = OUT / "runs" / f"seed-{seed:02d}"
            paths += [folder / name for name in ("model.pt", "pred-test.parquet", "complete.json")
                      if (folder / name).exists()]
        record = {"sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
                  "prior_workflow": json.loads((OUT / "workflow.json").read_text(encoding="utf-8"))}
        write_json(snapshot, record)
    for relative, expected in record["sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen model/input changed: {relative}")
    return len(record["sha256"])


def seed_job(seed):
    folder = OUT / "runs" / f"seed-{seed:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, arguments in (
        ("training", ["scripts/train_hist_reproduction.py", "--seeds", str(seed)]),
        ("replay", ["scripts/replay_hist_reproduction.py", "--seed", str(seed), "--no-summary"]),
    ):
        ready = training_complete(folder, seed) if name == "training" else replay_complete(folder)
        if ready:
            print("SKIP verified", name, seed, flush=True)
            continue
        for attempt in range(1, 3):
            path = folder / f"{name}-recovery-attempt-{attempt}.log"
            with path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                write_json(folder / "worker.json", {"pid": process.pid, "stage": name, "seed": seed,
                    "status": "running", "attempt": attempt, "log": str(path.relative_to(ROOT)),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                code = process.wait()
            if code == 0:
                ready = training_complete(folder, seed) if name == "training" else replay_complete(folder)
                if not ready:
                    raise ValueError(f"Seed {seed} {name} exited without verified completion artifacts")
                break
            tail = path.read_bytes()[-8000:].decode("utf-8", errors="replace")
            if attempt == 2 or not retryable_failure(tail):
                failure = {"seed": seed, "status": "failed", "stage": name, "exit_code": code,
                           "attempts": attempt, "log": str(path.relative_to(ROOT))}
                write_json(folder / "worker.json", failure)
                return failure
            time.sleep(2)
    write_json(folder / "worker.json", {"seed": seed, "status": "complete", "stage": "verified",
                                       "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return {"seed": seed, "status": "complete"}


def main():
    with FileLock(str(OUT / "workflow.lock"), timeout=0):
        verified_inputs = input_snapshot()
        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(seed_job, seed): seed for seed in CFG["seeds"]}
            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    seed = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:  # noqa: BLE001 -- persist failure instead of losing queued research jobs
                        result = {"seed": seed, "status": "failed", "exception_type": type(error).__name__}
                    results.append(result)
                    print(result, flush=True)
                write_json(OUT / "workflow.json", {"pid": os.getpid(), "status": terminal_status(results, pending),
                    "completed": results, "pending": sorted(pending.values()),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "real_orders": False})
        assert input_snapshot() == verified_inputs
        summary = report_results()
        status = terminal_status(results, [])
        if summary["status"] != "complete":
            status = "failed_requires_attention"
        write_json(OUT / "workflow.json", {"pid": os.getpid(), "status": status,
            "completed": results, "pending": [], "verified_unchanged_inputs": verified_inputs,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "real_orders": False})
        print("FINAL", summary["status"], "completed", summary["completed_backtests"], flush=True)
        return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
