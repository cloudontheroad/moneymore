"""One finite local research batch: await current replay, then risk/cost diagnostics."""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import paired_return_robustness, read_json


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-release"
    directory = base / "corrected-suites" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "WAITING_FOR_REFERENCE",
        "output_directory": str(directory),
        "real_orders_enabled": False,
        "production_changed": False,
        "recurring_task": False,
        "stages": {},
    }
    pointer = base / "corrected-execution-suite.json"
    # A single suite owner; OS releases the lock on process exit. Never retry failed children.
    with (base / "corrected-suite.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_diagnostic(pointer, report)
            deadline = time.monotonic() + 10800
            reference_path = base / "corrected-daily-replay.json"
            expected_source = read_json(reference_path)["output_directory"]
            while True:
                reference = read_json(reference_path)
                if reference["output_directory"] != expected_source:
                    raise ValueError("REFERENCE_CHANGED_WHILE_WAITING")
                if reference["status"] == "COMPLETED_RESEARCH_NOT_PUBLISHED":
                    break
                if reference["status"] not in {"RUNNING", "REPLAYING"}:
                    raise ValueError("REFERENCE_NOT_RUNNING_OR_COMPLETED:" + reference["status"])
                if time.monotonic() >= deadline:
                    raise TimeoutError("REFERENCE_WAIT_EXCEEDED_THREE_HOURS")
                time.sleep(5)
            report["stages"]["reference"] = reference
            protocol = yaml.safe_load(
                (root / "configs/baseline_v2_release_vendor.yaml").read_text(encoding="utf-8")
            )
            source = Path(reference["output_directory"])
            report["paired_return_robustness"] = paired_return_robustness(
                pd.read_parquet(source / "fixed-equity.parquet"),
                pd.read_parquet(source / "learned-equity.parquet"),
                protocol,
            )
            # Diagnostics run even when statistical advantage fails; failure is never promoted.
            for scenario in ("native_risk", "cost_10bps"):
                report["status"] = "RUNNING_" + scenario.upper()
                write_diagnostic(pointer, report)
                with (directory / (scenario + ".log")).open("w", encoding="utf-8") as output:
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(root / "scripts/stress_corrected_v2.py"),
                            "--scenario",
                            scenario,
                        ],
                        cwd=root,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=10800,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        check=False,
                    )
                if result.returncode:
                    raise RuntimeError(
                        "ISOLATED_SCENARIO_FAILED:" + scenario + ":" + str(result.returncode)
                    )
                outcome = read_json(base / ("corrected-" + scenario + ".json"))
                if (
                    outcome["source_directory"] != expected_source
                    or outcome["status"] != "COMPLETED_RESEARCH_NOT_PUBLISHED"
                ):
                    raise ValueError("SCENARIO_SOURCE_OR_COMPLETION_MISMATCH")
                report["stages"][scenario] = outcome
            report["status"] = "COMPLETED_RESEARCH_NOT_PUBLISHED"
        except Exception as exc:
            report.update(status="FAILED", error=type(exc).__name__ + ":" + str(exc))
            raise
        finally:
            write_diagnostic(directory / "report.json", report)
            write_diagnostic(pointer, report)
        print(json.dumps(report, default=str), flush=True)


if __name__ == "__main__":
    main()
