"""Sequential 10/20bps N3 replay. No scheduler, login, or production trading."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from moneymore.research.ashare_mainline import sha256

ROOT = Path(__file__).resolve().parents[1]


def main(output, solver_cache=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        ROOT / "scripts" / name
        for name in (
            "replay_ashare_mainline.py",
            "solve_ashare_mainline_cvx.py",
            "research_ashare_paper_routes.py",
            "report_ashare_mainline_replay.py",
        )
    ] + [
        ROOT / "src/moneymore/research/mainline_execution.py",
        ROOT / "configs/ashare_mainline_n3.json",
    ]
    hashes = {str(p): sha256(p) for p in sources}
    state = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "completed_costs": [],
        "source_hashes": hashes,
        "verified_solver_cache": str(solver_cache) if solver_cache else None,
    }
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONHASHSEED="42",
    )

    def save():
        state["updated_utc"] = datetime.now(UTC).isoformat()
        (output / "progress.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    save()
    runs = []
    try:
        for bps in (10, 20):
            if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
                raise ValueError("FROZEN_SOURCE_CHANGED_BETWEEN_COST_RUNS")
            child = output / f"cost{bps}"
            runs.append(child)
            state["active_cost"] = bps
            state["active_progress"] = str(child / f"bps{bps}/progress.json")
            save()
            with (output / f"cost{bps}.log").open("w", encoding="utf-8") as log:
                extra = ["--solver-cache", str(solver_cache / f"cost{bps}")] if solver_cache else []
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/replay_ashare_mainline.py"),
                        "--output",
                        str(child),
                        "--bps",
                        str(bps),
                    ]
                    + extra,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            state["completed_costs"].append(bps)
            save()
            print(json.dumps({"cost_complete": bps}), flush=True)
        from report_ashare_mainline_replay import main as report

        report(runs, output / "report")
        state.update(status="COMPLETE", active_cost=None)
    except BaseException as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver-cache", type=Path)
    args = parser.parse_args()
    main(args.output, args.solver_cache)
