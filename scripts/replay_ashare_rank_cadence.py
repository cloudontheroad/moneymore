"""Frozen raw-ranking cadence x membership-buffer experiment; isolated accounts."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import replay_ashare_rank_preserving as previous
from threadpoolctl import threadpool_limits

from moneymore.research.ashare_mainline import sha256

ROOT = previous.ROOT
CONFIG = ROOT / "configs/ashare_rank_cadence_v1.json"
CASE_SPECS = {
    "rank_cap": {"rebalance_sessions": 20, "max_optional_exits": None},
    "rank_cap_buffer": {"rebalance_sessions": 20, "max_optional_exits": 5},
    "rank_cap_fast": {"rebalance_sessions": 10, "max_optional_exits": None},
    "rank_cap_fast_buffer": {"rebalance_sessions": 10, "max_optional_exits": 5},
}


def validate_contract(contract):
    fixed = {
        "models": ["doubleensemble_full"],
        "eligibility_models": [
            "ridge_three_scores",
            "ridge_full",
            "doubleensemble_alpha158",
            "doubleensemble_full",
        ],
        "allocators": list(CASE_SPECS),
        "case_specs": CASE_SPECS,
        "window": ["20260101", "20260930"],
        "initial_cash": 10000000,
        "slippage_bps": [10, 20],
        "rebalance_sessions": 20,
        "risk_sessions": 120,
        "risk_horizon": 20,
        "stock_exposure": 0.995,
        "max_stock_weight": 0.05,
        "industry_active_upper_planning": 0.08,
        "industry_active_upper_execution": 0.10,
        "topk": 50,
        "n_drop": 5,
        "real_orders_enabled": False,
        "production_enabled": False,
    }
    for field, value in fixed.items():
        if contract.get(field) != value:
            raise ValueError("CADENCE_CONTRACT_CHANGED:" + field)


class CadencePlanner(previous.RankPlanner):
    """Reuse the frozen ranking/execution path, changing only registered cadence.

    The engine invokes accounts sequentially. The contract is restored even on
    exceptions, so no case can leak its scheduling period to the next account.
    The closed case map explicitly verifies the inherited raw-score/buffer
    dispatch instead of permitting arbitrary suffix-derived strategies.
    """

    def __init__(self, output, contract, loaded):
        validate_contract(contract)
        super().__init__(output, contract, loaded)

    def __call__(self, **context):
        model, allocator = context["kind"].split("--")
        if model != "doubleensemble_full" or allocator not in CASE_SPECS:
            raise ValueError("UNKNOWN_CADENCE_CASE")
        spec = CASE_SPECS[allocator]
        if self.contract.get("case_specs") != CASE_SPECS:
            raise ValueError("CADENCE_CASE_SPECS_CHANGED")
        if allocator.startswith("sector_") or allocator.endswith("buffer") != (
            spec["max_optional_exits"] == 5
        ):
            raise ValueError("CADENCE_INHERITED_DISPATCH_MISMATCH")
        original = self.contract
        try:
            self.contract = {**original, "rebalance_sessions": spec["rebalance_sessions"]}
            result = super().__call__(**context)
            path = self.output / "decisions" / context["kind"] / (context["day"] + ".json")
            decision = json.loads(path.read_bytes())
            decision.update(
                cadence_policy=spec,
                execution_index=context["index"],
                scheduled_rebalance=context["index"] % spec["rebalance_sessions"] == 0,
            )
            path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
            return result
        finally:
            self.contract = original


def source_files():
    extra = [
        Path(__file__),
        CONFIG,
        ROOT / "scripts/report_ashare_rank_cadence.py",
        ROOT / "scripts/report_ashare_rank_preserving.py",
        ROOT / "scripts/report_ashare_mainline_replay.py",
        ROOT / "scripts/finalize_ashare_mainline_n3.py",
        ROOT / "src/moneymore/research/period_statistics.py",
        ROOT / "tests/test_rank_cadence_execution.py",
        ROOT / "tests/test_rank_cadence_report.py",
        ROOT / "tests/test_rank_preserving_execution.py",
    ]
    return list(dict.fromkeys(previous.source_files() + extra))


def replay(output, bps, smoke=False):
    contract = json.loads(CONFIG.read_bytes())
    validate_contract(contract)
    output.mkdir(parents=True, exist_ok=False)
    (output / "contract.json").write_bytes(CONFIG.read_bytes())
    hashes = {}
    for source in source_files():
        relative = source.relative_to(ROOT)
        hashes[str(relative)] = sha256(source)
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    state = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "bps": bps,
        "smoke": smoke,
        "source_hashes": hashes,
        "started_utc": datetime.now(UTC).isoformat(),
    }
    started = time.monotonic()

    def save():
        state.update(
            elapsed_seconds=round(time.monotonic() - started, 2),
            updated_utc=datetime.now(UTC).isoformat(),
        )
        (output / "runtime.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    save()
    try:
        engine = previous.engine
        loaded = engine.load_inputs()
        planner = CadencePlanner(output, contract, loaded)
        engine.load_inputs = lambda: loaded
        engine.KINDS = ["doubleensemble_full--" + a for a in contract["allocators"]]
        engine.OUT = output
        engine.run(
            bps,
            smoke=smoke,
            execution_planner=planner,
            config_overrides={
                "max_position_weight": contract["max_stock_weight"],
                "initial_cash": contract["initial_cash"],
            },
        )
        state["status"] = "COMPLETE"
    except BaseException as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()


def matrix(output):
    validate_contract(json.loads(CONFIG.read_bytes()))
    output.mkdir(parents=True, exist_ok=False)
    frozen = {str(p): sha256(p) for p in source_files()}
    state = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "completed_costs": [],
        "source_hashes": frozen,
    }

    def save():
        state["updated_utc"] = datetime.now(UTC).isoformat()
        (output / "progress.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    save()
    try:
        for bps in (10, 20):
            if any(sha256(Path(p)) != digest for p, digest in frozen.items()):
                raise ValueError("FROZEN_SOURCE_CHANGED")
            state["active_cost"] = bps
            save()
            with (output / f"cost{bps}.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__)),
                        "--output",
                        str(output / f"cost{bps}"),
                        "--bps",
                        str(bps),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=dict(
                        os.environ,
                        OMP_NUM_THREADS="2",
                        OPENBLAS_NUM_THREADS="2",
                        MKL_NUM_THREADS="2",
                    ),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            state["completed_costs"].append(bps)
            save()
            print(json.dumps({"cost_complete": bps}), flush=True)
        state.update(status="RUNS_COMPLETE", active_cost=None)
    except BaseException as error:
        state.update(status="FAILED", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bps", type=int, choices=[10, 20])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.bps is None:
            matrix(args.output.resolve())
        else:
            replay(args.output.resolve(), args.bps, args.smoke)
