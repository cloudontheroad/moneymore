"""Fixed-prediction rank-preserving A-share experiment, isolated accounts only."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import research_ashare_paper_routes as engine
from replay_ashare_mainline import ROOT, Planner
from threadpoolctl import threadpool_limits

from moneymore.research.ashare_mainline import sha256
from moneymore.research.mainline_execution import physical_open_plan
from moneymore.research.mature_multifactor import topk_dropout_targets
from moneymore.research.rank_preserving import allocate_ranked, sector_relative_scores

CONFIG = ROOT / "configs/ashare_rank_preserving_v1.json"


class RankPlanner(Planner):
    def __init__(self, output, contract, loaded):
        # Keep exactly the N3 common universe, including its four forecast intersections.
        super().__init__(output, {**contract, "models": contract["eligibility_models"]}, loaded)

    def __call__(self, **c):
        case, allocator = c["kind"].split("--")
        if allocator == "topk":
            return super().__call__(**c)
        day, signal, held, nh = c["day"], c["signal"], c["held"], c["night_holdings"]
        dest = self.output / "decisions" / c["kind"]
        dest.mkdir(parents=True, exist_ok=True)
        exits = set(held) - c["members"]
        previous = sorted(p for p in dest.glob("*.json") if p.stem < day)
        retry = bool(previous and json.loads(previous[-1].read_bytes()).get("retry_next_session"))

        def save(status, reason=None, **details):
            (dest / f"{day}.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "reason": reason,
                        "retry_next_session": bool(reason),
                        "signal_date": signal,
                        "execution_date": day,
                        **details,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        def protected(reason, **details):
            feasible, _, clips = physical_open_plan(
                {s: q for s, q in held.items() if s not in exits},
                c["snapshot"],
                c["opened"],
                c["buy"],
                c["sell"],
                c["config"],
            )
            save(
                "BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED",
                reason,
                exit_symbols=sorted(exits),
                clips=clips,
                **details,
            )
            return feasible, reason

        if c["index"] % self.contract["rebalance_sessions"] and held and not retry:
            if exits:
                feasible, _, clips = physical_open_plan(
                    {s: q for s, q in held.items() if s not in exits},
                    c["snapshot"],
                    c["opened"],
                    c["buy"],
                    c["sell"],
                    c["config"],
                )
                save(
                    "MANDATORY_EXITS_ONLY",
                    exit_symbols=sorted(exits),
                    clips=clips,
                    pending_exit_symbols=sorted(exits & set(feasible)),
                )
                return feasible, None
            save("NOT_SCHEDULED")
            return held, None

        covariance, attrs, risk_audit = self.risk(signal)
        rights = c["night"]["unsettled_stock_rights"]
        economic_held = set(nh) | set(rights)
        missing = economic_held - set(covariance.index)
        if missing:
            return protected("HELD_RISK_HISTORY_MISSING", missing_symbols=sorted(missing))
        frozen = (economic_held & c["members"]) - set(attrs.index)
        frozen |= {s for s in exits if not c["sell"].get(s, False)}
        symbols = sorted(set(attrs.index) | economic_held)
        equity = c["night"]["equity"]
        weights = pd.Series(
            {s: (nh.get(s, 0) + rights.get(s, 0)) * c["last"][s] / equity for s in economic_held},
            dtype=float,
        ).reindex(symbols, fill_value=0.0)
        stock_exposure = min(
            self.contract["stock_exposure"], 1 - c["night"].get("receivable_cash", 0.0) / equity
        )
        industries = attrs.industry.reindex(symbols)
        for symbol in industries.index[industries.isna()]:
            past = self.industry_history.get(symbol, pd.DataFrame(columns=["date", "industry"]))
            past = past.loc[past.date.le(signal)]
            if past.empty:
                return protected("HELD_INDUSTRY_MISSING", missing_symbols=[symbol])
            industries[symbol] = past.iloc[-1].industry
        benchmark = (attrs.total_mv / attrs.total_mv.sum() * stock_exposure).reindex(
            symbols, fill_value=0
        )
        benchmark_industry = benchmark.groupby(industries).sum()
        scores = self.scores[(case, signal)].reindex(attrs.index)
        if allocator.startswith("sector_"):
            scores = sector_relative_scores(scores, attrs.industry)
        if scores.std(ddof=0) < 1e-10:
            return protected("DEGENERATE_SCORE")
        allocation = None
        if allocator == "sector_topk":
            active = weights.reindex(attrs.index)
            active = active.loc[active.gt(0)]
            free = max(0.0, stock_exposure - float(weights.reindex(sorted(frozen)).sum()))
            selected = (
                topk_dropout_targets(
                    scores,
                    active / max(free, 1e-12),
                    topk=max(1, self.contract["topk"] - len(frozen)),
                    n_drop=self.contract["n_drop"],
                )
                if free > 0
                else {}
            )
            targets = {s: w * free for s, w in selected.items()}
        else:
            locked = weights.reindex(sorted(frozen))
            locked_industry = locked.groupby(industries.reindex(locked.index)).sum()
            candidates = scores.drop(index=list(frozen), errors="ignore")
            previous_symbols = set(nh) & set(candidates.index)
            try:
                targets, allocation = allocate_ranked(
                    candidates,
                    industries.reindex(candidates.index),
                    benchmark_industry,
                    previous_symbols,
                    slots=self.contract["topk"],
                    slot_weight=stock_exposure / self.contract["topk"],
                    industry_active_limit=1.0
                    if allocator == "rank_equal"
                    else self.contract["industry_active_upper_planning"],
                    frozen_industry_weights=locked_industry.to_dict(),
                    max_optional_exits=self.contract["n_drop"]
                    if allocator.endswith("buffer")
                    else None,
                )
            except ValueError as error:
                if str(error) not in {
                    "FROZEN_EXPOSURE_EXCEEDS_TARGET",
                    "FROZEN_INDUSTRY_CAP_VIOLATION",
                }:
                    raise
                return protected("RANK_ALLOCATION_" + str(error))
            if allocation["status"] != "OK":
                return protected("RANK_ALLOCATION_" + allocation["status"], allocation=allocation)
        targets.update(weights.reindex(sorted(frozen)).to_dict())
        desired = {}
        for symbol, weight in targets.items():
            if weight <= 0 or symbol not in c["last"]:
                continue
            if symbol in frozen or (symbol in nh and abs(weight - weights.get(symbol, 0)) < 1e-9):
                desired[symbol] = held.get(symbol, 0)
                continue
            quantity = weight * equity / c["last"][symbol]
            for event in (
                c["actions"]
                .loc[c["actions"].symbol.eq(symbol) & c["actions"].ex_date.eq(pd.Timestamp(day))]
                .itertuples()
            ):
                quantity *= 1 + event.stock_ratio
            quantity = max(0.0, quantity - c["snapshot"]["unsettled_stock_rights"].get(symbol, 0))
            desired[symbol] = int(quantity / 100) * 100
        feasible, cash, clips = physical_open_plan(
            desired,
            c["snapshot"],
            c["opened"],
            c["buy"],
            c["sell"],
            c["config"],
        )
        open_postcheck = None
        if allocation is not None:
            rights = c["snapshot"]["unsettled_stock_rights"]
            values = pd.Series(
                {
                    s: (feasible.get(s, 0) + rights.get(s, 0)) * c["opened"][s]
                    for s in set(feasible) | set(rights)
                },
                dtype=float,
            )
            post_equity = cash + values.sum() + c["snapshot"].get("receivable_cash", 0)
            actual = values.reindex(symbols, fill_value=0) / post_equity
            industry_actual = actual.groupby(industries).sum()
            limits = benchmark_industry + (
                1.0
                if allocator == "rank_equal"
                else self.contract["industry_active_upper_execution"]
            )
            violation = float((industry_actual - limits).max())
            open_postcheck = {
                "passed": bool(
                    violation <= 1e-5 and actual.max() <= self.contract["max_stock_weight"] + 1e-5
                ),
                "actual_stock_exposure": float(actual.sum()),
                "maximum_stock_weight": float(actual.max()),
                "maximum_industry_violation": violation,
                "stock_limit": self.contract["max_stock_weight"],
                "industry_weights": industry_actual.to_dict(),
                "industry_limits": limits.to_dict(),
            }
            if not open_postcheck["passed"]:
                return protected(
                    "INTEGER_OPEN_RISK_POSTCHECK",
                    open_postcheck=open_postcheck,
                    allocation=allocation,
                )
        save(
            "OPEN_PLAN_VERIFIED",
            risk_audit=risk_audit,
            allocation=allocation,
            frozen_symbols=sorted(frozen),
            target_quantities=desired,
            executable_quantities=feasible,
            open_clips=clips,
            open_postcheck=open_postcheck,
            next_open_used_for_selection=False,
        )
        return feasible, None


def source_files():
    return [
        Path(__file__),
        CONFIG,
        ROOT / "scripts/replay_ashare_mainline.py",
        Path(engine.__file__),
        ROOT / "src/moneymore/research/rank_preserving.py",
        ROOT / "src/moneymore/research/mainline_execution.py",
        ROOT / "src/moneymore/research/mature_multifactor.py",
        ROOT / "src/moneymore/execution/paper.py",
        ROOT / "src/moneymore/research/replay_actions.py",
    ]


def replay(output, bps, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads(CONFIG.read_bytes())
    (output / "contract.json").write_bytes(CONFIG.read_bytes())
    sources = source_files()
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    for path in sources:
        dest = output / "sources" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
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
        loaded = engine.load_inputs()
        planner = RankPlanner(output, contract, loaded)
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
            if any(sha256(Path(p)) != h for p, h in frozen.items()):
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
