"""N3: fixed predictions, common A-share broker, three allocation mechanisms."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import research_ashare_paper_routes as engine
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from moneymore.research.ashare_mainline import sha256
from moneymore.research.cost_aware import transaction_rates
from moneymore.research.mainline_execution import (
    physical_open_plan,
    risk_window_with_halt_seed,
    run_native,
    validate_calibration,
)
from moneymore.research.mature_multifactor import topk_dropout_targets

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "state/ashare-mainline-20261112"


class Planner:
    def __init__(self, output, contract, loaded, solver_cache=None):
        self.output, self.contract = output, contract
        self.solver_cache = solver_cache
        summary = BASE / "n2-summary-v1"
        manifest = json.loads((summary / "results.json").read_bytes())
        for name, digest in manifest["hashes"].items():
            if sha256(summary / name) != digest:
                raise ValueError("N2_SUMMARY_CHANGED:" + name)
        predictions = pd.read_parquet(summary / "predictions.parquet")
        self.scores = {
            (case, day): g.set_index("symbol").score
            for (case, day), g in predictions.groupby(["case", "date"])
        }
        self.calibration = {
            (r["case"], r["month"]): r
            for r in json.loads((summary / "calibration.json").read_bytes())
        }
        panel_path = BASE / "panel-v2/panel.parquet"
        training = json.loads((BASE / "registered-matrix-v1/runtime.json").read_bytes())
        if sha256(panel_path) != training["panel_sha256"]:
            raise ValueError("N1_PANEL_CHANGED_SINCE_TRAINING")
        panel = pd.read_parquet(panel_path, columns=["date", "symbol", "industry", "total_mv"])
        self.attributes = {day: g.set_index("symbol") for day, g in panel.groupby("date")}
        self.industry_history = {
            symbol: g.loc[g.industry.notna(), ["date", "industry"]].sort_values("date")
            for symbol, g in panel.groupby("symbol")
        }
        bars, self.membership, _, _, _, _, _, official, _ = loaded
        self.prices = (
            bars.assign(p=bars.close * bars.adj_factor)
            .pivot(index="trade_date", columns="ts_code", values="p")
            .sort_index()
        )
        self.calendar = pd.DatetimeIndex(
            pd.to_datetime(
                (ROOT / "state/mature-multifactor/p1-native-20261103-a/provider/calendars/day.txt")
                .read_text()
                .splitlines()
            )
        ).strftime("%Y%m%d")
        self.prices = self.prices.reindex(self.calendar)
        sessions, signals = loaded[2], loaded[3]
        expected = [d for d in self.calendar if contract["window"][0] <= d <= contract["window"][1]]
        if sessions != expected or any(
            signals[d] != self.calendar[list(self.calendar).index(d) - 1] for d in sessions
        ):
            raise ValueError("EXECUTION_CALENDAR_OR_SIGNAL_LAG_MISMATCH")
        proofs = [data["suspend_d"] for data in official.values() if not data["suspend_d"].empty]
        self.proofs = (
            pd.concat(proofs, ignore_index=True).drop_duplicates() if proofs else pd.DataFrame()
        )
        self.cache = {}

    def risk(self, signal):
        if signal in self.cache:
            return self.cache[signal]
        dates = [d for d in self.calendar if d <= signal][-(self.contract["risk_sessions"] + 1) :]
        if len(dates) != self.contract["risk_sessions"] + 1 or dates[-1] != signal:
            raise ValueError("INCOMPLETE_RISK_CALENDAR")
        # Risk coverage is independent of today's prediction/cap availability.
        # Previously seen members cover still-owned exits without future universe information.
        known = set().union(*(m for day, m in self.membership.items() if day <= signal))
        history = self.prices.reindex(columns=sorted(known))
        proof = (
            self.proofs.loc[self.proofs.trade_date.astype(str).le(signal)]
            if not self.proofs.empty
            else self.proofs
        )
        history, carries = risk_window_with_halt_seed(history, proof, dates)
        returns = history.pct_change(fill_method=None).iloc[1:]
        attrs = self.attributes[signal]
        risk_eligible = [s for s in returns if np.isfinite(returns[s]).all()]
        eligible = [
            s
            for s in risk_eligible
            if s in self.membership[signal]
            and s in attrs.index
            and pd.notna(attrs.loc[s, "industry"])
            and np.isfinite(attrs.loc[s, "total_mv"])
            and attrs.loc[s, "total_mv"] > 0
            and all(s in self.scores[(case, signal)].index for case in self.contract["models"])
        ]
        if len(eligible) < 100:
            raise ValueError("INSUFFICIENT_SHARED_RISK_UNIVERSE")
        covariance = (
            LedoitWolf().fit(returns[risk_eligible].to_numpy()).covariance_
            * self.contract["risk_horizon"]
        )
        covariance = pd.DataFrame(covariance, index=risk_eligible, columns=risk_eligible)
        attrs = attrs.loc[eligible]
        report = {
            "signal_date": signal,
            "risk_start": dates[0],
            "risk_end": dates[-1],
            "eligible": len(eligible),
            "risk_eligible": len(risk_eligible),
            "members": len(self.membership[signal]),
            "excluded": sorted(self.membership[signal] - set(eligible)),
            "official_halt_carries": carries,
        }
        self.cache.clear()  # twelve accounts reuse today's matrix; don't retain every dense matrix
        self.cache[signal] = covariance, attrs, report
        return self.cache[signal]

    def __call__(self, **c):
        case, allocator = c["kind"].split("--")
        day, signal, held, nh = c["day"], c["signal"], c["held"], c["night_holdings"]
        dest = self.output / "decisions" / c["kind"]
        dest.mkdir(parents=True, exist_ok=True)
        exits = set(held) - c["members"]
        previous = sorted(p for p in dest.glob("*.json") if p.stem < day)
        retry = bool(previous and json.loads(previous[-1].read_bytes()).get("retry_next_session"))

        def save(status, reason=None, **extra):
            (dest / f"{day}.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "reason": reason,
                        "retry_next_session": bool(reason),
                        "signal_date": signal,
                        "execution_date": day,
                        **extra,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        def protected(reason, **details):
            # Data failure is not an instruction to sell a still-eligible holding.
            quantities = {s: q for s, q in held.items() if s not in exits}
            feasible, _, clips = physical_open_plan(
                quantities, c["snapshot"], c["opened"], c["buy"], c["sell"], c["config"]
            )
            save("BLOCKED_OPTIONAL_TRADES_MANDATORY_EXITS_CHECKED", reason, clips=clips, **details)
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
        covariance, attrs, audit = self.risk(signal)
        rights = c["night"]["unsettled_stock_rights"]
        economic_held = set(nh) | set(rights)
        missing_risk = economic_held - set(covariance.index)
        if missing_risk:
            return protected(
                "HELD_RISK_HISTORY_MISSING", missing_symbols=sorted(missing_risk), risk_audit=audit
            )
        # Missing scores/cap for a held member are NOT exits and must not freeze other holdings.
        frozen = (economic_held & c["members"]) - set(attrs.index)
        frozen |= {s for s in exits if not c["sell"].get(s, False)}
        symbols = sorted(set(attrs.index) | economic_held)
        full_scores = self.scores[(case, signal)]
        equity = c["night"]["equity"]
        weights = pd.Series(
            {s: (nh.get(s, 0) + rights.get(s, 0)) * c["last"][s] / equity for s in economic_held},
            dtype=float,
        ).reindex(symbols, fill_value=0.0)
        reserve = c["night"].get("receivable_cash", 0.0) / equity
        stock_exposure = min(self.contract["stock_exposure"], 1 - reserve)
        cap = attrs.total_mv / attrs.total_mv.sum() * stock_exposure
        benchmark = cap.reindex(symbols, fill_value=0.0)
        current_score = full_scores.reindex(attrs.index)
        if current_score.std(ddof=0) < 1e-10:
            return protected("DEGENERATE_SCORE")
        risk_bounds = None
        if allocator == "topk":
            active = weights.reindex(attrs.index)
            active = active.loc[active.gt(0)]
            free = max(0.0, stock_exposure - float(weights.reindex(sorted(frozen)).sum()))
            targets = (
                topk_dropout_targets(
                    current_score,
                    active / max(free, 1e-12),
                    topk=max(1, self.contract["topk"] - len(frozen)),
                    n_drop=self.contract["n_drop"],
                )
                if free > 0
                else {}
            )
            targets = {s: w * free for s, w in targets.items()}
            targets.update(weights.reindex(sorted(frozen)).to_dict())
            continuous = {"status": "TOPK_REFERENCE", "next_open_used_for_selection": False}
        else:
            cal = self.calibration[(case, day[:6])]
            validate_calibration(cal, signal, day)
            z = ((full_scores - full_scores.mean()) / full_scores.std(ddof=0)).clip(-3, 3)
            mu = (z * cal["beta_conservative"]).reindex(symbols, fill_value=0.0)
            lower = pd.Series({s: rights.get(s, 0) * c["last"].get(s, 0) / equity for s in symbols})
            upper = pd.Series(self.contract["max_stock_weight_planning"], index=symbols)
            for symbol in set(symbols) - set(attrs.index):
                upper[symbol] = lower[symbol]  # sell tradable shares, never unlisted stock rights
            for symbol in frozen:
                upper[symbol] = lower[symbol] = weights[symbol]
            if any(weights[s] > self.contract["max_stock_weight"] for s in frozen):
                return protected("FROZEN_HOLDING_EXCEEDS_RISK_CAP", frozen_symbols=sorted(frozen))
            if (lower > upper + 1e-8).any():
                return protected("LOCKED_RIGHTS_EXCEED_STOCK_CAP")
            industries = attrs.industry.reindex(symbols)
            for symbol in industries.index[industries.isna()]:
                known = self.industry_history.get(
                    symbol, pd.DataFrame(columns=["date", "industry"])
                )
                past = known.loc[known.date.le(signal)]
                if past.empty:
                    return protected("HELD_INDUSTRY_MISSING", missing_symbols=[symbol])
                industries[symbol] = past.iloc[-1].industry
            exposure = pd.get_dummies(industries, dtype=float)
            bench_industry = exposure.T @ benchmark
            limits = bench_industry + self.contract["industry_active_upper_planning"]
            risk_bounds = (
                exposure,
                bench_industry + self.contract["industry_active_upper_execution"],
            )
            ordinary = self.contract["ordinary_one_way_turnover_limit"]
            turnover = min(
                1.0, ordinary + float(weights.reindex(list(exits), fill_value=0.0).sum())
            )
            if not economic_held:
                turnover = self.contract["bootstrap_one_way_turnover_limit"]
            buy, sell = transaction_rates(c["config"])
            request = {
                "symbols": symbols,
                "mu": mu.tolist(),
                "weights": [*weights.tolist(), 1 - float(weights.sum())],
                "benchmark": [*benchmark.tolist(), 1 - stock_exposure],
                "industry_exposure": exposure.to_numpy().tolist(),
                "industry_limits": limits.tolist(),
                "risk_aversion": self.contract["risk_aversion"],
                "stock_exposure": stock_exposure,
                "upper_weights": upper.tolist(),
                "lower_weights": lower.tolist(),
                "turnover_limit": turnover,
                "equity": equity,
                "signal_date": signal,
                "buy_rate": buy,
                "sell_rate": sell,
                "include_cost": allocator == "cvx_cost",
            }
            continuous = run_native(
                request,
                covariance.reindex(index=symbols, columns=symbols, fill_value=0.0).to_numpy(),
                self.output / "native" / c["kind"] / day,
                ROOT,
                verified_cache_directory=(self.solver_cache / "native" / c["kind"] / day)
                if self.solver_cache
                else None,
            )
            if continuous["status"] != "CONTINUOUS_PLAN_VERIFIED_NOT_EXECUTED":
                return protected("NATIVE_CONTINUOUS_INFEASIBLE")
            targets = continuous["weights"]
        desired = {}
        for symbol, weight in targets.items():
            if weight <= 0 or symbol not in c["last"]:
                continue
            if symbol in nh and abs(weight - weights.get(symbol, 0.0)) < 1e-9:
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
            desired, c["snapshot"], c["opened"], c["buy"], c["sell"], c["config"]
        )
        if risk_bounds is not None:
            locked = c["snapshot"]["unsettled_stock_rights"]
            values = pd.Series(
                {
                    s: (feasible.get(s, 0) + locked.get(s, 0)) * c["opened"][s]
                    for s in set(feasible) | set(locked)
                },
                dtype=float,
            )
            post_equity = float(cash + values.sum() + c["snapshot"].get("receivable_cash", 0.0))
            actual = values.reindex(symbols, fill_value=0.0) / post_equity
            exposure, limits = risk_bounds
            if (exposure.T @ actual > limits + 1e-5).any() or actual.max() > self.contract[
                "max_stock_weight"
            ] + 1e-5:
                return protected("INTEGER_OPEN_RISK_POSTCHECK")
        save(
            "OPEN_PLAN_VERIFIED",
            risk_audit=audit,
            continuous=continuous,
            frozen_symbols=sorted(frozen),
            target_quantities=desired,
            executable_quantities=feasible,
            open_clips=clips,
            next_open_used_for_selection=False,
        )
        return feasible, None


def main(output, bps, smoke, solver_cache=None):
    if output.exists():
        raise FileExistsError("IMMUTABLE_N3_OUTPUT")
    contract_path = ROOT / "configs/ashare_mainline_n3.json"
    contract = json.loads(contract_path.read_bytes())
    output.mkdir(parents=True)
    (output / "contract.json").write_bytes(contract_path.read_bytes())
    source_paths = [
        Path(__file__),
        Path(engine.__file__),
        ROOT / "scripts/solve_ashare_mainline_cvx.py",
        ROOT / "src/moneymore/research/mainline_execution.py",
        ROOT / "src/moneymore/execution/paper.py",
        ROOT / "src/moneymore/research/replay_actions.py",
        ROOT / "configs/ashare_cvxportfolio_requirements.txt",
    ]
    for path in source_paths:
        dest = output / "sources" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    started = time.monotonic()
    runtime = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_utc": datetime.now(UTC).isoformat(),
        "bps": bps,
        "smoke": smoke,
        "verified_solver_cache": str(solver_cache) if solver_cache else None,
        "progress_file": "smoke-v2/progress.json" if smoke else f"bps{bps}/progress.json",
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
    }

    def status():
        runtime["updated_utc"] = datetime.now(UTC).isoformat()
        runtime["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    status()
    try:
        replay(output, contract, bps, smoke, solver_cache)
        runtime["status"] = "COMPLETE"
    except BaseException as error:
        runtime.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        status()


def replay(output, contract, bps, smoke, solver_cache=None):
    contract_path = ROOT / "configs/ashare_mainline_n3.json"
    loaded = engine.load_inputs()
    planner = Planner(output, contract, loaded, solver_cache=solver_cache)
    engine.load_inputs = lambda: loaded
    engine.KINDS = [
        case + "--" + allocator
        for case in contract["models"]
        for allocator in contract["allocators"]
    ]
    engine.OUT = output
    provenance = {
        "config_sha256": sha256(contract_path),
        "runner_sha256": sha256(Path(__file__)),
        "engine_sha256": sha256(Path(engine.__file__)),
        "n2_results_sha256": sha256(BASE / "n2-summary-v1/results.json"),
        "actual_orders_enabled": False,
        "isolated_dependencies": True,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    engine.run(
        bps,
        smoke=smoke,
        execution_planner=planner,
        config_overrides={
            "max_position_weight": contract["max_stock_weight"],
            "initial_cash": contract["initial_cash"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bps", type=int, choices=[10, 20], default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--solver-cache", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.output, args.bps, args.smoke, args.solver_cache)
