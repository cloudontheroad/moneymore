"""Six isolated paper books, 36 months; preserve rank-cap execution rules."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import replay_ashare_rank_preserving as old
import yaml
from prepare_ashare_three_year import BASE, CONFIG, ROOT
from prepare_four_strategy_2026 import save
from threadpoolctl import threadpool_limits

from moneymore.research.ashare_mainline import sha256
from moneymore.research.mainline_execution import physical_open_plan, risk_window_with_halt_seed
from moneymore.research.three_year import causal_volatility_scale
from moneymore.research.three_year_actions import AppendOnlyActionCache, reviewed_actions

PREDICTION_CASES = ["rank_cap", "financial_rank_cap", "rnn_rank_cap", "tra_rank_cap"]


def inputs(smoke=False):
    cfg = json.loads(CONFIG.read_bytes())
    data = BASE / "data-v1"
    panel_manifest = json.loads((BASE / "panel-v1/results.json").read_bytes())
    core_manifest = json.loads((BASE / "panel-v1/core-results.json").read_bytes())
    if (
        core_manifest["panel_sha256"] != sha256(BASE / "panel-v1/core-panel.parquet")
        or panel_manifest["core_panel_sha256"] != core_manifest["panel_sha256"]
    ):
        raise ValueError("FROZEN_COMMON_CORE_CHANGED")
    if panel_manifest["panel_sha256"] != sha256(BASE / "panel-v1/panel.parquet"):
        raise ValueError("PANEL_CHANGED_AFTER_TRAINING")
    panel = pd.read_parquet(
        BASE / "panel-v1/panel.parquet", columns=["date", "symbol", "industry", "total_mv"]
    )
    calendar = json.loads((data / "calendar.json").read_bytes())["days"]
    sessions = [d for d in calendar if cfg["evaluation_start"] <= d <= cfg["evaluation_end"]]
    if smoke:
        sessions = sessions[:3]
    signals = {d: calendar[calendar.index(d) - 1] for d in sessions}
    folds = sorted({str(pd.Timestamp(d).to_period("Q")) for d in signals.values()})
    predictions = []
    prediction_hashes = {}
    for fold in folds:
        for case in PREDICTION_CASES:
            # Execution smoke uses the actual frozen models; only the replay calendar is shorter.
            folder = BASE / "training-v1" / fold / case
            result = json.loads((folder / "results.json").read_bytes())
            expected_panel = panel_manifest if case == "financial_rank_cap" else core_manifest
            if result["panel_sha256"] != expected_panel["panel_sha256"] or result[
                "predictions_sha256"
            ] != sha256(folder / "predictions.parquet"):
                raise ValueError("TRAINING_INPUT_OR_PREDICTION_CHANGED")
            f = pd.read_parquet(folder / "predictions.parquet")
            if f.duplicated(["date", "symbol"]).any():
                raise ValueError("DUPLICATE_PREDICTIONS")
            predictions.append(f)
            prediction_hashes[f"{fold}:{case}"] = sha256(folder / "predictions.parquet")
    predictions = pd.concat(predictions, ignore_index=True)
    bars = pd.read_parquet(data / "bars.parquet")
    repair_dir = BASE / "execution-repair-v1"
    repair = json.loads((repair_dir / "results.json").read_bytes())
    if repair["base_sha256"] != sha256(data / "bars.parquet") or repair["extra_sha256"] != sha256(
        repair_dir / "extra-bars.parquet"
    ):
        raise ValueError("EXECUTION_REPAIR_INPUT_CHANGED")
    extra = pd.read_parquet(repair_dir / "extra-bars.parquet")
    if (
        pd.MultiIndex.from_frame(extra[["trade_date", "ts_code"]])
        .isin(pd.MultiIndex.from_frame(bars[["trade_date", "ts_code"]]))
        .any()
    ):
        raise ValueError("EXECUTION_REPAIR_CANNOT_OVERWRITE_EXISTING_QUOTES")
    bars = pd.concat([bars, extra], ignore_index=True)
    members = pd.read_parquet(data / "membership.parquet")
    membership = {d: set(g.symbol) for d, g in members.groupby("date")}
    context = pd.read_parquet(BASE / "panel-v1/market-context.parquet")
    scale, c = causal_volatility_scale(
        context.market_return1,
        list(signals.values()),
        calendar,
        cfg["volatility"]["calibration_start"],
        cfg["volatility"]["calibration_end"],
    )
    baseline = {
        d: g.set_index("symbol").score
        for d, g in predictions.loc[predictions.case.eq("rank_cap")].groupby("date")
    }
    needed = sorted(set(sessions) | set(signals.values()))
    official = {}
    retail_rules = yaml.safe_load((ROOT / "configs/baseline_v2_action_classes.yaml").read_text())[
        "classes"
    ]
    aggregate_rules = json.loads(
        (ROOT / "configs/ashare_three_year_action_evidence.json").read_bytes()
    )["cash_aggregate_overlap"]
    execution_meta = json.loads((data / "execution-results.json").read_bytes())
    hashes = {r["file"]: r["sha256"] for r in execution_meta["files"]}
    for day in needed:
        official[day] = {}
        for name in ["daily_basic", "stk_limit", "suspend_d", "dividend"]:
            path = data / "official" / f"{name}-{day}.parquet"
            if name != "daily_basic" and sha256(path) != hashes[path.name]:
                raise ValueError("OFFICIAL_EXECUTION_INPUT_CHANGED")
            official[day][name] = pd.read_parquet(path)
            if name == "dividend":
                official[day][name], _ = reviewed_actions(
                    official[day][name], retail_rules, aggregate_rules
                )
    instrument = (
        pd.read_parquet(data / "instruments.parquet")
        .drop_duplicates("ts_code")
        .set_index("ts_code")
    )
    loaded = (bars, membership, sessions, signals, scale, c, baseline, official, instrument)
    return cfg, loaded, panel, predictions, calendar, prediction_hashes


class ThreeYearPlanner(old.RankPlanner):
    def __init__(self, output, cfg, loaded, panel, predictions, calendar):
        self.output = output
        self.contract = {**cfg, "models": PREDICTION_CASES}
        self.scores = {
            (case, d): g.set_index("symbol").score
            for (case, d), g in predictions.groupby(["case", "date"])
        }
        for day in sorted(predictions.date.unique()):
            for case in ["rank_cap_vol", "rank_cap_static"]:
                self.scores[(case, day)] = self.scores[("rank_cap", day)]
        self.attributes = {d: g.set_index("symbol") for d, g in panel.groupby("date")}
        self.industry_history = {
            s: g.loc[g.industry.notna(), ["date", "industry"]].sort_values("date")
            for s, g in panel.groupby("symbol")
        }
        _, self.membership, sessions, signals, scale, _, _, _, _ = loaded
        # Eligibility and all model inputs stay frozen. Repairs affect fills/marks only.
        frozen_risk_bars = pd.read_parquet(BASE / "data-v1/bars.parquet")
        self.prices = (
            frozen_risk_bars.assign(p=frozen_risk_bars.close * frozen_risk_bars.adj_factor)
            .pivot(index="trade_date", columns="ts_code", values="p")
            .reindex(calendar)
        )
        self.calendar = calendar
        self.proofs = pd.concat(
            [pd.read_parquet(p) for p in (BASE / "data-v1/official").glob("suspend_d-*.parquet")],
            ignore_index=True,
        ).drop_duplicates()
        self.scale = scale
        scheduled = {
            d: scale[
                signals[sessions[(i // cfg["rebalance_sessions"]) * cfg["rebalance_sessions"]]]
            ]
            for i, d in enumerate(sessions)
        }
        self.static_scale = float(np.mean(list(scheduled.values())))
        self.scope = json.loads((ROOT / "configs/ashare_n4_execution_scope_v1.json").read_bytes())[
            "events"
        ]
        self.cache = {}

    def risk(self, signal):
        if signal in self.cache:
            return self.cache[signal]
        dates = [d for d in self.calendar if d <= signal][-(self.contract["risk_sessions"] + 1) :]
        if len(dates) != self.contract["risk_sessions"] + 1:
            raise ValueError("INCOMPLETE_RISK_HISTORY")
        known = set().union(*(m for d, m in self.membership.items() if d <= signal))
        history, carries = risk_window_with_halt_seed(
            self.prices.reindex(columns=sorted(known)), self.proofs, dates
        )
        returns = history.pct_change(fill_method=None).iloc[1:]
        risk_eligible = returns.columns[np.isfinite(returns).all()].tolist()
        attrs = self.attributes[signal]
        excluded = {e["symbol"] for e in self.scope if e["available_from"] <= signal}
        eligible = [
            s
            for s in risk_eligible
            if s in self.membership[signal]
            and s in attrs.index
            and s not in excluded
            and pd.notna(attrs.loc[s, "industry"])
            and np.isfinite(attrs.loc[s, "total_mv"])
            and attrs.loc[s, "total_mv"] > 0
            and all(s in self.scores[(case, signal)].index for case in PREDICTION_CASES)
        ]
        if len(eligible) < 100:
            raise ValueError("INSUFFICIENT_COMMON_ELIGIBILITY")
        # The ordinal allocator uses covariance.index only. Preserve identical
        # finite-history eligibility without computing unused dense covariance.
        risk_index = pd.DataFrame(index=risk_eligible)
        audit = {
            "signal_date": signal,
            "risk_start": dates[0],
            "risk_end": dates[-1],
            "eligible": len(eligible),
            "risk_eligible": len(risk_eligible),
            "members": len(self.membership[signal]),
            "official_halt_carries": carries,
            "known_unsupported_mergers": sorted(excluded),
            "covariance_unused_by_ordinal_rank_allocator": True,
        }
        self.cache.clear()
        self.cache[signal] = (risk_index, attrs.loc[eligible], audit)
        return self.cache[signal]

    def __call__(self, **context):
        case = context["kind"].split("--")[0]
        excluded = {e["symbol"] for e in self.scope if e["available_from"] <= context["signal"]}
        context = {**context, "members": context["members"] - excluded}
        result, reason = super().__call__(**context)
        path = self.output / "decisions" / context["kind"] / (context["day"] + ".json")
        decision = json.loads(path.read_bytes())
        decision["known_unsupported_merger_scope"] = sorted(excluded)
        decision["overlay_scale"] = 1.0
        if (
            case in ["rank_cap_vol", "rank_cap_static"]
            and decision["status"] == "OPEN_PLAN_VERIFIED"
        ):
            scale = (
                self.static_scale if case == "rank_cap_static" else self.scale[context["signal"]]
            )
            frozen = set(decision["frozen_symbols"])
            rights = context["snapshot"]["unsettled_stock_rights"]
            desired = {}
            for symbol, q in decision["target_quantities"].items():
                desired[symbol] = (
                    q
                    if symbol in frozen
                    else max(
                        0,
                        int(((q + rights.get(symbol, 0)) * scale - rights.get(symbol, 0)) / 100)
                        * 100,
                    )
                )
            result, cash, clips = physical_open_plan(
                desired,
                context["snapshot"],
                context["opened"],
                context["buy"],
                context["sell"],
                context["config"],
            )
            decision["unscaled_target_quantities"] = decision["target_quantities"]
            decision["unscaled_open_postcheck"] = decision.pop("open_postcheck")
            decision["target_quantities"] = desired
            decision["executable_quantities"] = result
            decision["overlay_scale"] = scale
            decision["overlay_clips"] = clips
            decision["risk_limits_basis"] = "unchanged_full_exposure_rank_selection_limits"
            decision["static_control_ex_post"] = case == "rank_cap_static"
            values = pd.Series(
                {
                    s: (result.get(s, 0) + rights.get(s, 0)) * context["opened"][s]
                    for s in set(result) | set(rights)
                },
                dtype=float,
            )
            equity = cash + values.sum() + context["snapshot"].get("receivable_cash", 0)
            weights = values / equity
            industry = {}
            for symbol in values.index:
                history = self.industry_history[symbol]
                industry[symbol] = history.loc[history.date.le(context["signal"])].iloc[-1].industry
            actual = weights.groupby(pd.Series(industry)).sum()
            limits = pd.Series(decision["unscaled_open_postcheck"]["industry_limits"])
            violation = float((actual - limits.reindex(actual.index)).max()) if len(actual) else 0.0
            maximum = float(weights.max()) if len(weights) else 0.0
            passed = bool(
                cash >= -0.01
                and np.isfinite(violation)
                and violation <= 1e-5
                and maximum <= self.contract["max_stock_weight"] + 1e-5
            )
            decision["open_postcheck"] = {
                "passed": passed,
                "actual_stock_exposure": float(weights.sum()),
                "maximum_stock_weight": maximum,
                "maximum_industry_violation": violation,
                "industry_weights": actual.to_dict(),
                "industry_limits": limits.to_dict(),
                "cash": cash,
                "stock_limit": self.contract["max_stock_weight"],
            }
            if not passed:
                save(path, {**decision, "status": "OVERLAY_POSTCHECK_FAILED"})
                raise ValueError("OVERLAY_INTEGER_OPEN_RISK_POSTCHECK:" + context["day"])
        save(path, decision)
        return result, reason


def main(bps, smoke=False):
    cfg, loaded, panel, predictions, calendar, prediction_hashes = inputs(smoke)
    output = BASE / ("replay-smoke" if smoke else "replay-v1") / f"cost{bps}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        print("ALREADY_COMPLETE", flush=True)
        return
    planner = ThreeYearPlanner(output, cfg, loaded, panel, predictions, calendar)
    sources = list(
        dict.fromkeys(
            old.source_files()
            + [
                Path(__file__),
                CONFIG,
                ROOT / "src/moneymore/research/three_year.py",
                ROOT / "configs/ashare_n4_execution_scope_v1.json",
                ROOT / "configs/ashare_three_year_action_evidence.json",
                ROOT / "configs/baseline_v2_action_classes.yaml",
                ROOT / "src/moneymore/research/three_year_actions.py",
                ROOT / "src/moneymore/research/reviewed_action_classes.py",
                ROOT / "scripts/repair_ashare_three_year_execution_quotes.py",
            ]
        )
    )
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    for path in sources:
        dest = output / "sources" / path.relative_to(ROOT)
        if dest.exists() and sha256(dest) != sha256(path):
            raise ValueError("RUN_SOURCE_CHANGED_ON_RESUME")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
    save(
        output / "runtime.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "source_hashes": hashes,
            "prediction_hashes": prediction_hashes,
            "execution_repair_manifest_sha256": sha256(BASE / "execution-repair-v1/results.json"),
            "static_target_scale_ex_post": planner.static_scale,
            "config": cfg,
        },
    )
    old.engine.load_inputs = lambda: loaded
    old.engine.KINDS = [c + "--rank_cap" for c in cfg["cases"]]
    old.engine.OUT = output

    def missing_quotes(root, bars, symbols, day):
        raise ValueError(
            "MISSING_HELD_PRICE_REQUIRES_CORPORATE_ACTION_OR_DATA_AUDIT:"
            + str(day)
            + ":"
            + ",".join(sorted(symbols))
        )

    old.engine.ensure_quotes = missing_quotes
    old.engine.official_replay_actions = AppendOnlyActionCache()
    old.engine.run(
        bps,
        smoke=smoke,
        execution_planner=planner,
        config_overrides={
            "initial_cash": cfg["initial_cash"],
            "max_position_weight": cfg["max_stock_weight"],
            "stamp_duty_rate": 0.0005,
            "transfer_fee_rate": 0.00001,
        },
    )
    save(
        output / "completed.json",
        {
            "status": "COMPLETE",
            "smoke": smoke,
            "bps": bps,
            "source_hashes": hashes,
            "prediction_hashes": prediction_hashes,
            "execution_repair_manifest_sha256": sha256(BASE / "execution-repair-v1/results.json"),
            "production_modified": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bps", type=int, choices=[10, 20], required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    with threadpool_limits(limits=2):
        main(a.bps, a.smoke)
