"""Formal daily constructor with actual PaperBroker incumbents and strict evidence gates."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from ..challenger_diagnostics import trading_dates, write_diagnostic
from ..config import BacktestConfig
from ..execution_replay import run_execution_replay
from .baseline_v2_continuous_replay import (
    EMPTY,
    MonthlyOpeningPlans,
    audit_fill_rules,
    audit_ledger,
    prepare_market,
)
from .baseline_v2_daily_readiness import classify_held_inputs, construct_daily_targets
from .baseline_v2_daily_scores import logical_hash
from .security_identity import SecurityIdentity


class MissingExecutionEvidence(ValueError):
    def __init__(self, symbols):
        self.symbols = sorted(symbols)
        super().__init__("DAILY_EXECUTION_EVIDENCE_MISSING")


def diagnostic_performance(equity, fills, initial_cash):
    """Descriptive completed-window statistics, never a promotion decision."""
    values = equity.equity.astype(float)
    previous = float(initial_cash)
    annual = {}
    for year, rows in equity.groupby(equity.trade_date.astype(str).str[:4], sort=True):
        annual[year] = float(rows.equity.iloc[-1] / previous - 1)
        previous = float(rows.equity.iloc[-1])
    peaks = values.cummax().clip(lower=initial_cash)
    return {
        "cumulative_return": float(values.iloc[-1] / initial_cash - 1),
        "maximum_drawdown": float((1 - values / peaks).max()),
        "period_returns_by_calendar_year": annual,
        "calendar_year_caveat": "First year starts Jan31, not Jan1.",
        "fees": float(sum(float(f["fee"]) for f in fills)),
        "gross_traded_notional": float(sum(float(f["price"]) * f["quantity"] for f in fills)),
        "mean_gross_exposure": float(equity.gross_exposure.mean()),
        "descriptive_only_not_model_release": True,
    }


class DailyOpeningPlans(MonthlyOpeningPlans):
    def __init__(
        self, scores, pool, prices, policy, config, protocol, calendar, terminal, market_symbols
    ):
        super().__init__(
            pd.DataFrame(columns=["date", "symbol", "target_weight"]), config, protocol, terminal
        )
        self.scores = {pd.Timestamp(d).strftime("%Y%m%d"): f for d, f in scores.groupby("date")}
        self.pool = {str(d): set(f.security_id) for d, f in pool.groupby("signal_date")}
        self.prices, self.policy, self.calendar = prices, policy, calendar
        self.market_symbols = set(market_symbols)
        self.events, self.input_hold = [], None

    def closing(self, date, snapshot):
        if date >= self.terminal_date:
            return EMPTY.copy()
        if date not in self.scores or date not in self.pool:
            raise ValueError("EXACT_DAILY_SIGNAL_OR_COMPLETE_POOL_MISSING")
        quantities = {p["symbol"]: int(p["quantity"]) for p in snapshot["positions"]}
        for symbol, quantity in snapshot.get("unsettled_stock_rights", {}).items():
            quantities[symbol] = quantities.get(symbol, 0) + int(quantity)
        scores = self.scores[date]
        classification = classify_held_inputs(
            quantities, scores.symbol, universe_members=self.pool[date]
        )
        event = {"signal_date": date, "actual_incumbents": quantities, **classification}
        if classification["input_faults"]:
            self.context = {}  # Never execute an older target after a fresh input fault.
            self.input_hold = {**event, "status": "BLOCKED_HELD_INPUT_HOLD_AND_ALERT"}
            self.events.append(self.input_hold)
            return EMPTY.copy()
        execution_date = self.calendar[self.calendar.index(date) + 1]
        selected, weights, _ranking = construct_daily_targets(
            scores,
            quantities,
            self.prices.reindex(columns=scores.symbol),
            date,
            execution_date,
            self.policy,
            universe_members=self.pool[date],
        )
        support = set(weights) | {s for s, q in quantities.items() if q > 0}
        missing = support - self.market_symbols
        event.update(
            status="TARGET_CONSTRUCTED",
            weights=weights,
            selected=selected,
            missing_execution_evidence=sorted(missing),
        )
        self.events.append(event)
        if missing:
            self.context = {}
            raise MissingExecutionEvidence(missing)
        if set(classification["universe_exits"]) & set(weights):
            raise ValueError("EXCLUDED_UNIVERSE_MEMBER_RESELECTED")
        self.context = {"signal_date": date, "weights": weights, "due": True}
        self.input_hold = None
        return EMPTY.copy()

    def opening(self, date, snapshot, quotes):
        if self.input_hold:
            return {**self.input_hold, "orders": []}
        plan = super().opening(date, snapshot, quotes)
        if "monthly_target_weights" in plan:
            plan["daily_target_weights"] = plan.pop("monthly_target_weights")
        return plan


def frozen_prices(root, store, frozen, symbols):
    coverage = frozen["coverage_audit"]
    identity = SecurityIdentity(coverage["identity_policy"])
    source = set(symbols)
    for row in identity.changes:
        if row["old"] in source:
            source.add(row["new"])
    calendar = trading_dates(store, "20261028")
    dates = [d for d in calendar if coverage["identity_scope_start"] <= d <= "20251230"]
    filters = [
        ("ts_code", "in", sorted(source)),
        ("trade_date", ">=", dates[0]),
        ("trade_date", "<=", dates[-1]),
    ]
    values = {}
    for table, field, name in [
        ("daily", "close", "raw_close_matrix"),
        ("adj_factor", "adj_factor", "source_adjustment_matrix"),
    ]:
        frame = identity.market_rows(
            store.read(table, columns=["ts_code", "trade_date", field], filters=filters),
            "trade_date",
            [field],
        )
        matrix = frame.pivot(index="trade_date", columns="ts_code", values=field).reindex(
            index=dates, columns=sorted(symbols)
        )
        if logical_hash(matrix) != coverage["source_logical_hashes"][name]:
            raise ValueError("FROZEN_DAILY_PRICE_INPUT_CHANGED")
        values[field] = matrix
    prices = values["close"] * values["adj_factor"]
    prices.index = pd.to_datetime(prices.index)
    return prices


STRESS_SCENARIOS = {
    "platform_risk_hold": {"slippage_bps": 2},
    "platform_risk_reductions": {"slippage_bps": 2},
    "slippage_5bps": {"slippage_bps": 5},
    "slippage_10bps": {"slippage_bps": 10},
    "combined_10bps_double_commission_tax20": {
        "slippage_bps": 10,
        "commission_multiplier": 2,
        "cash_dividend_multiplier": 0.8,
    },
}


def run_daily_replay(root, store, end="20240208", *, scenario=None):
    if scenario is not None and scenario not in STRESS_SCENARIOS:
        raise ValueError("UNKNOWN_PREDECLARED_STRESS_SCENARIO")
    folder = root / "state/baseline-v2-validation"
    base = folder / "daily-inputs"
    score_report = json.loads((base / "daily-scores.json").read_text(encoding="utf-8"))
    score_run = Path(score_report["output_directory"])
    score_audit = json.loads((base / "daily-score-audit.json").read_text(encoding="utf-8"))
    if (
        score_audit["status"] != "SAVED_DAILY_RANKINGS_AUDITED_NOT_RELEASED"
        or Path(score_audit["score_run"]).resolve() != score_run.resolve()
    ):
        raise ValueError("CURRENT_SAVED_DAILY_SCORE_AUDIT_REQUIRED")
    score_path = score_run / "daily-scores.parquet"
    if hashlib.sha256(score_path.read_bytes()).hexdigest() != score_report["panel_sha256"]:
        raise ValueError("DAILY_SCORE_PANEL_HASH_MISMATCH")
    frozen = json.loads((score_run / "frozen-inputs.json").read_text(encoding="utf-8"))
    pool_path = (
        Path(frozen["identity_audit"]["output_directory"]) / "identity-top1000-preview.parquet"
    )
    if hashlib.sha256(pool_path.read_bytes()).hexdigest() != frozen["pool_sha256"]:
        raise ValueError("DAILY_POOL_HASH_MISMATCH")
    pool, panel = pd.read_parquet(pool_path), pd.read_parquet(score_path)
    calendar = trading_dates(store, "20261028")
    if end not in calendar or not "20240201" <= end <= "20251231":
        raise ValueError("INVALID_BOUNDED_REPLAY_END")
    if not pool.groupby("signal_date").security_id.nunique().eq(1000).all():
        raise ValueError("COMPLETE_UNIQUE_DAILY_TOP1000_REQUIRED")
    prices = frozen_prices(root, store, frozen, sorted(pool.security_id.unique()))
    evidence_report = json.loads((folder / "execution-preflight.json").read_text(encoding="utf-8"))
    if (
        evidence_report["data_status"] != "READY_FOR_ISOLATED_REPLAY_DATA"
        or evidence_report["data_gate_failures"]
    ):
        raise ValueError("VERIFIED_EXECUTION_MARKET_REQUIRED")
    evidence = Path(evidence_report["output_directory"])
    protocol_path = root / "configs/baseline_v2_replay.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("real_orders_enabled") is not False
        or protocol.get("automatic_promotion") is not False
    ):
        raise ValueError("RESEARCH_REPLAY_MUST_NOT_AUTHORIZE_REAL_ORDERS_OR_PROMOTION")
    protocol = {
        **protocol,
        "version": "formal_daily_constructor_evidence_bounded_v1_20261028",
        "end_date": end,
        "universe_exit_policy": "MANDATORY_EXIT_NOT_VOLUNTARY_REPLACEMENT_BUDGET",
        "pool_member_input_fault": "HOLD_ALL_AND_ALERT_CLEAR_OLD_TARGET",
        "signal_frequency": "EVERY_SESSION_ACTUAL_PAPER_HOLDINGS",
    }
    stress = STRESS_SCENARIOS.get(scenario, {})
    if scenario:
        protocol["version"] += ":" + scenario
        protocol["stress_scenario"] = scenario
        protocol["slippage_bps"] = stress["slippage_bps"]
        for field in ("commission_rate", "minimum_commission"):
            protocol[field] *= stress.get("commission_multiplier", 1)
    config = BacktestConfig.model_validate({k: protocol[k] for k in BacktestConfig.model_fields})
    actions = pd.read_parquet(evidence / "replay-actions.parquet")
    market = pd.read_parquet(evidence / "market-data-readiness.parquet")
    extensions = []
    for extra_report in sorted((base / "execution-extensions").glob("*/report.json")):
        extra = json.loads(extra_report.read_text(encoding="utf-8"))
        if (
            extra["status"] != "READY_FOR_BOUNDED_DAILY_REFERENCE_EXECUTION"
            or extra["score_panel_sha256"] != score_report["panel_sha256"]
            or extra["start"] > "20240131"
            or extra["end"] < end
        ):
            continue
        location = Path(extra["output_directory"])
        for name, digest in extra["artifact_hashes"].items():
            if hashlib.sha256((location / name).read_bytes()).hexdigest() != digest:
                raise ValueError("EXECUTION_EXTENSION_HASH_MISMATCH")
        symbols = set(extra["symbols"])
        existing = set(market.ts_code)
        if symbols <= existing:
            continue
        if symbols & existing:
            raise ValueError("OVERLAPPING_EXECUTION_EXTENSION_VERSIONS")
        market = pd.concat(
            [market, pd.read_parquet(location / "market-data-readiness-extra.parquet")],
            ignore_index=True,
        )
        actions = pd.concat(
            [actions, pd.read_parquet(location / "replay-actions-extra.parquet")], ignore_index=True
        )
        extensions.append(extra)
    if actions.action_key.duplicated().any():
        raise ValueError("DUPLICATE_EXTENDED_CORPORATE_ACTION")
    if scenario:
        actions = actions.copy()
        actions["cash_per_share"] *= stress.get("cash_dividend_multiplier", 1)
    # A delisted security has no post-delisting quotes. Do not carry a fake price.
    # The execution engine still hard-stops if any actual holding lacks a mark;
    # absorption conversion is NOT implicitly modelled by this lifecycle filter.
    post_delisting = market.get("after_verified_delisting", pd.Series(False, index=market.index))
    market = market.loc[~post_delisting.fillna(False).astype(bool)]
    bars = prepare_market(market, actions, config.slippage_bps)
    bars = bars.loc[bars.date.between(pd.Timestamp("20240131"), pd.Timestamp(end))]
    alias_symbols = {
        r[k] for r in frozen["coverage_audit"]["identity_policy"] for k in ("old", "new")
    }
    if set(bars.symbol) & alias_symbols:
        raise ValueError("EXECUTION_EVIDENCE_REQUIRES_IDENTITY_NORMALIZATION")
    directory = (
        base
        / "daily-replays"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    directory.mkdir(parents=True)
    write_diagnostic(
        directory / "frozen-protocol.json",
        {
            "protocol": protocol,
            "scoring": frozen,
            "saved_score_audit": score_audit,
            "execution_preflight": evidence_report,
            "execution_extensions": extensions,
            "input_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    score_path,
                    pool_path,
                    protocol_path,
                    evidence / "replay-actions.parquet",
                    evidence / "market-data-readiness.parquet",
                ]
            },
        },
    )
    bars.to_parquet(directory / "execution-bars.parquet", index=False)
    actions.to_parquet(directory / "execution-actions.parquet", index=False)
    expected_dates = [d for d in calendar if "20240131" <= d <= end]
    if sorted(bars.date.dt.strftime("%Y%m%d").unique()) != expected_dates:
        raise ValueError("EXECUTION_CALENDAR_WINDOW_INCOMPLETE")
    expected_days = len(expected_dates)
    results = []
    for variant in ("fixed", "learned"):
        scores = panel.rename(columns={"score_" + variant: "score"})
        database = directory / (variant + ".sqlite3")
        planner = DailyOpeningPlans
        risk_kwargs = {}
        if scenario in {"platform_risk_hold", "platform_risk_reductions"}:
            from .baseline_v2_platform_risk import (
                PlatformRiskDailyPlans,
                PlatformRiskReductionPlans,
            )

            planner = (
                PlatformRiskReductionPlans
                if scenario == "platform_risk_reductions"
                else PlatformRiskDailyPlans
            )
            risk_kwargs = {"risk_database": database, "account_id": variant}
        daily = planner(
            scores,
            pool,
            prices,
            frozen["policy"],
            config,
            protocol,
            calendar,
            end,
            set(bars.symbol),
            **risk_kwargs,
        )
        try:
            replay = run_execution_replay(
                bars=bars,
                targets=EMPTY,
                config=config,
                database=database,
                strategy_id=variant,
                account_id=variant,
                closing_target_provider=daily.closing,
                opening_plan_provider=daily.opening,
                corporate_actions=actions,
                accrue_action_receivables=True,
            )
        except MissingExecutionEvidence as exc:
            results.append(
                {
                    "variant": variant,
                    "status": "BLOCKED_PARTIAL_NO_PERFORMANCE_CONCLUSION",
                    "missing_execution_symbols": exc.symbols,
                    "blocked_signal_date": daily.events[-1]["signal_date"],
                }
            )
        else:
            if replay.equity.trade_date.astype(str).tolist() != expected_dates:
                raise ValueError("REPLAY_LEDGER_CALENDAR_MISMATCH")
            replay.equity.to_parquet(directory / (variant + "-equity.parquet"), index=False)
            pd.DataFrame(replay.fills).to_parquet(
                directory / (variant + "-fills.parquet"), index=False
            )
            proof, _, _ = audit_ledger(database, replay.equity, bars, actions, config.initial_cash)
            proof.to_parquet(directory / (variant + "-ledger-audit.parquet"), index=False)
            opening = replay.summary["opening_plan_audits"]
            holds = sum(e["status"] == "BLOCKED_HELD_INPUT_HOLD_AND_ALERT" for e in daily.events)
            blocked = sum(e["plan"]["status"] == "BLOCKED" for e in opening)
            results.append(
                {
                    "variant": variant,
                    "status": "BLOCKED_PLANS_OR_DATA_HOLDS"
                    if holds or blocked
                    else "BOUNDED_REPLAY_AUDITED_NOT_RELEASED",
                    "days": len(replay.equity),
                    "expected_days": expected_days,
                    "requested_window_complete": len(replay.equity) == expected_days,
                    "fills": len(replay.fills),
                    "input_hold_days": holds,
                    "blocked_solver_plans": blocked,
                    "platform_risk_blocked_days": sum(
                        e["plan"]["status"] == "ACCOUNT_RISK_BLOCKED" for e in opening
                    ),
                    "forced_universe_exit_events": sum(
                        bool(e["universe_exits"]) for e in daily.events
                    ),
                    "ledger_max_error": float(proof.equity_error.abs().max()),
                    "fill_rules": audit_fill_rules(replay.fills, bars, config),
                    "reconciliation": replay.reconciliation,
                    "performance": diagnostic_performance(
                        replay.equity, replay.fills, config.initial_cash
                    ),
                }
            )
            write_diagnostic(directory / (variant + "-opening-audits.json"), {"plans": opening})
        write_diagnostic(
            directory / (variant + "-closing-decisions.json"), {"events": daily.events}
        )
    window_complete = all(r.get("requested_window_complete", False) for r in results)
    full_window = window_complete and end == "20251231"
    result = {
        "status": (
            "FULL_LEDGER_WITH_BLOCKED_PLANS_OR_DATA_HOLDS"
            if full_window
            else "BLOCKED_PARTIAL_NOT_FULL_DAILY_VALIDATION"
        )
        if any(r["status"].startswith("BLOCKED") for r in results)
        else "TWO_YEAR_DAILY_REPLAY_AUDITED_NOT_RELEASED"
        if full_window
        else "BOUNDED_DAILY_REPLAY_AUDITED_NOT_FULL_VALIDATION",
        "start": "20240131",
        "end": end,
        "output_directory": str(directory),
        "runs": results,
        "requested_window_complete": window_complete,
        "two_year_window_complete": full_window,
        "release_authorized": False,
        "real_orders_enabled": False,
        "production_changed": False,
        "stress_scenario": scenario,
        "limitations": [
            "HISTORICAL_RESEARCH_NOT_RELEASE_VALIDATION",
            "REFERENCE_OPEN_NOT_REAL_EXECUTION_PRINTS",
            "NATIVE_STICKY_RISK_NO_AUTOMATIC_RECOVERY"
            if scenario in {"platform_risk_hold", "platform_risk_reductions"}
            else "NO_EXTRA_MARKET_TIMING_OR_DRAWDOWN_STOP",
            "DIVIDEND_IMMEDIATE_20_PERCENT_WITHHOLDING_STRESS_NOT_ACTUAL_TAX"
            if stress.get("cash_dividend_multiplier") == 0.8
            else "GROSS_DIVIDEND_PRIMARY_NOT_HOLDING_PERIOD_TAX",
        ],
    }
    write_diagnostic(directory / "report.json", result)
    if scenario:
        write_diagnostic(base / "stress-replays" / scenario / "latest.json", result)
    else:
        write_diagnostic(base / "daily-replay.json", result)
    return result
