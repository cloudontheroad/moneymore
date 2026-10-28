"""Continuous paired month-end signal experiment using the existing paper engine.

Not the original live baseline, not production daily V2, not new unseen OOS.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from ..challenger_diagnostics import write_diagnostic
from ..config import BacktestConfig
from ..costs import execution_price, transaction_fee
from ..execution_replay import run_execution_replay
from ..models import Side
from .integer_execution import plan_integer_execution
from .replay_attribution import reconstruct_cash, reconstruct_positions

EMPTY = pd.DataFrame(columns=["date", "symbol", "target_weight"])


def audit_fill_rules(fills, bars, config):
    quotes = bars.set_index(["date", "symbol"])
    for fill in fills:
        row = quotes.loc[(pd.Timestamp(fill["trade_date"]), fill["symbol"])]
        side = Side(fill["side"])
        if not row["can_buy" if side == Side.BUY else "can_sell"]:
            raise ValueError("FILL_ON_HALTED_OR_PRICE_LIMIT_BLOCKED_BAR")
        price = execution_price(float(row.open), side, config.slippage_bps)
        fee = transaction_fee(price * fill["quantity"], side, config)
        if not np.isclose(price, fill["price"], rtol=0, atol=1e-8) or not np.isclose(
            fee, fill["fee"], rtol=0, atol=1e-6
        ):
            raise ValueError("FILL_PRICE_OR_FEE_DOES_NOT_MATCH_FROZEN_PROTOCOL")
        if side == Side.BUY:
            minimum = 200 if fill["symbol"].startswith(("688", "689")) else config.lot_size
            if fill["quantity"] < minimum or fill["quantity"] % config.lot_size:
                raise ValueError("BUY_LOT_RULE_VIOLATION")
    return {"status": "PASS", "fills_checked": len(fills)}


def prepare_market(readiness, actions, slippage_bps):
    f = readiness.copy().sort_values(["ts_code", "trade_date"])
    f = f.loc[~f.not_yet_listed].copy()
    missing = ~f.price_ok
    if (missing & ~f.verified_full_session_suspension).any():
        raise ValueError("UNVERIFIED_MISSING_PRICE")
    halted = set(zip(f.loc[missing, "ts_code"], f.loc[missing, "trade_date"]))
    for event in actions.itertuples():
        if (event.symbol, event.ex_date.strftime("%Y%m%d")) in halted:
            raise ValueError("SUSPENDED_EX_DATE_REQUIRES_SEPARATE_ECONOMIC_MARK_POLICY")
    # Valuation-only stale marks are allowed ONLY with official whole-session proof.
    prior = f.groupby("ts_code").close.transform(
        lambda x: x.where(np.isfinite(x) & x.gt(0)).ffill()
    )
    f.loc[missing, "open"] = prior.loc[missing]
    f.loc[missing, "close"] = prior.loc[missing]
    if not (
        np.isfinite(f[["open", "close"]]).all().all() and f.open.gt(0).all() and f.close.gt(0).all()
    ):
        raise ValueError("NO_PRIOR_VALUATION_FOR_SUSPENSION")
    active = ~f.verified_full_session_suspension
    if (active & ~f.limits_ok).any():
        raise ValueError("UNVERIFIED_ACTIVE_PRICE_LIMIT")
    f["can_buy"] = active & (f.open * (1 + slippage_bps / 10000)).lt(f.up_limit - 1e-8)
    f["can_sell"] = active & (f.open * (1 - slippage_bps / 10000)).gt(f.down_limit + 1e-8)
    f["valuation_only_suspension_carry"] = missing
    return f.rename(columns={"trade_date": "date", "ts_code": "symbol"}).assign(
        date=lambda x: pd.to_datetime(x.date)
    )


class MonthlyOpeningPlans:
    """Frozen monthly signal; next-open integer plan; retry market-blocked changes only."""

    def __init__(self, targets, config, protocol, terminal_date):
        self.targets = {
            pd.Timestamp(date).strftime("%Y%m%d"): dict(zip(g.symbol, g.target_weight))
            for date, g in targets.groupby("date")
        }
        self.config, self.protocol, self.terminal_date = config, protocol, terminal_date
        self.context = {}

    def closing(self, date, snapshot):
        if date in self.targets and date < self.terminal_date:
            self.context = {"signal_date": date, "weights": self.targets[date], "due": True}
        return EMPTY.copy()

    def opening(self, date, snapshot, quotes):
        if not self.context or not self.context["due"]:
            return {"status": "NO_REBALANCE_TODAY", "orders": []}
        signal = self.context["signal_date"]
        if date <= signal:
            raise ValueError("SIGNAL_NOT_PRIOR_TO_EXECUTION")
        queued = self.context.get("market_wait", {})
        if queued and not any(
            bool(quotes.loc[s, "can_buy" if side == "BUY" else "can_sell"])
            for s, side in queued.items()
        ):
            return {
                "status": "WAITING_FOR_PREVIOUSLY_BLOCKED_MARKET",
                "orders": [],
                "signal_date": signal,
                "market_wait": queued,
            }
        quantities = {p["symbol"]: int(p["quantity"]) for p in snapshot["positions"]}
        locked = snapshot.get("unsettled_stock_rights", {})
        support = sorted(set(self.context["weights"]) | set(quantities) | set(locked))
        # Zero columns impose NO style choice. This reuses the integer cost planner,
        # not a new style rule. The experiment compares weights under identical rules.
        styles = pd.DataFrame({"execution_only_zero": 0.0}, index=support)
        plan = plan_integer_execution(
            self.context["weights"],
            quotes.open,
            quantities,
            quantities,
            float(snapshot["cash"]),
            float(snapshot["equity"]),
            self.config,
            styles,
            {"execution_only_zero": 1.0},
            locked_shares=locked,
            turnover=self.protocol["turnover_l1"],
            can_buy=quotes.can_buy.to_dict(),
            can_sell=quotes.can_sell.to_dict(),
            time_limit=self.protocol["solver_time_limit"],
            solver_threads=self.protocol["solver_threads"],
        )
        if plan["status"] == "PASS_REFERENCE_QUOTES_ONLY":
            market_wait = {}
            for s in support:
                desired = (
                    int(
                        float(snapshot["equity"])
                        * self.context["weights"].get(s, 0)
                        / float(quotes.loc[s, "open"])
                        / self.config.lot_size
                    )
                    * self.config.lot_size
                )
                held = quantities.get(s, 0) + locked.get(s, 0)
                if desired > held and not quotes.loc[s, "can_buy"]:
                    market_wait[s] = "BUY"
                elif desired < held and not quotes.loc[s, "can_sell"]:
                    market_wait[s] = "SELL"
            self.context["market_wait"] = market_wait
            self.context["due"] = bool(market_wait)
        return {
            **plan,
            "signal_date": signal,
            "monthly_target_weights": self.context["weights"],
            "retry_market_blocked": self.context["due"],
        }


def audit_ledger(database, equity, bars, actions, initial_cash):
    """Rebuild EVERY day's cash, positions and economic NAV without final-state flags."""
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        fills = pd.read_sql_query(
            "SELECT f.*, o.signal_date FROM fills f JOIN orders o "
            "ON f.idempotency_key = o.idempotency_key",
            conn,
        )
        ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", conn)
        rights = pd.read_sql_query("SELECT * FROM corporate_entitlements", conn)
    dates = equity.trade_date.astype(str).tolist()
    holdings = reconstruct_positions(fills, ledger, dates)
    cash = reconstruct_cash(fills, ledger, dates, initial_cash)
    marks = bars.pivot(index="date", columns="symbol", values="close")
    by_action = actions.set_index("action_key")
    rows = []
    for day in equity.itertuples():
        date = pd.Timestamp(day.trade_date)
        values = sum(q * marks.loc[date, s] for s, q in holdings[day.trade_date].items())
        cash_receivable, stock_receivable = 0.0, 0.0
        for entitlement in rights.itertuples():
            event = by_action.loc[entitlement.action_key]
            if not event.ex_date <= date:
                continue
            if entitlement.cash_per_share > 0 and date < pd.Timestamp(entitlement.cash_pay_date):
                cash_receivable += entitlement.entitled_quantity * entitlement.cash_per_share
            if entitlement.stock_ratio > 0 and date < pd.Timestamp(entitlement.stock_list_date):
                stock_receivable += (
                    int(entitlement.entitled_quantity * entitlement.stock_ratio)
                    * marks.loc[date, entitlement.symbol]
                )
        expected = cash[day.trade_date] + values + cash_receivable + stock_receivable
        rows.append(
            {
                "trade_date": day.trade_date,
                "cash_error": cash[day.trade_date] - day.cash,
                "equity_error": expected - day.equity,
            }
        )
    proof = pd.DataFrame(rows)
    if proof[["cash_error", "equity_error"]].abs().max().max() > 0.01:
        raise ValueError("INDEPENDENT_DAILY_LEDGER_RECONSTRUCTION_FAILED")
    if not all(fills.signal_date.astype(str) < fills.trade_date.astype(str)):
        raise ValueError("SAME_DAY_OR_FUTURE_SIGNAL_FILL")
    return proof, ledger, rights


def run(root):
    folder = root / "state/baseline-v2-validation"
    source = folder / "execution-preflight.json"
    readiness = json.loads(source.read_text(encoding="utf-8"))
    if (
        readiness["data_status"] != "READY_FOR_ISOLATED_REPLAY_DATA"
        or readiness["data_gate_failures"]
    ):
        raise ValueError("EXECUTION_DATA_GATE_NOT_PASSED")
    evidence = Path(readiness["output_directory"])
    config_path = root / "configs/baseline_v2_replay.yaml"
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        readiness["start_date"] != protocol["start_date"]
        or readiness["end_date"] != protocol["end_date"]
    ):
        raise ValueError("REPLAY_PROTOCOL_DATE_SCOPE_CHANGED")
    config = BacktestConfig.model_validate({k: protocol[k] for k in BacktestConfig.model_fields})
    output = (
        folder
        / "continuous-replays"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    output.mkdir(parents=True)
    write_diagnostic(output / "frozen-protocol.json", protocol)
    targets = pd.read_parquet(evidence / "paired-monthly-targets.parquet")
    original_actions = pd.read_parquet(evidence / "replay-actions.parquet")
    bars = prepare_market(
        pd.read_parquet(evidence / "market-data-readiness.parquet"),
        original_actions,
        config.slippage_bps,
    )
    bars.to_parquet(output / "execution-bars.parquet", index=False)
    result = {
        "status": "RUNNING",
        "protocol": protocol,
        "config": config.model_dump(),
        "source_preflight_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_weight_run": readiness["source_weight_run"],
        "output_directory": str(output),
        "scope": "MONTHLY_FROZEN_WEIGHT_PAIRED_ABLATION_NOT_PRODUCTION_DAILY_V2_NOT_NEW_UNSEEN_OOS",
        "real_orders_enabled": False,
        "production_policy_updated": False,
        "automatic_promotion": False,
        "dividend_tax_caveat": "Gross primary; uniform20pct immediate withholding is conservative sensitivity, NOT actual holding-period tax.",
        "execution_caveat": "Known-open reference with modelled slippage, no claim of actual executable prints or depth.",
        "max_drawdown_note": "Measured only: no extra portfolio timing/stop rule introduced by this weight-only ablation.",
        "runs": [],
    }
    inputs = [
        source,
        config_path,
        evidence / "paired-monthly-targets.parquet",
        evidence / "replay-actions.parquet",
        evidence / "market-data-readiness.parquet",
    ]
    result["input_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    for tax_name, cash_multiplier in protocol["dividend_scenarios"].items():
        actions = original_actions.copy()
        actions["cash_per_share"] *= cash_multiplier
        for variant in ("fixed", "learned"):
            name = variant + "-" + tax_name
            monthly = MonthlyOpeningPlans(
                targets.loc[targets.variant.eq(variant)], config, protocol, readiness["end_date"]
            )
            print(json.dumps({"status": "REPLAY_START", "variant": name}), flush=True)
            replay = run_execution_replay(
                bars=bars,
                targets=EMPTY,
                config=config,
                database=output / (name + ".sqlite3"),
                strategy_id=name,
                account_id=name,
                closing_target_provider=monthly.closing,
                opening_plan_provider=monthly.opening,
                corporate_actions=actions,
                accrue_action_receivables=True,
            )
            replay.equity.to_parquet(output / (name + "-equity.parquet"), index=False)
            pd.DataFrame(replay.fills).to_parquet(output / (name + "-fills.parquet"), index=False)
            audit, ledger, rights = audit_ledger(
                output / (name + ".sqlite3"), replay.equity, bars, actions, config.initial_cash
            )
            fill_rules = audit_fill_rules(replay.fills, bars, config)
            audit.to_parquet(output / (name + "-daily-ledger-audit.parquet"), index=False)
            ledger.to_parquet(output / (name + "-corporate-ledger.parquet"), index=False)
            rights.to_parquet(output / (name + "-entitlements.parquet"), index=False)
            summary = {k: v for k, v in replay.summary.items() if k != "opening_plan_audits"}
            summary.update(
                fill_rules=fill_rules,
                final_equity=float(replay.equity.equity.iloc[-1]),
                cash_dividends_paid=float(ledger.cash_amount.sum()),
                daily_ledger_max_error=float(audit.equity_error.abs().max()),
                blocked_opening_plans=sum(
                    x["plan"]["status"] == "BLOCKED" for x in replay.summary["opening_plan_audits"]
                ),
                primary_optimal_fallbacks=sum(
                    x["plan"].get("primary_optimal_fallback", False)
                    for x in replay.summary["opening_plan_audits"]
                ),
                market_wait_days=sum(
                    x["plan"]["status"] == "WAITING_FOR_PREVIOUSLY_BLOCKED_MARKET"
                    for x in replay.summary["opening_plan_audits"]
                ),
            )
            write_diagnostic(
                output / (name + "-execution-audits.json"),
                {"opening_plan_audits": replay.summary["opening_plan_audits"]},
            )
            result["runs"].append(
                {
                    "variant": variant,
                    "dividend_scenario": tax_name,
                    "summary": summary,
                    "reconciliation": replay.reconciliation,
                }
            )
            write_diagnostic(output / "report.json", result)
            print(
                json.dumps({"status": "REPLAY_DONE", "variant": name, "summary": summary}),
                flush=True,
            )
    result["pairs"] = []
    for scenario in protocol["dividend_scenarios"]:
        pair = {
            x["variant"]: x["summary"] for x in result["runs"] if x["dividend_scenario"] == scenario
        }
        result["pairs"].append(
            {
                "dividend_scenario": scenario,
                "learned_minus_fixed_total_return": pair["learned"]["total_return"]
                - pair["fixed"]["total_return"],
                "learned_minus_fixed_max_drawdown": pair["learned"]["max_drawdown"]
                - pair["fixed"]["max_drawdown"],
            }
        )
    for p in inputs:
        if hashlib.sha256(p.read_bytes()).hexdigest() != result["input_hashes"][str(p)]:
            raise ValueError("FROZEN_REPLAY_INPUT_CHANGED_DURING_RUN")
    result["status"] = (
        "BLOCKED_EXECUTION_SOLVER_FAILURES"
        if any(x["summary"]["blocked_opening_plans"] for x in result["runs"])
        else "COMPLETED_PAIRED_EXECUTION_DIAGNOSTIC_NOT_RELEASE"
    )
    write_diagnostic(output / "report.json", result)
    write_diagnostic(folder / "continuous-replay.json", result)
    return result


def audit_saved_replay(root):
    """Separate post-run audit preserves original experiments, including failed ones."""
    from collections import Counter

    folder = root / "state/baseline-v2-validation"
    source = folder / "continuous-replay.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    directory = Path(result["output_directory"])
    config = BacktestConfig.model_validate(result["config"])
    bars = pd.read_parquet(directory / "execution-bars.parquet")
    records = []
    blocked_count = 0
    for run_record in result["runs"]:
        name = run_record["variant"] + "-" + run_record["dividend_scenario"]
        fills = pd.read_parquet(directory / (name + "-fills.parquet")).to_dict("records")
        rules = audit_fill_rules(fills, bars, config)
        plans = json.loads(
            (directory / (name + "-execution-audits.json")).read_text(encoding="utf-8")
        )["opening_plan_audits"]
        reasons = Counter(
            p["plan"].get("reason") for p in plans if p["plan"]["status"] == "BLOCKED"
        )
        blocked_count += sum(reasons.values())
        equity = pd.read_parquet(directory / (name + "-equity.parquet"))
        previous = config.initial_cash
        annual = {}
        for year, rows in equity.groupby(equity.trade_date.astype(str).str[:4]):
            annual[year] = float(rows.equity.iloc[-1] / previous - 1)
            previous = float(rows.equity.iloc[-1])
        records.append(
            {
                "name": name,
                "fill_rule_audit": rules,
                "blocked_plan_reasons": dict(reasons),
                "period_returns_by_calendar_year": annual,
                "calendar_year_caveat": "First year starts Jan31, not Jan1.",
                "primary_optimal_fallbacks": sum(
                    p["plan"].get("primary_optimal_fallback", False) for p in plans
                ),
            }
        )
    audit = {
        "status": "BLOCKED_NUMERICAL_EXECUTION_FAILURES"
        if blocked_count
        else "AUDITED_PAIRED_REPLAY_ONLY",
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_directory": str(directory),
        "blocked_plans": blocked_count,
        "runs": records,
        "not_final_v2_release": True,
        "real_orders_enabled": False,
    }
    write_diagnostic(directory / "post-run-audit.json", audit)
    write_diagnostic(folder / "continuous-replay-audit.json", audit)
    return audit
