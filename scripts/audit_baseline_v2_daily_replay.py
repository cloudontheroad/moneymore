"""Independent saved daily replay audit. Research only; no account mutations."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.research.baseline_v2_continuous_replay import audit_fill_rules, audit_ledger
from moneymore.research.baseline_v2_daily_readiness import classify_held_inputs
from moneymore.research.replay_attribution import reconstruct_positions


def run(root):
    base = root / "state/baseline-v2-validation/daily-inputs"
    source = base / "daily-replay.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    if not report.get("requested_window_complete"):
        raise ValueError("COMPLETED_REQUESTED_WINDOW_REQUIRED_FOR_POST_RUN_AUDIT")
    directory = Path(report["output_directory"])
    frozen = json.loads((directory / "frozen-protocol.json").read_text(encoding="utf-8"))
    for name, digest in frozen["input_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("REPLAY_FROZEN_INPUT_CHANGED")
    score_report = json.loads((base / "daily-scores.json").read_text(encoding="utf-8"))
    scores = pd.read_parquet(Path(score_report["output_directory"]) / "daily-scores.parquet")
    pool_path = (
        Path(frozen["scoring"]["identity_audit"]["output_directory"])
        / "identity-top1000-preview.parquet"
    )
    if hashlib.sha256(pool_path.read_bytes()).hexdigest() != frozen["scoring"]["pool_sha256"]:
        raise ValueError("POST_RUN_POOL_HASH_MISMATCH")
    score_path = Path(score_report["output_directory"]) / "daily-scores.parquet"
    if str(score_path) not in frozen["input_hashes"]:
        raise ValueError("POST_RUN_SCORE_PANEL_NOT_FROZEN")
    pools = pd.read_parquet(pool_path)
    score_sets = {
        pd.Timestamp(d).strftime("%Y%m%d"): set(f.symbol) for d, f in scores.groupby("date")
    }
    pool_sets = {str(d): set(f.security_id) for d, f in pools.groupby("signal_date")}
    config = BacktestConfig.model_validate(
        {k: frozen["protocol"][k] for k in BacktestConfig.model_fields}
    )
    bars = pd.read_parquet(directory / "execution-bars.parquet")
    actions = pd.read_parquet(directory / "execution-actions.parquet")
    action_index = actions.set_index("action_key")
    results = []
    for item in report["runs"]:
        variant = item["variant"]
        equity = pd.read_parquet(directory / f"{variant}-equity.parquet")
        database = directory / f"{variant}.sqlite3"
        proof, ledger, rights = audit_ledger(database, equity, bars, actions, config.initial_cash)
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
            fills = pd.read_sql_query("SELECT * FROM fills", conn)
        holdings = reconstruct_positions(fills, ledger, equity.trade_date.astype(str).tolist())
        economic_holdings = {}
        holding_rows = []
        marks = bars.set_index(["date", "symbol"])["close"]
        for day in equity.itertuples():
            quantities = holdings[str(day.trade_date)].copy()
            for right in rights.itertuples():
                action = action_index.loc[right.action_key]
                if (
                    right.stock_ratio > 0
                    and action.ex_date <= pd.Timestamp(day.trade_date) < action.stock_list_date
                ):
                    quantities[right.symbol] = quantities.get(right.symbol, 0) + int(
                        right.entitled_quantity * right.stock_ratio
                    )
            quantities = {s: int(q) for s, q in quantities.items() if q > 0}
            economic_holdings[str(day.trade_date)] = quantities
            for symbol, quantity in quantities.items():
                value = quantity * float(marks.loc[(pd.Timestamp(day.trade_date), symbol)])
                holding_rows.append(
                    {
                        "trade_date": str(day.trade_date),
                        "symbol": symbol,
                        "economic_quantity": quantity,
                        "market_value": value,
                        "account_weight": value / day.equity,
                        "growth_board": symbol.startswith(("300", "301", "302", "688", "689")),
                    }
                )
        holding_frame = pd.DataFrame(holding_rows)
        holding_frame.to_parquet(
            directory / f"{variant}-reconstructed-holdings.parquet", index=False
        )
        events = json.loads(
            (directory / f"{variant}-closing-decisions.json").read_text(encoding="utf-8")
        )["events"]
        if [e["signal_date"] for e in events] != equity.trade_date.astype(str).tolist()[:-1]:
            raise ValueError("POST_RUN_CLOSING_DECISION_CALENDAR_MISMATCH")
        for event in events:
            day = event["signal_date"]
            quantities = economic_holdings[day]
            observed = {s: int(q) for s, q in event["actual_incumbents"].items() if q > 0}
            if quantities != observed:
                raise ValueError("CLOSING_INCUMBENTS_NOT_ACTUAL_RECONSTRUCTED_HOLDINGS")
            classified = classify_held_inputs(
                quantities, score_sets[day], universe_members=pool_sets[day]
            )
            if any(event[k] != classified[k] for k in classified):
                raise ValueError("SAVED_EXIT_OR_FAULT_CLASSIFICATION_MISMATCH")
            if (
                classified["input_faults"]
                and event["status"] != "BLOCKED_HELD_INPUT_HOLD_AND_ALERT"
            ):
                raise ValueError("INPUT_FAULT_PROTECTION_NOT_APPLIED")
            if set(classified["universe_exits"]) & set(event.get("weights", {})):
                raise ValueError("UNIVERSE_EXIT_IN_TARGET_WEIGHTS")
        opening = json.loads(
            (directory / f"{variant}-opening-audits.json").read_text(encoding="utf-8")
        )["plans"]
        blocked = Counter(
            p["plan"].get("reason", "UNSPECIFIED")
            for p in opening
            if p["plan"]["status"] == "BLOCKED"
        )
        results.append(
            {
                "variant": variant,
                "closing_incumbents_verified": len(events),
                "fill_rules": audit_fill_rules(fills.to_dict("records"), bars, config),
                "ledger_max_error": float(proof.equity_error.abs().max()),
                "corporate_ledger_entries": len(ledger),
                "entitlements": len(rights),
                "blocked_plan_reasons": dict(blocked),
                "primary_optimal_fallbacks": sum(
                    bool(p["plan"].get("primary_optimal_fallback", False)) for p in opening
                ),
                "input_hold_days": item["input_hold_days"],
                "mean_growth_board_account_weight": float(
                    holding_frame.loc[holding_frame.growth_board]
                    .groupby("trade_date")
                    .account_weight.sum()
                    .reindex(equity.trade_date.astype(str), fill_value=0)
                    .mean()
                ),
                "period_trading_by_calendar_year": {
                    str(year): {
                        "fills": len(rows),
                        "fees": float(rows.fee.sum()),
                        "gross_notional": float((rows.price * rows.quantity).sum()),
                    }
                    for year, rows in fills.groupby(fills.trade_date.astype(str).str[:4])
                },
            }
        )
    output = {
        "status": "SAVED_DAILY_LEDGER_AND_INCUMBENT_AUDIT_PASS",
        "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_directory": str(directory),
        "runs": results,
        "release_authorized": False,
        "production_changed": False,
    }
    write_diagnostic(directory / "post-run-audit.json", output)
    write_diagnostic(base / "daily-replay-audit.json", output)
    gates = {
        "status": "BLOCKED_NOT_FINAL_V2",
        "source_report_sha256": output["source_report_sha256"],
        "source_audit_sha256": hashlib.sha256(
            (directory / "post-run-audit.json").read_bytes()
        ).hexdigest(),
        "frozen_weight_report_sha256": frozen["scoring"]["weight_report_sha256"],
        "new_evidence": {
            "full_two_year_daily_integer_cost_replay": bool(report["two_year_window_complete"]),
            "independent_daily_ledger_and_actual_incumbents": True,
            "no_input_holds_or_blocked_solver_plans": all(
                not r["input_hold_days"] and not r["blocked_plan_reasons"] for r in results
            ),
        },
        "remaining": [
            "ORIGINAL_FINANCIAL_VINTAGES_NOT_ATTESTED",
            "FULL_UNION_ADJUSTMENT_ECONOMIC_VALIDATION_NOT_COMPLETED",
            "LEARNED_WEIGHT_STATISTICAL_AND_ROLLING_ROBUSTNESS_NOT_REVALIDATED",
            "YEAR_STYLE_AND_STOCK_ATTRIBUTION_AND_EXECUTION_STRESS_REQUIRED",
            "NOT_NEW_UNTOUCHED_HOLDOUT_AFTER_RESEARCH_ITERATION",
        ],
        "frozen_weight_report_not_overwritten": True,
        "release_authorized": False,
        "production_changed": False,
    }
    write_diagnostic(directory / "validation-gates.json", gates)
    write_diagnostic(base / "daily-validation-gates.json", gates)
    return output


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).resolve().parents[1])))
