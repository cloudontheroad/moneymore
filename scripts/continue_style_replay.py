"""Continue a validated isolated ledger by one session, never reset holdings."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import clean_json, trading_dates
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.execution.paper import PaperBroker, ExecutionBar
from moneymore.execution_replay import _summary
from moneymore.research.integer_execution import plan_integer_execution, paper_intents_from_integer_plan
from moneymore.research.replay_actions import official_replay_actions, economic_snapshot
from moneymore.research.replay_attribution import reconstruct_cash, reconstruct_positions
from moneymore.research.style_replay_inputs import prospective_capture, daily_risk_snapshot, optimize_daily, measured_style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable continuation")
    parent_path = args.source/"results.json"
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)
    audit = json.loads((args.source/"independent-audit.json").read_text(encoding="utf-8"))
    if (parent["end"] != "20261026" or parent.get("variant") != "named_style"
            or audit["status"] != "PASS_ENGINEERING_LEDGER_ONLY" or audit["source_sha256"] != hashlib.sha256(parent_bytes).hexdigest()):
        raise ValueError("SOURCE_NOT_VALIDATED_CHECKPOINT")
    manifest = json.loads((args.evidence/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "READY_FOR_ISOLATED_EXTENSION" or manifest["date"] != "20261027":
        raise ValueError("EXTENSION_EVIDENCE_INCOMPLETE")
    for row in manifest["requests"]:
        if hashlib.sha256((args.evidence/row["file"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("EXTENSION_EVIDENCE_HASH_CHANGED")
    for path, digest in parent["input_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("PARENT_INPUT_CHANGED")
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root/"data")
    protected = [store.curated/(n+".parquet") for n in ["multi_sector_account_daily", "qlib_challenger_account_daily", "qlib_candidate_account_daily"]]
    protected_before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    signal_date, date = "20261026", "20261027"
    calendar = trading_dates(store, date)
    score_reports = {}
    for day in (signal_date, date):
        path = root/f"state/qlib-candidate-shadow/20261010/{day}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["trade_date"] != day or not prospective_capture(day, row.get("created_at"), calendar[calendar.index(day)+1]):
            raise ValueError("INVALID_CAPTURE_TIME")
        row["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest(); score_reports[day] = row
    held = parent["reconstructed_holdings"][signal_date]
    symbols = sorted(set(held) | {s["instrument"] for r in score_reports.values() for s in r["scores"]})
    risk_start = calendar[calendar.index(signal_date)-120]
    filters = [("trade_date", ">=", risk_start), ("trade_date", "<=", date), ("ts_code", "in", symbols)]
    daily = store.read("daily", columns=["ts_code", "trade_date", "open", "close", "amount"], filters=filters)
    daily = daily.merge(store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters), on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    daily["total_price"] = daily.close*daily.adj_factor
    basic = store.read("daily_basic", filters=[("trade_date", ">=", signal_date), ("trade_date", "<=", date), ("ts_code", "in", symbols)])
    finances = research_financial_frame(store); membership = store.read("baseline_v2_industry_membership")
    prior_action_path = next(Path(p) for p in parent["input_hashes"] if Path(p).name == "dividend-window.parquet")
    merged_actions = pd.concat([pd.read_parquet(prior_action_path), pd.read_parquet(args.evidence/"dividend-ex-20261027.parquet"),
        pd.read_parquet(args.evidence/"dividend-record-20261027.parquet")], ignore_index=True).drop_duplicates()
    actions = official_replay_actions(merged_actions.loc[merged_actions.ts_code.isin(symbols)])
    args.output.mkdir(parents=True)
    merged_actions.to_parquet(args.output/"dividend-window.parquet", index=False)
    source_db = args.source/"paper.sqlite3"; source_db_hash = hashlib.sha256(source_db.read_bytes()).hexdigest()
    database = args.output/"paper.sqlite3"
    with sqlite3.connect(source_db.resolve().as_uri()+"?mode=ro", uri=True) as source_connection, sqlite3.connect(database) as destination:
        if source_connection.execute("SELECT MAX(trade_date) FROM fills").fetchone()[0] >= date:
            raise ValueError("SOURCE_HAS_FUTURE_FILLS")
        source_connection.backup(destination)
    broker = PaperBroker(database)
    config = BacktestConfig.model_validate(parent["config"])
    prior_marks = daily.loc[daily.trade_date.eq(signal_date)].set_index("ts_code").close
    prior_snapshot = economic_snapshot(broker, broker.account_snapshot(prior_marks.to_dict(), "isolated"), prior_marks.to_dict(), "isolated", signal_date, actions)
    if {p["symbol"]:int(p["quantity"]) for p in prior_snapshot["positions"]} != held:
        raise ValueError("CHECKPOINT_POSITIONS_CHANGED")
    original_equity = pd.read_parquet(args.source/"equity.parquet")
    if abs(prior_snapshot["equity"]-float(original_equity.iloc[-1].equity)) > .01:
        raise ValueError("CHECKPOINT_EQUITY_CHANGED")
    scores = pd.DataFrame(score_reports[signal_date]["scores"]).set_index("instrument").score
    selected, controls, F, covariance, specific, wb, bound, missing = daily_risk_snapshot(signal_date, scores, held, calendar, daily, basic, finances, membership)
    pd.testing.assert_frame_equal(controls, pd.read_parquet(args.source/f"controls-{signal_date}.parquet"), check_exact=True)
    actual_style = measured_style(pd.Series(held)*prior_marks.reindex(held), controls, F, covariance, specific, wb)
    stored = next(e["actual_style"]["active_controls"] for e in parent["events"] if e["date"] == signal_date)
    if abs(pd.Series(actual_style["active_controls"])-pd.Series(stored)).max() > 1e-8:
        raise ValueError("CHECKPOINT_STYLE_BASIS_CHANGED")
    target = optimize_daily(scores, selected, controls, F, covariance, specific, wb, bound, pd.Series(held)*prior_marks.reindex(held))
    controls.to_parquet(args.output/f"controls-{signal_date}.parquet")
    selected.to_parquet(args.output/f"observations-{signal_date}.parquet")
    pd.DataFrame({"target":target, "benchmark":wb}).to_parquet(args.output/f"weights-{signal_date}.parquet")
    quote = daily.loc[daily.trade_date.eq(date)].set_index("ts_code")
    limits = pd.read_parquet(args.evidence/"stk_limit-20261027.parquet").set_index("ts_code")
    susp = pd.read_parquet(args.evidence/"suspend_d-20261027.parquet")
    full_halt = set(susp.loc[susp.suspend_type.eq("S") & (susp.suspend_timing.isna() | susp.suspend_timing.eq("")), "ts_code"])
    needed = set(held) | set(target[target>1e-6].index)
    if set(held)-set(quote.index) or not np.isfinite(quote.reindex(list(needed))[["open", "close"]].to_numpy()).all():
        raise ValueError("HELD_OR_REQUIRED_OPEN_MARK_MISSING_NO_FILL")
    can_buy = (quote.open*(1+config.slippage_bps/10000) < limits.up_limit.reindex(quote.index)-1e-8) & ~quote.index.isin(full_halt)
    can_sell = (quote.open*(1-config.slippage_bps/10000) > limits.down_limit.reindex(quote.index)+1e-8) & ~quote.index.isin(full_halt)
    broker.settle_corporate_actions("isolated", date)
    broker.settle_available_positions("isolated", date)
    opening = economic_snapshot(broker, broker.account_snapshot(quote.open.to_dict(), "isolated"), quote.open.to_dict(), "isolated", date, actions)
    if opening["unsettled_stock_rights"]:
        raise ValueError("UNSETTLED_STOCK_RIGHTS_REQUIRE_SEPARATE_PLANNING")
    quantities = {p["symbol"]:int(p["quantity"]) for p in opening["positions"]}
    available = {p["symbol"]:int(p["available_quantity"]) for p in opening["positions"]}
    plan = plan_integer_execution(target, quote.open, quantities, available, float(opening["cash"]), float(opening["equity"]), config,
        controls, bound, can_buy=can_buy.to_dict(), can_sell=can_sell.to_dict(), benchmark_weights=wb, style_controls_prestandardized=True,
        time_limit=20., solver_threads=2, mip_relative_gap=.01, research_staged_warm_start=True, research_target_time_limit=60.)
    plan.update(signal_date=signal_date, source_sha256=score_reports[signal_date]["source_sha256"])
    extension_audit = {"trade_date":date, "plan":plan}
    if plan["status"] == "PASS_REFERENCE_QUOTES_ONLY":
        centered = controls-(wb @ controls)
        plan["execution_style_proof"] = {"standardized":centered.to_dict("index"), "limits":bound.to_dict()}
        broker.cancel_pending("isolated", "RESEARCH_CONTINUATION_REPLAN", signal_date=None)
        for risk in paper_intents_from_integer_plan(plan, "named_style_engineering", signal_date):
            broker.submit(risk, "isolated")
        for order in plan["orders"]:
            symbol = order["symbol"]
            broker.execute_pending(ExecutionBar(symbol, date, float(quote.loc[symbol,"open"]), float(quote.loc[symbol,"close"]),
                bool(can_buy[symbol]), bool(can_sell[symbol])), config, "isolated")
        actual = broker.account_snapshot(quote.open.to_dict(), "isolated")
        actual_q = {p["symbol"]:int(p["quantity"]) for p in actual["positions"]}
        expected = {s:int(q) for s,q in plan["target_quantities"].items() if q>0}
        if actual_q != expected or abs(actual["cash"]-plan["estimated_cash"]) > .01:
            raise ValueError("CONTINUED_EXECUTION_MISMATCH")
        exposures = (pd.Series(actual_q)*quote.open.reindex(actual_q)); exposures /= exposures.sum()
        measured = exposures @ centered.reindex(exposures.index)
        excess = float((abs(measured)-bound).max())
        if excess > 1e-5:
            raise ValueError("CONTINUED_ACTUAL_STYLE_BREACH")
        extension_audit["execution_check"] = {"status":"MATCHED", "quantities_match":True, "cash_error":actual["cash"]-plan["estimated_cash"],
            "actual_quantities":actual_q, "actual_style_exposures":measured.to_dict(), "maximum_style_excess":excess}
    for action in actions.loc[actions.record_date.eq(pd.Timestamp(date))].to_dict("records"):
        broker.register_corporate_entitlement("isolated", action["symbol"], action["action_key"], date,
            float(action["cash_per_share"]), float(action["stock_ratio"]),
            action["cash_pay_date"].strftime("%Y%m%d") if pd.notna(action["cash_pay_date"]) else None,
            action["stock_list_date"].strftime("%Y%m%d") if pd.notna(action["stock_list_date"]) else None)
    closing = economic_snapshot(broker, broker.account_snapshot(quote.close.to_dict(), "isolated"), quote.close.to_dict(), "isolated", date, actions)
    equity = pd.concat([original_equity, pd.DataFrame([{"trade_date":date, "equity":closing["equity"], "cash":closing["cash"],
        "market_value":closing["market_value"], "receivable_cash":closing["receivable_cash"], "unsettled_stock_value":closing["unsettled_stock_value"],
        "gross_exposure":closing["market_value"]/closing["equity"], "daily_return":closing["equity"]/float(original_equity.iloc[-1].equity)-1}])], ignore_index=True)
    equity["normalized_nav"] = equity.equity/equity.equity.iloc[0]; equity["drawdown"] = equity.normalized_nav/equity.normalized_nav.cummax()-1
    equity.to_parquet(args.output/"equity.parquet", index=False)
    with sqlite3.connect(database) as connection:
        fills = pd.read_sql_query("SELECT * FROM fills", connection); ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", connection)
    dates = equity.trade_date.astype(str).tolist()
    cash = reconstruct_cash(fills, ledger, dates, config.initial_cash); positions = reconstruct_positions(fills, ledger, dates)
    max_error = max(abs(cash[d]-float(equity.set_index("trade_date").loc[d,"cash"])) for d in dates)
    if max_error > .01 or positions[date] != {p["symbol"]:int(p["quantity"]) for p in closing["positions"]}:
        raise ValueError("CONTINUATION_INDEPENDENT_LEDGER_MISMATCH")
    reconciliation = asdict(broker.reconcile("isolated"))
    if not reconciliation["matched"]:
        raise ValueError("CONTINUATION_RECONCILIATION_FAILED")
    final_scores = pd.DataFrame(score_reports[date]["scores"]).set_index("instrument").score
    final_selected, final_controls, f, c, s, final_wb, final_bound, _ = daily_risk_snapshot(date, final_scores, positions[date], calendar, daily, basic, finances, membership)
    final_controls.to_parquet(args.output/f"controls-{date}.parquet")
    final_selected.to_parquet(args.output/f"observations-{date}.parquet")
    final_style = measured_style(pd.Series(positions[date])*quote.close.reindex(positions[date]), final_controls, f, c, s, final_wb)
    events = [e for e in parent["events"] if e["date"] != signal_date]+[
        {"date":signal_date, "status":"CONTINUOUS_PASS", "actual_style":actual_style, "monitoring_limits":bound.to_dict(),
            "source_sha256":score_reports[signal_date]["source_sha256"], "created_at":score_reports[signal_date]["created_at"], "initial_state":"ACTUAL_CONTINUED_HOLDINGS"},
        {"date":date, "status":"TERMINAL_MARK_ONLY", "actual_style":final_style, "monitoring_limits":final_bound.to_dict()}]
    for day in dates:
        path = args.source/f"controls-{day}.parquet"
        if day not in (signal_date,date) and path.exists():
            pd.read_parquet(path).to_parquet(args.output/path.name)
    summary = _summary(equity, broker.orders(), broker.fills("isolated"), broker.execution_attempts(account_id="isolated"), reconciliation)
    summary["opening_plan_audits"] = parent["summary"]["opening_plan_audits"]+[extension_audit]
    input_hashes = {p:h for p,h in parent["input_hashes"].items() if p != str(prior_action_path)}
    input_hashes.update({str(args.evidence/r["file"]):r["sha256"] for r in manifest["requests"]})
    input_hashes[str(args.output/"dividend-window.parquet")] = hashlib.sha256((args.output/"dividend-window.parquet").read_bytes()).hexdigest()
    unchanged = hashlib.sha256(source_db.read_bytes()).hexdigest() == source_db_hash and parent_path.read_bytes() == parent_bytes
    if not unchanged:
        raise ValueError("SOURCE_CHECKPOINT_CHANGED")
    result = {**parent, "end":date, "events":events, "summary":summary, "reconciliation":reconciliation, "cash_max_error":max_error,
        "reconstructed_holdings":positions, "input_hashes":input_hashes, "resume_source_sha256":hashlib.sha256(parent_bytes).hexdigest(),
        "resume_source_database_unchanged":True, "resume_method":"READONLY_SQLITE_BACKUP_WITH_ALL_CA_ENTITLEMENTS_NO_REINITIALIZATION",
        "continuation_open_status":plan["status"], "continuation_orders":len(plan["orders"])}
    result["protected_equities_unchanged"] = protected_before == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    result["obligation_namespace"] = parent.get("obligation_namespace", hashlib.sha256(parent_bytes).hexdigest())
    (args.output/"results.json").write_text(json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"open_status":plan["status"], "orders":len(plan["orders"]), "cash_error":max_error, "source_unchanged":unchanged}), flush=True)


if __name__ == "__main__":
    main()
