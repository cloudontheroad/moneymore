"""Bounded retrospective mechanics replay using genuinely archived scores."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import clean_json, trading_dates
from moneymore.config import BacktestConfig
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.data.store import ParquetStore
from moneymore.execution_replay import run_execution_replay
from moneymore.research.integer_execution import plan_integer_execution
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.replay_attribution import reconstruct_cash, reconstruct_positions
from moneymore.research.style_replay_inputs import prospective_capture, daily_risk_snapshot, optimize_daily
from moneymore.research.style_replay_inputs import measured_style
from moneymore.portfolio_constructor import global_topk_portfolio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slippage-bps", type=int, choices=[2, 10], default=10)
    parser.add_argument("--start", default="20261016")
    parser.add_argument("--end", default="20261021")
    parser.add_argument("--variant", choices=["named_style", "topk_daily_control"], default="named_style")
    parser.add_argument("--integer-target-seconds", type=int, choices=[30, 60], default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable replay")
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root/"data")
    calendar = trading_dates(store, args.end)
    start, end = args.start, args.end
    if not "20261012" <= start < end <= "20261026":
        raise ValueError("WINDOW_OUTSIDE_BOUNDED_PREPARED_EVIDENCE")
    dates = [d for d in calendar if start <= d <= end]
    reports = {}
    source = root/"state/qlib-candidate-shadow/20261010"
    for date in dates:
        path = source/(date+".json"); row = json.loads(path.read_text(encoding="utf-8"))
        if row["trade_date"] != date or not prospective_capture(date, row.get("created_at"), calendar[calendar.index(date)+1]):
            raise ValueError("NONPROSPECTIVE_ARCHIVED_PREDICTION:"+date)
        row["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest(); reports[date] = row
    symbols = sorted({s["instrument"] for r in reports.values() for s in r["scores"]})
    risk_start = calendar[calendar.index(start)-120]
    filters = [("trade_date", ">=", risk_start), ("trade_date", "<=", end), ("ts_code", "in", symbols)]
    daily = store.read("daily", columns=["trade_date", "ts_code", "open", "close", "amount"], filters=filters)
    adj = store.read("adj_factor", columns=["trade_date", "ts_code", "adj_factor"], filters=filters)
    daily = daily.merge(adj, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    daily["total_price"] = daily.close*daily.adj_factor
    basic = store.read("daily_basic", filters=[("trade_date", ">=", start), ("trade_date", "<=", end), ("ts_code", "in", symbols)])
    finances = research_financial_frame(store)
    membership = store.read("baseline_v2_industry_membership")
    preflight = json.loads((root/"state/enhanced-indexing-research/replay-preflight.json").read_text(encoding="utf-8"))
    evidence = Path(preflight["evidence_directory"])
    limit_files = [evidence/f"stk_limit-{d}.parquet" for d in dates]
    limits = pd.concat([pd.read_parquet(p) for p in limit_files])
    market = daily.loc[daily.trade_date.isin(dates)].merge(limits[["trade_date", "ts_code", "up_limit", "down_limit"]], on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    market["can_buy"] = market.open.lt(market.up_limit-1e-8)
    market["can_sell"] = market.open.gt(market.down_limit+1e-8)
    market = market.loc[market.open.gt(0) & market.close.gt(0)].rename(columns={"trade_date": "date", "ts_code": "symbol"})
    market["date"] = pd.to_datetime(market.date)
    ca_report = json.loads((root/"state/enhanced-indexing-research/corporate-action-audit.json").read_text(encoding="utf-8"))
    if ca_report["status"] != "READY_FOR_CA_REPLAY_AUDIT":
        raise ValueError("CORPORATE_ACTION_EVIDENCE_NOT_READY")
    action_path = Path(ca_report["action_file"])
    raw_actions = pd.read_parquet(action_path)
    actions = official_replay_actions(raw_actions.loc[raw_actions.ts_code.isin(symbols)])
    config = BacktestConfig.from_yaml(root/"configs/default.yaml").model_copy(update={"slippage_bps": args.slippage_bps})
    args.output.mkdir(parents=True)
    protected = [store.curated/(n+".parquet") for n in ["multi_sector_account_daily", "qlib_challenger_account_daily", "qlib_candidate_account_daily"]]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    context = {}; events = []
    empty = pd.DataFrame(columns=["date", "symbol", "target_weight"])

    def closing(date, snapshot):
        context.clear()
        row = reports[date]
        scores = pd.DataFrame(row["scores"]).set_index("instrument").score
        if not scores.index.is_unique or not np.isfinite(scores).all():
            raise ValueError("INVALID_ARCHIVED_SCORES")
        held = {p["symbol"]: int(p["quantity"]) for p in snapshot["positions"] if p["quantity"] > 0}
        try:
            selected, controls, F, covariance, specific, wb, bound, missing = daily_risk_snapshot(date, scores, held, calendar, daily, basic, finances, membership)
            close = daily.loc[daily.trade_date.eq(date)].set_index("ts_code").close
            held_values = pd.Series(held, dtype=float)*close.reindex(list(held))
            if not np.isfinite(held_values).all():
                raise ValueError("MISSING_HELD_MARK")
            actual_style = measured_style(held_values, controls, F, covariance, specific, wb)
            if date == end:
                events.append({"date": date, "status": "TERMINAL_MARK_ONLY", "actual_style": actual_style, "monitoring_limits": bound.to_dict()})
                controls.to_parquet(args.output/f"controls-{date}.parquet")
                return empty.copy()
            if args.variant == "named_style":
                target = optimize_daily(scores, selected, controls, F, covariance, specific, wb, bound, held_values)
            else:
                # Same observed risk-eligible pool and daily decision schedule.
                # Reuse native global portfolio constructor, not a new formula.
                ranking = scores.reindex(selected.index).dropna().rename("score").reset_index()
                ranking.columns = ["instrument", "score"]
                _, allocations, _ = global_topk_portfolio(ranking, set(held), symbol_column="instrument",
                    top_k=10, exit_rank=15, max_replacements=3, weighting_method="confidence_softmax",
                    confidence_temperature=1.25, confidence_shrinkage=.45)
                target = pd.Series(allocations).reindex(selected.index, fill_value=0.)
            target_style = measured_style(target, controls, F, covariance, specific, wb)
            monitoring_bound = bound.copy()
            if args.variant == "topk_daily_control":
                bound = pd.Series(1e6, index=bound.index)
            context.update(date=date, target=target, controls=controls, benchmark=wb, limits=bound, source_sha256=row["source_sha256"])
            controls.to_parquet(args.output/f"controls-{date}.parquet")
            selected.to_parquet(args.output/f"observations-{date}.parquet")
            pd.DataFrame({"target": target, "benchmark": wb}).to_parquet(args.output/f"weights-{date}.parquet")
            events.append({"date": date, "status": "CONTINUOUS_PASS", "initial_state": "ACTUAL_REPLAY_HOLDINGS" if held else "COLD_CASH",
                "risk_assets": len(controls), "score_assets": len(scores), "support": int((target>1e-6).sum()),
                "held_count": len(held), "source_sha256": row["source_sha256"], "created_at": row["created_at"],
                "actual_style": actual_style, "target_style": target_style, "monitoring_limits": monitoring_bound.to_dict(),
                "unknown_counts": missing["missing_counts"], "pool_exit_held": sorted(set(held)-set(scores.index))})
        except ValueError as error:
            events.append({"date": date, "status": "BLOCKED_CLOSE", "reason": str(error)})
        print(json.dumps({k: v for k, v in events[-1].items() if k in ["date", "status", "reason", "support", "held_count"]}), flush=True)
        return empty.copy()

    def opening(date, snapshot, quotes):
        if not context:
            return {"status": "NO_ELIGIBLE_PRIOR_SIGNAL", "orders": []}
        if snapshot.get("unsettled_stock_rights"):
            return {"status": "BLOCKED_UNSETTLED_STOCK_RIGHTS", "orders": []}
        quantities = {p["symbol"]: int(p["quantity"]) for p in snapshot["positions"] if p["quantity"] > 0}
        # One open per increasing exchange day; all existing purchases predate
        # this session. PaperBroker independently enforces its own T+1 book.
        plan = plan_integer_execution(context["target"], quotes.open, quantities, quantities,
            float(snapshot["cash"]), float(snapshot["equity"]), config, context["controls"], context["limits"],
            can_buy=quotes.can_buy.to_dict(), can_sell=quotes.can_sell.to_dict(), benchmark_weights=context["benchmark"],
            style_controls_prestandardized=True, time_limit=20., solver_threads=2, mip_relative_gap=.01,
            research_staged_warm_start=True, research_target_time_limit=float(args.integer_target_seconds))
        plan.update(signal_date=context["date"], source_sha256=context["source_sha256"])
        if plan["status"] == "PASS_REFERENCE_QUOTES_ONLY":
            centered = context["controls"]-(context["benchmark"] @ context["controls"])
            plan["execution_style_proof"] = {"standardized": centered.to_dict("index"), "limits": context["limits"].to_dict()}
        print(json.dumps({"open": date, "status": plan["status"], "reason": plan.get("reason"), "orders": len(plan["orders"])}), flush=True)
        return plan

    database = args.output/"paper.sqlite3"
    result = run_execution_replay(bars=market, targets=empty, config=config, database=database,
        strategy_id="named_style_engineering", account_id="isolated", corporate_actions=actions,
        closing_target_provider=closing, opening_plan_provider=opening, accrue_action_receivables=True)
    result.equity.to_parquet(args.output/"equity.parquet", index=False)
    with sqlite3.connect(database) as connection:
        fills = pd.read_sql_query("SELECT * FROM fills", connection)
        ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", connection)
        final_positions = pd.read_sql_query("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0", connection).set_index("symbol").quantity.to_dict()
    actual_dates = result.equity.trade_date.astype(str).tolist()
    cash = reconstruct_cash(fills, ledger, actual_dates, config.initial_cash)
    holdings = reconstruct_positions(fills, ledger, actual_dates)
    max_cash_error = max(abs(cash[d]-float(result.equity.set_index("trade_date").loc[d,"cash"])) for d in actual_dates)
    if max_cash_error > .01 or not result.reconciliation["matched"]:
        raise ValueError("INDEPENDENT_LEDGER_RECONCILIATION_FAILED")
    if holdings[actual_dates[-1]] != final_positions:
        raise ValueError("INDEPENDENT_POSITION_RECONCILIATION_FAILED")
    fill_limits = limits.set_index(["trade_date", "ts_code"])
    for fill in fills.itertuples():
        limit = fill_limits.loc[(str(fill.trade_date), fill.symbol)]
        if not limit.down_limit-1e-8 <= fill.price <= limit.up_limit+1e-8:
            raise ValueError("SLIPPED_FILL_OUTSIDE_OFFICIAL_LIMITS")
    unchanged = before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report = {"status": "COMPLETED_EX_POST_ENGINEERING_NOT_STRATEGY_APPROVAL", "orders_enabled": False,
        "variant": args.variant, "decision_frequency": "DAILY_MATCHED_ABLATION_NOT_LIVE_SCHEDULE_REPRODUCTION",
        "integer_target_seconds": args.integer_target_seconds,
        "config": config.model_dump(),
        "start": start, "end": end, "slippage_bps": args.slippage_bps, "events": events,
        "summary": result.summary, "reconciliation": result.reconciliation, "cash_max_error": max_cash_error,
        "reconstructed_holdings": holdings, "protected_equities_unchanged": unchanged,
        "independent_final_positions_match": True, "slipped_fills_within_official_limits": True,
        "input_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*limit_files, action_path]},
        "limitations": ["retrospective mechanics test, not unseen performance or publication evidence", "as-of financial visibility but original supplier vintages not proven",
            "daily open reference matching, not live/intraday execution proof", "gross cash dividend, supplemental holding-period dividend tax not modeled",
            "missing risk history excluded; no unsupported suspension forward-fill", "any blocked sessions remain explicit, not skipped or filled at hindsight prices"]}
    (args.output/"results.json").write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "fills": len(result.fills), "cash_max_error": max_cash_error, "protected_unchanged": unchanged}), flush=True)


if __name__ == "__main__":
    main()
