"""Independent same-signal continuous style ablation; no promotion or live writes."""

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import prediction_reports, trading_dates, write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.execution_replay import run_execution_replay
from moneymore.research.enhanced_indexing import (
    EnhancedPolicy,
    carry_confirmed_suspensions,
    optimize_snapshot,
)
from moneymore.research.integer_execution import plan_integer_execution
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.style_feasibility import attainable_style_retention

EMPTY = pd.DataFrame(columns=["date", "symbol", "target_weight"])


def prospective(report, calendar):
    date = report["trade_date"]
    i = calendar.index(date)
    captured = pd.to_datetime(report.get("created_at"), utc=True, errors="coerce")
    return (i + 1 < len(calendar) and not pd.isna(captured)
            and pd.Timestamp(date + " 15:00", tz="Asia/Shanghai").tz_convert("UTC") <= captured
            < pd.Timestamp(calendar[i + 1] + " 09:30", tz="Asia/Shanghai").tz_convert("UTC"))


def run(root, account_filter=None, integer=False):
    store = ParquetStore(root / "data")
    folder = root / "state/enhanced-indexing-research"
    preflight = json.loads((folder / "replay-preflight.json").read_text(encoding="utf-8"))
    ca_audit = json.loads((folder / "corporate-action-audit.json").read_text(encoding="utf-8"))
    if ca_audit["status"] != "READY_FOR_CA_REPLAY_AUDIT":
        raise ValueError("CORPORATE_ACTION_FETCH_INCOMPLETE")
    start, end = preflight["start_date"], preflight["as_of_date"]
    calendar = trading_dates(store, end)
    dates = [d for d in calendar if start <= d <= end]
    risk_start = calendar[calendar.index(start) - 120]
    evidence = Path(preflight["evidence_directory"])
    output_dir = folder / "continuous-replay" / uuid4().hex
    output_dir.mkdir(parents=True)
    daily = store.read("daily", filters=[("trade_date", ">=", risk_start), ("trade_date", "<=", end)])
    adj = store.read("adj_factor", filters=[("trade_date", ">=", risk_start), ("trade_date", "<=", end)])
    joined = daily.merge(adj, on=["trade_date", "ts_code"], validate="one_to_one")
    joined["adjusted"] = joined.close * joined.adj_factor
    prices = joined.pivot(index="trade_date", columns="ts_code", values="adjusted").reindex([d for d in calendar if risk_start <= d <= end])
    suspensions = pd.concat([pd.read_parquet(evidence / f"suspend_d-{d}.parquet") for d in dates])
    proof = pd.concat([suspensions, store.read("enhanced_indexing_suspension_proof")], ignore_index=True)
    history_audit = folder / "risk-history-audit.json"
    if history_audit.exists():
        verified = json.loads(history_audit.read_text(encoding="utf-8"))
        for item in verified.get("symbols", []):
            if item.get("suspension_source_file"):
                source = Path(item["suspension_source_file"])
                if hashlib.sha256(source.read_bytes()).hexdigest() != item["suspension_source_sha256"]:
                    raise ValueError("SUSPENSION_EVIDENCE_HASH_MISMATCH")
                proof = pd.concat([proof, pd.read_parquet(source)], ignore_index=True)
    # Raw source proof has null suspend_timing; curated derived proof carries its own flag.
    proof["verified_full_session"] = (proof.suspend_type.eq("S") &
        (proof.suspend_timing.isna() | proof.suspend_timing.eq("")))
    prices, carries = carry_confirmed_suspensions(prices, proof)
    basic = store.read("daily_basic", filters=[("trade_date", ">=", risk_start), ("trade_date", "<=", end)])
    actions = pd.read_parquet(ca_audit["action_file"])
    # This protocol is A-share only. B-share settlement chronology differs.
    excluded_b_share_actions = int(actions.ts_code.str.startswith(("200", "900")).sum())
    actions = actions.loc[~actions.ts_code.str.startswith(("200", "900"))]
    replay_actions = official_replay_actions(actions)
    limits = pd.concat([pd.read_parquet(evidence / f"stk_limit-{d}.parquet") for d in dates])
    market = daily.loc[daily.trade_date.isin(dates)].merge(limits, on=["trade_date", "ts_code"], suffixes=("", "_official"), validate="one_to_one")
    market["can_buy"] = market.open.lt(market.up_limit_official - 1e-8)
    market["can_sell"] = market.open.gt(market.down_limit_official + 1e-8)
    sources = {"main_gru": root / "state/qlib-challenger-shadow"}
    sources.update({"candidate_" + p.name: p for p in (root / "state/qlib-candidate-shadow").iterdir() if p.is_dir()})
    config = BacktestConfig.from_yaml(root / "configs/default.yaml")
    protected = [store.curated / (n + ".parquet") for n in ("multi_sector_account_daily", "qlib_challenger_account_daily", "qlib_candidate_account_daily")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    result = dict(as_of_date=end, start_date=start, orders_enabled=False, automatic_promotion=False,
                  mode="EX_POST_MECHANISM_ABLATION_NOT_UNSEEN_OOS", output_directory=str(output_dir), config=config.model_dump(),
                  warning="Variants selected after observing this period; no winner/promotion inference. Same target gross, actual exposures can differ due to cash/lots/fills.",
                  runs=[], suspension_carries=carries)
    result["corporate_action_policy"] = "implemented_only_ex_date_accrual_gross_cash_no_double_count_of_stock_transfers"
    result["dividend_tax_caveat"] = "Gross cash dividend; holding-period supplemental dividend tax not modeled. Not after-tax live PnL."
    result["action_events"] = len(replay_actions)
    result["excluded_b_share_actions"] = excluded_b_share_actions
    for account, directory in sources.items():
        if account_filter and account != account_filter:
            continue
        reports = {r["trade_date"]: r for r in prediction_reports(directory, end) if r["trade_date"] in dates}
        pool = sorted({s["instrument"] for r in reports.values() for s in r["scores"]})
        bars = market.loc[market.ts_code.isin(pool)].rename(columns={"trade_date": "date", "ts_code": "symbol"})
        bars = bars[["date", "symbol", "open", "close", "can_buy", "can_sell"]].copy()
        bars["date"] = pd.to_datetime(bars.date)
        variants = ("native_pca", "style_feasible") if integer else ("saved_targets", "native_pca", "style_retain_75pct", "style_feasible")
        for variant in variants:
            audit = []
            context = {}

            def opening(date, snapshot, quotes):
                if not context:
                    return dict(status="NO_PRIOR_SIGNAL", orders=[])
                if snapshot.get("unsettled_stock_rights"):
                    return dict(status="BLOCKED_UNSETTLED_STOCK_BASIS", orders=[])
                quantities = {p["symbol"]: int(p["quantity"]) for p in snapshot["positions"]}
                # All physical positions predate this open; A-share T+1 has matured.
                plan = plan_integer_execution(
                    targets=pd.Series(context["targets"]), prices=quotes.open,
                    quantities=quantities, available=quantities,
                    cash=float(snapshot["cash"]), equity=float(snapshot["equity"]),
                    config=config, styles=context["styles"], style_limits=context["limits"],
                    can_buy=quotes.can_buy.to_dict(), can_sell=quotes.can_sell.to_dict())
                if plan["status"] == "PASS_REFERENCE_QUOTES_ONLY":
                    styles = context["styles"]
                    standardized = (styles - styles.mean()) / styles.std(ddof=0).replace(0, 1)
                    plan["execution_style_proof"] = dict(
                        standardized=standardized.reindex(plan["target_quantities"]).to_dict("index"),
                        limits=context["limits"])
                return {**plan, "signal_date": context["date"],
                        "source_sha256": context["source_sha256"],
                        "signal_target_weights": context["targets"],
                        "signal_style_limits": context["limits"]}

            def provider(date, snapshot):
                held = {p["symbol"]: p["quantity"] for p in snapshot["positions"] if p["quantity"] > 0}
                for symbol, quantity in snapshot.get("unsettled_stock_rights", {}).items():
                    held[symbol] = held.get(symbol, 0) + quantity
                missing_held = set(held) - set(bars.loc[bars.date.eq(pd.Timestamp(date)), "symbol"])
                if missing_held:
                    raise ValueError("HELD_MARK_MISSING:" + ",".join(sorted(missing_held)))
                if date == end:
                    return EMPTY.copy()  # No unobservable next-open trades at the terminal close.
                report = reports.get(date)
                if report is None or not prospective(report, calendar):
                    raise ValueError("MISSING_OR_RECONSTRUCTED_PREDICTION:" + date)
                score_frame = pd.DataFrame(report["scores"])
                if set(score_frame.datetime.astype(str).str.replace("-", "", regex=False)) != {date}:
                    raise ValueError("SCORE_DATE_MISMATCH")
                # Fail rather than silently lose a halted held asset from marking.
                targets = report.get("target_weights", {})
                if not targets or abs(sum(targets.values()) - 1.) > 1e-6:
                    raise ValueError("SAVED_FULL_GROSS_TARGET_REQUIRED:" + date)
                if variant != "saved_targets" and (held or integer):
                    window = prices.loc[:date].tail(121)
                    if len(window) != 121:
                        raise ValueError("INSUFFICIENT_ROLLING_RISK")
                    returns = window.pct_change(fill_method=None).iloc[1:]
                    scores = score_frame.set_index("instrument").score
                    size_rows = basic.loc[basic.trade_date.astype(str).between(window.index[-10], date)].sort_values("trade_date").groupby("ts_code").tail(1)
                    size = size_rows.set_index("ts_code").total_mv
                    eligible = [s for s in scores.index if s in returns and np.isfinite(returns[s]).all() and size.get(s, 0) > 0]
                    if set(held) - set(eligible):
                        missing = sorted(set(held) - set(eligible))
                        audit.append(dict(date=date, status="BLOCKED_HELD_RISK", symbols=missing,
                            missing_return_dates={s: returns[s].index[returns[s].isna()].tolist() if s in returns else ["NO_HISTORY"] for s in missing}))
                        raise ValueError("HELD_RISK_MISSING:" + date + ":" + ",".join(missing))
                    risk = returns[eligible]
                    current_close = daily.loc[daily.trade_date.eq(date)].set_index("ts_code").close
                    w0 = pd.Series({s: q * current_close[s] for s, q in held.items()})
                    if not w0.empty:
                        w0 /= w0.sum()
                    proxy = risk.mean(axis=1)
                    styles = pd.DataFrame(dict(momentum_60=(1 + risk.tail(60)).prod() - 1,
                        volatility_20=risk.tail(20).std(), market_beta_120=risk.apply(lambda x: x.cov(proxy) / proxy.var()),
                        log_size=np.log(size.reindex(eligible))))
                    retention = .75
                    if variant == "style_feasible" and held:
                        phase_one = attainable_style_retention(styles, w0)
                        retention = phase_one["chosen_retention"]
                        audit.append(dict(date=date, stage="STYLE_FEASIBILITY", **phase_one))
                    limits = {c: 1e6 for c in styles.columns}
                    if held:
                        solved = optimize_snapshot(scores, risk, size, w0, EnhancedPolicy(lamb=10.),
                            style_controls=styles if variant in ("style_retain_75pct", "style_feasible") else None,
                            style_retention=retention, solver_max_iters=1000 if variant == "style_feasible" else 200)
                        audit.append(dict(date=date, risk_end=window.index[-1], source_sha256=report["source_sha256"],
                                          **{k: v for k, v in solved.items() if k not in ("weights", "benchmark_weights")}))
                        if solved["status"] != "PASS_OFFLINE_ONLY":
                            raise ValueError("OPTIMIZATION_BLOCKED:" + date + ":" + ",".join(solved["failures"]))
                        targets = solved["weights"]
                        if variant == "style_feasible":
                            limits = {k: abs(v["before"]) * retention for k, v in solved["named_style_active_exposures"].items()}
                    if integer:
                        if set(s for s, w in targets.items() if w > 1e-6) - set(styles.index):
                            raise ValueError("INITIAL_TARGET_RISK_MISSING:" + date)
                        context.update(date=date, targets=targets, styles=styles.copy(), limits=limits,
                                       source_sha256=report["source_sha256"])
                # Explicit zero targets prevent stale target-state carryover.
                return pd.DataFrame([dict(date=pd.Timestamp(date), symbol=s, target_weight=targets.get(s, 0.)) for s in pool])

            name = account + "-" + variant
            try:
                replay = run_execution_replay(bars=bars, targets=EMPTY, config=config, database=output_dir / (name + ".sqlite3"),
                    strategy_id=name, account_id=name, closing_target_provider=provider,
                    corporate_actions=replay_actions, accrue_action_receivables=True,
                    opening_plan_provider=opening if integer else None)
                replay.equity.to_parquet(output_dir / (name + "-equity.parquet"), index=False)
                pd.DataFrame(replay.fills).to_parquet(output_dir / (name + "-fills.parquet"), index=False)
                row = dict(account=account, variant=variant, status="COMPLETED_DIAGNOSTIC_ONLY", summary=replay.summary,
                           reconciliation=replay.reconciliation)
                if integer:
                    events = replay.summary["opening_plan_audits"]
                    blocked = [e for e in events if e["plan"]["status"] not in ("PASS_REFERENCE_QUOTES_ONLY", "NO_PRIOR_SIGNAL")]
                    row["opening_execution_review"] = dict(
                        planned_opens=sum(e["plan"]["status"] == "PASS_REFERENCE_QUOTES_ONLY" for e in events),
                        blocked_opens=len(blocked), blocked_events=blocked,
                        qualification="PAUSED_DAYS_NOT_FULL_EXECUTION" if blocked else "ALL_PLANNED_OPENS_MATCHED",
                        initial_policy="same_saved_target_integer_bootstrap_without_style_reduction",
                        style_basis="prior_close_frozen_full_risk_universe",
                        turnover_basis="current_open_stock_sleeve")
            except ValueError as error:
                row = dict(account=account, variant=variant, status="BLOCKED_NO_PERFORMANCE_CLAIM", reason=str(error))
            row["daily_optimizer_audit"] = audit
            result["runs"].append(row)
            write_diagnostic(output_dir / "progress.json", result)
            print(name, row["status"], row.get("reason", ""), flush=True)
    result["protected_equity_tables_unchanged"] = hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    prefix = "integer-continuous-replay" if integer else "continuous-replay"
    result["integer_opening_planning"] = integer
    report_name = f"{prefix}-{account_filter}.json" if account_filter else prefix + ".json"
    write_diagnostic(folder / report_name, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--account")
    parser.add_argument("--integer", action="store_true")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.account, args.integer)
