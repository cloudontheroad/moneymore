"""Counterfactual integer planning at known signal-close quotes, no execution."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import prediction_reports, trading_dates, write_diagnostic
from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.research.enhanced_indexing import (
    EnhancedPolicy,
    carry_confirmed_suspensions,
    optimize_snapshot,
)
from moneymore.research.integer_execution import plan_integer_execution


def run(root, account_filter=None, date_filter=None):
    folder = root / "state/enhanced-indexing-research"
    replay = json.loads((folder / "continuous-replay.json").read_text(encoding="utf-8"))
    actual = json.loads((folder / "actual-fill-attribution.json").read_text(encoding="utf-8"))
    preflight = json.loads((folder / "replay-preflight.json").read_text(encoding="utf-8"))
    if actual["replay_directory"] != replay["output_directory"]:
        raise ValueError("ATTRIBUTION_REPLAY_SOURCE_MISMATCH")
    store = ParquetStore(root / "data")
    calendar = trading_dates(store, replay["as_of_date"])
    first = calendar[calendar.index(replay["start_date"]) - 120]
    dates = [d for d in calendar if first <= d <= replay["as_of_date"]]
    filters = [("trade_date", ">=", first), ("trade_date", "<=", replay["as_of_date"])]
    daily = store.read("daily", filters=filters)
    adj = store.read("adj_factor", filters=filters)
    basic = store.read("daily_basic", filters=filters)
    close = daily.pivot(index="trade_date", columns="ts_code", values="close").reindex(dates)
    prices = close * adj.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex(
        dates
    )
    evidence = Path(preflight["evidence_directory"])
    proof = pd.concat(
        [
            pd.read_parquet(evidence / f"suspend_d-{d}.parquet")
            for d in preflight["suspension_response_dates"]
        ]
    )
    proof = pd.concat([proof, store.read("enhanced_indexing_suspension_proof")], ignore_index=True)
    risk_audit = json.loads((folder / "risk-history-audit.json").read_text(encoding="utf-8"))
    for item in risk_audit["symbols"]:
        source = Path(item["suspension_source_file"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["suspension_source_sha256"]:
            raise ValueError("SUSPENSION_PROOF_HASH_MISMATCH")
        proof = pd.concat([proof, pd.read_parquet(source)], ignore_index=True)
    proof["verified_full_session"] = proof.suspend_type.eq("S") & (
        proof.suspend_timing.isna() | proof.suspend_timing.eq("")
    )
    prices, _ = carry_confirmed_suspensions(prices, proof)
    protected = [
        store.curated / (n + ".parquet")
        for n in (
            "multi_sector_account_daily",
            "qlib_challenger_account_daily",
            "qlib_candidate_account_daily",
        )
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    config = BacktestConfig.model_validate(replay["config"])
    output = {
        "as_of_date": replay["as_of_date"],
        "mode": "COUNTERFACTUAL_SIGNAL_CLOSE_QUOTES_ONLY",
        "orders_enabled": False,
        "future_prices_used": False,
        "cases": [],
        "warning": "Independent one-step plans from historical actual positions, not a continuous integer-policy backtest. Open-price gaps, next-day limits and settlement require new runtime validation.",
        "availability_policy": "All current physical A-share quantities assumed T+1-available at NEXT open, not available for same-day selling.",
    }
    for trial in replay["runs"]:
        if trial["variant"] != "style_feasible" or trial["status"] != "COMPLETED_DIAGNOSTIC_ONLY":
            continue
        account = trial["account"]
        if account_filter and account != account_filter:
            continue
        holdings = next(
            r["holdings_daily"]
            for r in actual["runs"]
            if r["account"] == account and r["variant"] == "style_feasible"
        )
        equities = pd.read_parquet(
            Path(replay["output_directory"]) / (account + "-style_feasible-equity.parquet")
        ).set_index("trade_date")
        reports = {
            r["trade_date"]: r
            for r in prediction_reports(
                root / "state/qlib-candidate-shadow" / account.removeprefix("candidate_"),
                replay["as_of_date"],
            )
        }
        for event in trial["daily_optimizer_audit"]:
            if event.get("status") != "PASS_OFFLINE_ONLY":
                continue
            date = event["date"]
            if date_filter and date != date_filter:
                continue
            report = reports[date]
            if report["source_sha256"] != event["source_sha256"]:
                raise ValueError("RECORDED_PREDICTION_HASH_MISMATCH")
            if equities.loc[date, "unsettled_stock_value"] != 0:
                output["cases"].append(
                    {"account": account, "date": date, "status": "BLOCKED_UNSETTLED_STOCK_BASIS"}
                )
                continue
            scores = pd.DataFrame(report["scores"]).set_index("instrument").score
            window = prices.loc[:date].tail(121)
            returns = window.pct_change(fill_method=None).iloc[1:]
            latest_size = (
                basic.loc[basic.trade_date.astype(str).between(window.index[-10], date)]
                .sort_values("trade_date")
                .groupby("ts_code")
                .tail(1)
                .set_index("ts_code")
                .total_mv
            )
            eligible = [
                s
                for s in scores.index
                if s in returns and np.isfinite(returns[s]).all() and latest_size.get(s, 0) > 0
            ]
            risk = returns[eligible]
            proxy = risk.mean(axis=1)
            styles = pd.DataFrame(
                {
                    "log_size": np.log(latest_size.reindex(eligible)),
                    "momentum_60": (1 + risk.tail(60)).prod() - 1,
                    "volatility_20": risk.tail(20).std(),
                    "market_beta_120": risk.apply(
                        lambda x, benchmark=proxy: x.cov(benchmark) / benchmark.var()
                    ),
                }
            )
            # Preserve the optimizer's original column order, not a new risk model.
            styles = styles[["momentum_60", "volatility_20", "market_beta_120", "log_size"]]
            quantity = pd.Series(holdings[date], dtype=float)
            current = quantity * close.loc[date].reindex(quantity.index)
            current /= current.sum()
            solved = optimize_snapshot(
                scores,
                risk,
                latest_size,
                current,
                EnhancedPolicy(lamb=10.0),
                style_controls=styles,
                style_retention=event["named_style_retention"],
                solver_max_iters=1000,
            )
            if solved["status"] != "PASS_OFFLINE_ONLY":
                output["cases"].append(
                    {
                        "account": account,
                        "date": date,
                        "status": "BLOCKED_REPRODUCED_CONTINUOUS_SOLVE",
                        "failures": solved["failures"],
                    }
                )
                continue
            limits = pd.Series(
                {
                    k: abs(v["before"]) * event["named_style_retention"]
                    for k, v in event["named_style_active_exposures"].items()
                }
            )
            planned = plan_integer_execution(
                pd.Series(solved["weights"]),
                close.loc[date],
                quantity,
                quantity,
                float(equities.loc[date, "cash"]),
                float(equities.loc[date, "equity"]),
                config,
                styles,
                limits,
            )
            output["cases"].append(
                dict(
                    account=account,
                    date=date,
                    source_sha256=report["source_sha256"],
                    style_limits=limits.to_dict(),
                    **planned,
                )
            )
            print(account, date, planned["status"], planned.get("reason", ""), flush=True)
    output["protected_equity_tables_unchanged"] = before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected
    }
    suffix = (
        ("-" + (account_filter or "all") + "-" + (date_filter or "all"))
        if account_filter or date_filter
        else ""
    )
    write_diagnostic(folder / ("integer-plans-audit" + suffix + ".json"), output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--account")
    parser.add_argument("--date")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.account, args.date)
