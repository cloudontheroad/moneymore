"""Read-only actual-fill style attribution for all completed research branches."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_attribution import linked_attribution, matched_day
from moneymore.challenger_diagnostics import prediction_reports, trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.enhanced_indexing import carry_confirmed_suspensions
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.replay_attribution import (
    economic_daily_return,
    exposure_snapshot,
    reconstruct_cash,
    reconstruct_positions,
)


def run(root, integer=False, fixed_budget=False):
    folder = root / "state/enhanced-indexing-research"
    integer = integer or fixed_budget
    source_name = (
        "fixed-budget-continuous-replay.json"
        if fixed_budget
        else "integer-continuous-replay.json"
        if integer
        else "continuous-replay.json"
    )
    style_variant = "style_fixed_budget" if fixed_budget else "style_feasible"
    replay = json.loads((folder / source_name).read_text(encoding="utf-8"))
    preflight = json.loads((folder / "replay-preflight.json").read_text(encoding="utf-8"))
    action_audit = json.loads((folder / "corporate-action-audit.json").read_text(encoding="utf-8"))
    directory = Path(replay["output_directory"])
    store = ParquetStore(root / "data")
    protected = [
        store.curated / (name + ".parquet")
        for name in (
            "multi_sector_account_daily",
            "qlib_challenger_account_daily",
            "qlib_candidate_account_daily",
        )
    ]
    protected_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    calendar = trading_dates(store, replay["as_of_date"])
    dates = [d for d in calendar if replay["start_date"] <= d <= replay["as_of_date"]]
    first = calendar[calendar.index(dates[0]) - 120]
    filters = [("trade_date", ">=", first), ("trade_date", "<=", dates[-1])]
    daily = store.read("daily", filters=filters)
    adj = store.read("adj_factor", filters=filters)
    basic = store.read("daily_basic", filters=filters)
    all_dates = [d for d in calendar if first <= d <= dates[-1]]

    def pivot(frame, value):
        return frame.pivot(index="trade_date", columns="ts_code", values=value).reindex(all_dates)

    raw_close = pivot(daily, "close")
    adjusted = raw_close * pivot(adj, "adj_factor")
    evidence = Path(preflight["evidence_directory"])
    proof = pd.concat([pd.read_parquet(evidence / f"suspend_d-{d}.parquet") for d in dates])
    proof = pd.concat([proof, store.read("enhanced_indexing_suspension_proof")], ignore_index=True)
    risk_audit = json.loads((folder / "risk-history-audit.json").read_text(encoding="utf-8"))
    for item in risk_audit["symbols"]:
        if "suspension_source_file" in item:
            proof = pd.concat(
                [proof, pd.read_parquet(item["suspension_source_file"])], ignore_index=True
            )
    proof["verified_full_session"] = proof.suspend_type.eq("S") & (
        proof.suspend_timing.isna() | proof.suspend_timing.eq("")
    )
    adjusted, _ = carry_confirmed_suspensions(adjusted, proof)
    raw_close, _ = carry_confirmed_suspensions(raw_close, proof)
    returns = adjusted.pct_change(fill_method=None)
    size = np.log(pivot(basic, "total_mv").where(lambda x: x > 0).ffill(limit=9))
    actions = pd.read_parquet(action_audit["action_file"])
    actions = official_replay_actions(actions.loc[~actions.ts_code.str.startswith(("200", "900"))])
    output = {
        "as_of_date": dates[-1],
        "start_date": dates[0],
        "orders_enabled": False,
        "replay_directory": str(directory),
        "runs": [],
        "comparisons": [],
        "warning": "Lagged peer matching, not causal factor attribution. Primary 5 peers, sensitivity 3/10; no parameter selection.",
        "label": "OFFICIAL_EX_DATE_ECONOMIC_RETURN_PROXY_NOT_VENDOR_ADJUSTMENT_JUMPS",
    }
    feature_cache = {}
    for trial in replay["runs"]:
        if trial["status"] != "COMPLETED_DIAGNOSTIC_ONLY":
            output["runs"].append(
                {
                    "account": trial["account"],
                    "variant": trial["variant"],
                    "status": "REPLAY_NOT_COMPLETED",
                }
            )
            continue
        account, variant = trial["account"], trial["variant"]
        name = account + "-" + variant
        with sqlite3.connect(
            f"file:{(directory / (name + '.sqlite3')).as_posix()}?mode=ro", uri=True
        ) as connection:
            fills = pd.read_sql_query("SELECT * FROM fills", connection)
            ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", connection)
            entitlements = pd.read_sql_query("SELECT * FROM corporate_entitlements", connection)
            final = (
                pd.read_sql_query(
                    "SELECT symbol,quantity FROM positions WHERE quantity>0", connection
                )
                .set_index("symbol")
                .quantity.to_dict()
            )
        holdings = reconstruct_positions(fills, ledger, dates)
        cash_snapshots = reconstruct_cash(fills, ledger, dates, replay["config"]["initial_cash"])
        if holdings[dates[-1]] != final:
            raise ValueError("FILL_POSITION_RECONSTRUCTION_MISMATCH:" + name)
        equities = pd.read_parquet(directory / (name + "-equity.parquet")).set_index("trade_date")
        ledger_errors = []
        for date in dates:
            pending_cash, pending_stock = 0.0, 0.0
            for item in entitlements.itertuples():
                action = actions.loc[actions.action_key.eq(item.action_key)].iloc[0]
                if pd.Timestamp(date) < action.ex_date:
                    continue
                settled = ledger.loc[
                    ledger.action_key.eq(item.action_key) & ledger.trade_date.astype(str).le(date)
                ]
                if action.cash_per_share > 0 and not settled.action_type.eq("CASH_DIVIDEND").any():
                    pending_cash += item.entitled_quantity * action.cash_per_share
                if action.stock_ratio > 0 and not settled.action_type.eq("STOCK_DIVIDEND").any():
                    pending_stock += (
                        int(item.entitled_quantity * action.stock_ratio)
                        * raw_close.loc[date, item.symbol]
                    )
            physical = sum(q * raw_close.loc[date, s] for s, q in holdings[date].items())
            expected = cash_snapshots[date] + physical + pending_cash + pending_stock
            errors = {
                "date": date,
                "cash_error": cash_snapshots[date] - float(equities.loc[date, "cash"]),
                "equity_error": expected - float(equities.loc[date, "equity"]),
            }
            if (
                not np.isfinite([errors["cash_error"], errors["equity_error"]]).all()
                or max(abs(errors["cash_error"]), abs(errors["equity_error"])) > 0.01
            ):
                raise ValueError("DAILY_ECONOMIC_LEDGER_MISMATCH:" + name + ":" + date)
            ledger_errors.append(errors)
        report_dir = (
            root / "state/qlib-challenger-shadow"
            if account == "main_gru"
            else root / "state/qlib-candidate-shadow" / account.removeprefix("candidate_")
        )
        reports = {r["trade_date"]: r for r in prediction_reports(report_dir, dates[-1])}
        exposures, attribution = [], {n: [] for n in (3, 5, 10)}
        economic_holdings_by_date = {}
        for date in dates:
            economic_holdings = holdings[date].copy()
            for item in entitlements.itertuples():
                action = actions.loc[actions.action_key.eq(item.action_key)].iloc[0]
                if (
                    action.stock_ratio > 0
                    and action.ex_date <= pd.Timestamp(date) < action.stock_list_date
                ):
                    economic_holdings[item.symbol] = economic_holdings.get(item.symbol, 0) + int(
                        item.entitled_quantity * action.stock_ratio
                    )
            economic_holdings_by_date[date] = economic_holdings
            report = reports.get(date)
            if not report:  # Terminal missing report may occur for archived accounts.
                exposures.append({"date": date, "status": "MISSING_SAME_DATE_POOL"})
                continue
            pool = [s["instrument"] for s in report["scores"]]
            key = (date, tuple(sorted(pool)))
            if key not in feature_cache:
                window = returns.loc[:date].tail(120).reindex(columns=pool)
                eligible = window.columns[
                    window.notna().all() & size.loc[date].reindex(pool).notna()
                ]
                window = window[eligible]
                proxy = window.mean(axis=1)
                feature_cache[key] = pd.DataFrame(
                    {
                        "size": size.loc[date].reindex(eligible),
                        "momentum": (1 + window.tail(60)).prod(min_count=60) - 1,
                        "volatility": window.tail(20).std(),
                        "beta": window.apply(
                            lambda x, benchmark=proxy: (
                                x.cov(benchmark, min_periods=120) / benchmark.var()
                            )
                        ),
                    }
                )
            features = feature_cache[key]
            exposure = exposure_snapshot(
                economic_holdings,
                raw_close.loc[date],
                float(equities.loc[date, "equity"]),
                features,
                pool,
            )
            exposures.append(dict(date=date, **exposure))
            i = dates.index(date)
            if i == len(dates) - 1:
                continue
            following = dates[i + 1]
            ex_actions = actions.loc[actions.ex_date.eq(pd.Timestamp(following))]
            stock = ex_actions.groupby("symbol").stock_ratio.sum()
            cash = ex_actions.groupby("symbol").cash_per_share.sum()
            economic_return = economic_daily_return(
                raw_close.loc[date], raw_close.loc[following], stock, cash
            )
            current_equity = float(equities.loc[date, "equity"])
            actual = float(equities.loc[following, "equity"]) / current_equity - 1
            fee = float(fills.loc[fills.trade_date.eq(following), "fee"].sum())
            positions = [{"symbol": s, "quantity": q} for s, q in economic_holdings.items()]
            for peers, matched_rows in attribution.items():
                matched = matched_day(
                    positions,
                    current_equity,
                    actual,
                    fee,
                    raw_close.loc[date],
                    economic_return,
                    features,
                    pool,
                    neighbors=peers,
                )
                matched_rows.append(dict(prior_date=date, date=following, **matched))
        totals = {}
        for peers, rows in attribution.items():
            valid = [r for r in rows if r["status"] == "MATCHED"]
            totals[str(peers)] = {
                "matched_days": len(valid),
                "expected_days": len(dates) - 1,
                "status": "COMPLETE"
                if len(valid) == len(dates) - 1
                else "INCOMPLETE_NO_FULL_PERIOD_CLAIM",
                "linked": linked_attribution(valid) if len(valid) == len(dates) - 1 else None,
                "skipped": [
                    {"date": r["date"], "status": r["status"]}
                    for r in rows
                    if r["status"] != "MATCHED"
                ],
            }
        execution_bounds = []
        names = {
            "log_size": "size",
            "momentum_60": "momentum",
            "volatility_20": "volatility",
            "market_beta_120": "beta",
        }
        # Integer plans are independently checked at OPEN with frozen factors;
        # the old signal-close share projection is not their execution protocol.
        for event in [] if integer else trial.get("daily_optimizer_audit", []):
            if event.get("status") != "PASS_OFFLINE_ONLY" or not event.get(
                "named_style_active_exposures"
            ):
                continue
            date = event["date"]
            following = dates[dates.index(date) + 1]
            stock_ex = actions.loc[
                actions.ex_date.eq(pd.Timestamp(following)) & actions.stock_ratio.gt(0)
            ]
            if set(stock_ex.symbol) & set(economic_holdings_by_date[following]):
                execution_bounds.append(
                    {
                        "signal_date": date,
                        "execution_date": following,
                        "status": "STOCK_ACTION_FROZEN_SHARE_BASIS_REQUIRES_ADJUSTMENT",
                    }
                )
                continue
            pool = [s["instrument"] for s in reports[date]["scores"]]
            features = feature_cache[(date, tuple(sorted(pool)))]
            realized = exposure_snapshot(
                economic_holdings_by_date[following],
                raw_close.loc[date],
                float(equities.loc[following, "equity"]),
                features,
                pool,
            )
            if realized["status"] != "MEASURED":
                execution_bounds.append(
                    {"signal_date": date, "execution_date": following, "status": realized["status"]}
                )
                continue
            excess = {
                names[k]: max(
                    0.0,
                    abs(realized["active_stock_sleeve_exposures"][names[k]])
                    - abs(v["before"]) * event["named_style_retention"],
                )
                for k, v in event["named_style_active_exposures"].items()
            }
            execution_bounds.append(
                {
                    "signal_date": date,
                    "execution_date": following,
                    "status": "WITHIN_TOLERANCE"
                    if max(excess.values()) <= 1e-5
                    else "ACTUAL_FILL_STYLE_DEVIATION",
                    "excess_standardized_exposure": excess,
                    "basis": "Next closing actual share quantities marked with SIGNAL_DATE prices and factors; separates fill/rounding from market repricing.",
                }
            )
        output["runs"].append(
            {
                "account": account,
                "variant": variant,
                "status": "AUDITED",
                "reconstructed_positions_match": True,
                "economic_ledger_daily": ledger_errors,
                "holdings_daily": holdings,
                "exposure_daily": exposures,
                "execution_constraint_audit": execution_bounds,
                "attribution_sensitivity": totals,
                "primary_daily": attribution[5],
            }
        )
    for account in sorted({r["account"] for r in output["runs"]}):
        rows = {r["variant"]: r for r in output["runs"] if r["account"] == account}
        if all(rows.get(v, {}).get("status") == "AUDITED" for v in ("native_pca", style_variant)):
            a, b = rows["native_pca"], rows[style_variant]
            primary_a = a["attribution_sensitivity"]["5"]["linked"]
            primary_b = b["attribution_sensitivity"]["5"]["linked"]
            if primary_a and primary_b:
                output["comparisons"].append(
                    {
                        "account": account,
                        "linked_contribution_difference": {
                            k: primary_b[k] - primary_a[k] for k in primary_a
                        },
                        "actual_exposure_mean_abs": {
                            v: pd.DataFrame(
                                [
                                    d["active_stock_sleeve_exposures"]
                                    for d in rows[v]["exposure_daily"]
                                    if d["status"] == "MEASURED"
                                ]
                            )
                            .abs()
                            .mean()
                            .to_dict()
                            for v in ("native_pca", style_variant)
                        },
                        "interpretation": "Style minus native PCA; matching model is descriptive, not causal.",
                    }
                )
    output["protected_equity_tables_unchanged"] = protected_hashes == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected
    }
    write_diagnostic(
        folder
        / (
            "fixed-budget-actual-fill-attribution.json"
            if fixed_budget
            else "integer-actual-fill-attribution.json"
            if integer
            else "actual-fill-attribution.json"
        ),
        output,
    )
    print([(r["account"], r["linked_contribution_difference"]) for r in output["comparisons"]])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--integer", action="store_true")
    parser.add_argument("--fixed-budget", action="store_true")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.integer, args.fixed_budget)
