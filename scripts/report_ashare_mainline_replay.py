"""Read-only ledger verification and same-contract N3 results; no live accounts."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.research.ashare_mainline import sha256
from moneymore.research.period_statistics import period_statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "state/ashare-mainline-20261112"


def audit_ledger(path, exported):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        accounts = pd.read_sql_query("SELECT * FROM accounts", db)
        if accounts.account_id.tolist() != ["isolated"]:
            raise ValueError("NOT_AN_ISOLATED_ACCOUNT")
        fills = pd.read_sql_query("SELECT * FROM fills ORDER BY id", db)
        actions = pd.read_sql_query("SELECT * FROM corporate_action_ledger", db)
        positions = pd.read_sql_query("SELECT * FROM positions", db)
        orders = pd.read_sql_query("SELECT * FROM orders", db)
    if (
        len(fills) != len(exported)
        or len(fills) != len(orders)
        or not orders.status.eq("FILLED").all()
    ):
        raise ValueError("FILL_ORDER_EXPORT_COUNT_MISMATCH")
    if set(fills.idempotency_key) != set(orders.idempotency_key):
        raise ValueError("FILL_ORDER_ID_MISMATCH")
    for column in ("quantity", "price", "fee"):
        if not np.allclose(fills[column], exported[column], atol=1e-8, rtol=1e-10):
            raise ValueError("FILL_EXPORT_VALUE_MISMATCH:" + column)
    for column, export_column in (("symbol", "symbol"), ("side", "side"), ("trade_date", "date")):
        if fills[column].tolist() != exported[export_column].tolist():
            raise ValueError("FILL_EXPORT_KEY_MISMATCH:" + column)
    signed = fills.quantity * np.where(fills.side.eq("BUY"), 1, -1)
    expected_cash = float(
        accounts.initial_cash.iloc[0]
        - (signed * fills.price).sum()
        - fills.fee.sum()
        + actions.cash_amount.sum()
    )
    expected_shares = (
        signed.groupby(fills.symbol)
        .sum()
        .add(actions.groupby("symbol").share_quantity.sum(), fill_value=0)
    )
    actual = positions.set_index("symbol").quantity
    delta = expected_shares.subtract(actual, fill_value=0)
    cash_delta = expected_cash - float(accounts.cash.iloc[0])
    if abs(cash_delta) > 0.01 or delta.abs().sum() > 0 or accounts.cash.iloc[0] < -0.01:
        raise ValueError("INDEPENDENT_LEDGER_RECONCILIATION_FAILED")
    return {"matched": True, "cash_difference": cash_delta, "symbols": int(actual.gt(0).sum())}


def main(runs, output):
    if output.exists():
        raise FileExistsError("IMMUTABLE_REPORT")
    attrs = pd.read_parquet(BASE / "panel-v2/panel.parquet", columns=["date", "symbol", "industry"])
    attrs = {day: g for day, g in attrs.groupby("date")}
    all_rows, manifests, modes, windows = [], {}, set(), set()
    contract_hash = None
    for run in runs:
        contract = json.loads((run / "contract.json").read_bytes())
        current_hash = sha256(run / "contract.json")
        if contract_hash is not None and current_hash != contract_hash:
            raise ValueError("MIXED_EXPERIMENT_CONTRACTS")
        contract_hash = current_hash
        result_files = list(run.glob("*/results.json"))
        if len(result_files) != 1:
            raise ValueError("INCOMPLETE_OR_AMBIGUOUS_RUN")
        result_file = result_files[0]
        folder = result_file.parent
        result = json.loads(result_file.read_bytes())
        protocol = json.loads((folder / "protocol.json").read_bytes())
        modes.add(bool(result["smoke"]))
        windows.add(tuple(protocol["window"]))
        cases = {r["strategy"]: r for r in result["cases"]}
        expected = {m + "--" + a for m in contract["models"] for a in contract["allocators"]}
        if set(cases) != expected or not result["all_daily_reconciliations_pass"]:
            raise ValueError("INCOMPLETE_MATRIX_OR_RECONCILIATION")
        holdings = pd.read_parquet(folder / "holdings.parquet")
        known, style_rows = {}, []
        dates = iter(sorted(attrs))
        next_date = next(dates, None)
        for day, h in holdings.groupby("date", sort=True):
            while next_date is not None and next_date <= day:
                known.update(attrs[next_date].set_index("symbol").industry.dropna().to_dict())
                next_date = next(dates, None)
            h = h.copy()
            h["industry"] = h.symbol.map(known).fillna("UNKNOWN_HISTORICAL_INDUSTRY")
            style_rows.append(h)
        styles = pd.concat(style_rows, ignore_index=True)
        for case in sorted(cases):
            equity = pd.read_parquet(folder / case / "equity.parquet")
            fills = pd.read_parquet(folder / case / "fills.parquet")
            if equity.date.duplicated().any() or not equity.date.is_monotonic_increasing:
                raise ValueError("INVALID_EQUITY_CALENDAR")
            required = 3 if result["smoke"] else 181
            if (
                len(equity) != required
                or [equity.date.iloc[0], equity.date.iloc[-1]] != protocol["window"]
            ):
                raise ValueError("INCOMPLETE_EQUITY_WINDOW")
            ledger = audit_ledger(folder / case / "paper.sqlite3", fills)
            stats = period_statistics(equity.equity, contract["initial_cash"])
            for name in ("net_return", "sharpe", "maximum_drawdown", "annualized_volatility"):
                if not np.isclose(stats[name], cases[case][name], atol=1e-10):
                    raise ValueError("RETURN_METRICS_MISMATCH:" + name)
            daily = equity.set_index("date")
            previous_nav = daily.equity.shift(1).fillna(contract["initial_cash"])
            turnover = (
                fills.groupby("date").notional.sum().reindex(daily.index, fill_value=0)
                / previous_nav
                / 2
            )
            h = styles.loc[styles.kind.eq(case)]
            industry = h.groupby(["date", "industry"]).weight.sum()
            maxima = industry.groupby(level="date").max().reindex(daily.index, fill_value=0)
            decisions = [
                json.loads(p.read_bytes())
                for p in sorted((run / "decisions" / case).glob("*.json"))
            ]
            if len(decisions) != required:
                raise ValueError("INCOMPLETE_DECISION_AUDIT")
            all_rows.append(
                {
                    "case": case,
                    "bps": protocol["slippage_bps"],
                    **stats,
                    "mean_actual_stock_exposure": float(daily.exposure.mean()),
                    "minimum_actual_stock_exposure": float(daily.exposure.min()),
                    "maximum_actual_stock_exposure": float(daily.exposure.max()),
                    "total_half_gross_notional_over_previous_nav": float(turnover.sum()),
                    "annualized_half_gross_turnover": float(turnover.sum() * 252 / len(daily)),
                    "fees": float(fills.fee.sum()),
                    "slippage": float(fills.slippage.sum()),
                    "fills": len(fills),
                    "mean_growth_board_weight": float(
                        h.loc[h.symbol.str.startswith(("30", "68"))].weight.sum() / len(daily)
                    ),
                    "mean_largest_industry_weight": float(maxima.mean()),
                    "maximum_industry_weight": float(maxima.max()),
                    "industry_mean_weights": (
                        h.groupby("industry").weight.sum() / len(daily)
                    ).to_dict(),
                    "decision_reasons": dict(
                        Counter(d["reason"] for d in decisions if d.get("reason"))
                    ),
                    "ledger": ledger,
                    "quarters": cases[case]["quarters"],
                }
            )
        manifests[str(run)] = {
            "result_sha256": sha256(result_file),
            "holdings_sha256": sha256(folder / "holdings.parquet"),
        }
    if len(modes) != 1 or len(windows) != 1:
        raise ValueError("MIXED_SMOKE_OR_WINDOW")
    rows = pd.DataFrame(all_rows)
    if rows.duplicated(["case", "bps"]).any():
        raise ValueError("DUPLICATE_COST_CASE")
    output.mkdir(parents=True)
    for column in ("industry_mean_weights", "decision_reasons", "ledger", "quarters"):
        rows[column] = rows[column].map(json.dumps)
    rows.to_parquet(output / "metrics.parquet", index=False)
    payload = {
        "status": "SMOKE_LEDGER_VERIFIED" if True in modes else "FULL_REPLAY_LEDGER_VERIFIED",
        "cases": all_rows,
        "sources": manifests,
        "contract_sha256": contract_hash,
        "smoke_metrics_are_not_performance_evidence": True in modes,
        "actual_exposure_must_be_compared_not_assumed_equal": True,
        "production_modified": False,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps({"status": payload["status"], "cases": len(rows), "report": str(output)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.runs, args.output)
