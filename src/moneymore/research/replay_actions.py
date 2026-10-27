"""Opt-in corporate-action accounting for isolated historical replay only."""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path

import numpy as np
import pandas as pd


def official_replay_actions(raw: pd.DataFrame) -> pd.DataFrame:
    """Only implemented, unambiguous events; stk_div already includes transfers."""
    columns = ["symbol", "action_key", "record_date", "ex_date", "cash_per_share",
               "stock_ratio", "cash_pay_date", "stock_list_date", "known_date"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    implemented = raw.loc[raw.div_proc.eq("实施")].copy()
    for event in implemented.to_dict("records"):
        cash = float(event["cash_div_tax"] or 0) if pd.notna(event["cash_div_tax"]) else 0.
        total = event.get("stk_div")
        parts = [event.get("stk_bo_rate"), event.get("stk_co_rate")]
        component_sum = sum(float(v) for v in parts if pd.notna(v))
        stock = float(total) if pd.notna(total) else component_sum
        if any(pd.notna(v) for v in parts) and pd.notna(total) and not np.isclose(stock, component_sum, atol=1e-8):
            raise ValueError("CONFLICTING_STOCK_DIVIDEND_COMPONENTS")
        if cash < 0 or stock < 0 or not np.isfinite([cash, stock]).all():
            raise ValueError("INVALID_CORPORATE_ACTION_AMOUNT")
        if cash == 0 and stock == 0:
            continue
        record = pd.to_datetime(event["record_date"], errors="coerce")
        ex = pd.to_datetime(event["ex_date"], errors="coerce")
        pay = pd.to_datetime(event["pay_date"], errors="coerce")
        listing = pd.to_datetime(event["div_listdate"], errors="coerce")
        known = pd.to_datetime(event.get("imp_ann_date"), errors="coerce")
        if pd.isna(known):
            known = pd.to_datetime(event.get("ann_date"), errors="coerce")
        if pd.isna(record) or pd.isna(ex) or pd.isna(known) or not known <= record < ex:
            raise ValueError("INVALID_CORPORATE_ACTION_TIMELINE")
        if (cash > 0 and (pd.isna(pay) or pay < ex)) or (stock > 0 and (pd.isna(listing) or listing < ex)):
            raise ValueError("MISSING_OR_INVALID_ACTION_SETTLEMENT_DATE")
        rows.append(dict(symbol=event["ts_code"], action_key=f"{event['ts_code']}:{event['end_date']}:{record:%Y%m%d}",
                         record_date=record, ex_date=ex, cash_per_share=cash, stock_ratio=stock,
                         cash_pay_date=pay if cash else pd.NaT, stock_list_date=listing if stock else pd.NaT,
                         known_date=known))
    result = pd.DataFrame(rows, columns=columns).drop_duplicates()
    if result.action_key.duplicated().any():
        raise ValueError("CONFLICTING_IMPLEMENTED_ACTION_VERSIONS")
    return result


def economic_snapshot(broker, snapshot, marks, account_id, date, actions):
    """Accrue rights at ex-date, never treat them as spendable cash or shares.

    Settlement is exclusively performed by PaperBroker. Reading its status avoids
    double counting after pay/list dates, even after a sale of the original stock.
    """
    with sqlite3.connect(broker.database) as connection:
        connection.row_factory = sqlite3.Row
        entitlements = connection.execute(
            "SELECT * FROM corporate_entitlements WHERE account_id = ?", (account_id,)
        ).fetchall()
    by_key = actions.set_index("action_key") if not actions.empty else None
    cash, stock_value, stock_rights = 0., 0., {}
    for item in entitlements:
        if by_key is None or item["action_key"] not in by_key.index:
            raise ValueError("ENTITLEMENT_WITHOUT_EX_DATE")
        event = by_key.loc[item["action_key"]]
        if pd.Timestamp(date) < event.ex_date:
            continue
        if item["cash_status"] == "PENDING":
            cash += item["entitled_quantity"] * item["cash_per_share"]
        if item["stock_status"] == "PENDING":
            shares = int(item["entitled_quantity"] * item["stock_ratio"])
            symbol = item["symbol"]
            if symbol not in marks or not np.isfinite(marks[symbol]) or marks[symbol] <= 0:
                raise ValueError("MISSING_UNSETTLED_STOCK_RIGHTS_MARK")
            stock_rights[symbol] = stock_rights.get(symbol, 0) + shares
            stock_value += shares * marks[symbol]
    return {**snapshot, "receivable_cash": cash, "unsettled_stock_value": stock_value,
            "unsettled_stock_rights": stock_rights,
            "market_value": snapshot["market_value"] + stock_value,
            "equity": snapshot["equity"] + cash + stock_value}


def audit_replay_action_ledgers(root: Path):
    """Export evidence from completed research ledgers, never production databases."""
    from moneymore.challenger_diagnostics import write_diagnostic

    folder = root / "state/enhanced-indexing-research"
    result = json.loads((folder / "continuous-replay.json").read_text(encoding="utf-8"))
    records = []
    directory = Path(result["output_directory"])
    for run in result["runs"]:
        if run["status"] != "COMPLETED_DIAGNOSTIC_ONLY":
            continue
        name = run["account"] + "-" + run["variant"]
        with sqlite3.connect(directory / (name + ".sqlite3")) as connection:
            ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", connection)
            entitlements = pd.read_sql_query("SELECT * FROM corporate_entitlements", connection)
        equity = pd.read_parquet(directory / (name + "-equity.parquet"))
        ledger.to_parquet(directory / (name + "-corporate-ledger.parquet"), index=False)
        entitlements.to_parquet(directory / (name + "-entitlements.parquet"), index=False)
        records.append(dict(account=run["account"], variant=run["variant"],
            entitlements=len(entitlements), ledger_entries=len(ledger),
            cash_paid=float(ledger.cash_amount.sum()) if len(ledger) else 0.,
            shares_delivered=int(ledger.share_quantity.sum()) if len(ledger) else 0,
            peak_cash_receivable=float(equity.receivable_cash.max()),
            peak_stock_receivable=float(equity.unsettled_stock_value.max()),
            reconciled=run["reconciliation"]["matched"]))
    output = dict(as_of_date=result["as_of_date"], output_directory=str(directory),
                  orders_enabled=False, runs=records,
                  protected_equity_tables_unchanged=result["protected_equity_tables_unchanged"])
    write_diagnostic(folder / "corporate-action-replay-summary.json", output)
    return output
