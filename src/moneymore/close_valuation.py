"""Official-close valuation recovery, independent of model/strategy success.

Only missing daily NAVs (and provisional intraday marks) are written. Orders,
fills, positions and account cash are never mutated by this module.
"""
from pathlib import Path
import sqlite3
import math

import pandas as pd

from .data.store import ParquetStore


def snapshot_as_of(connection, account, day):
    cash = float(connection.execute("SELECT cash FROM accounts WHERE account_id=?", (account,)).fetchone()[0])
    quantities = dict(connection.execute("SELECT symbol,quantity FROM positions WHERE account_id=?", (account,)))
    # Reverse later events, never apply today's position to yesterday's close.
    for symbol, side, qty, price, fee in connection.execute(
        "SELECT symbol,side,quantity,price,fee FROM fills WHERE account_id=? AND trade_date>?", (account, day)
    ):
        sign = 1 if side == "BUY" else -1
        quantities[symbol] = quantities.get(symbol, 0) - sign * qty
        cash += sign * qty * price + fee
    for symbol, amount, shares in connection.execute(
        "SELECT symbol,cash_amount,share_quantity FROM corporate_action_ledger WHERE account_id=? AND trade_date>?", (account, day)
    ):
        cash -= amount
        quantities[symbol] = quantities.get(symbol, 0) - shares
    if any(q < 0 for q in quantities.values()):
        raise ValueError("NEGATIVE_ASOF_POSITION")
    return cash, {s: q for s, q in quantities.items() if q}


def recover_close_valuations(root: Path, cutoff: str, store=None):
    store = store or ParquetStore(root / "data")
    calendar = store.read("trade_calendar", columns=["cal_date", "is_open"])
    days = sorted(set(calendar.loc[calendar.is_open.astype(str).eq("1") & calendar.cal_date.astype(str).le(cutoff), "cal_date"].astype(str)))
    jobs = []
    for path in sorted(store.curated.glob("*account_daily.parquet")):
        history = pd.read_parquet(path)
        if history.empty:
            continue
        if "account_id" not in history:
            # Legacy baseline tables store a single account without an ID.
            account = path.stem.replace("_account_daily", "") + "_shadow"
            history = history.assign(account_id=account)
        for account, frame in history.groupby("account_id"):
            last = str(frame.trade_date.astype(str).max())
            pending = [d for d in days if last < d <= cutoff]
            # Intraday writers store provisional daily rows; finalize them too.
            if path.stem in {"baseline_intraday_account_daily", "baseline_pysystemtrade_intraday_account_daily"} and "observed_at" in frame:
                latest = frame.loc[frame.trade_date.astype(str).eq(last)].iloc[-1]
                if pd.notna(latest.get("observed_at")) and str(latest.get("valuation_source", "")) != "OFFICIAL_CLOSE_LEDGER_ONLY" and str(latest.get("observed_at", ""))[11:16] < "15:00":
                    pending = [last, *pending]
            for day in pending[:20]:
                jobs.append((path.stem, str(account), day, history))
    if not jobs:
        return {"status": "CURRENT", "updated": [], "failures": [], "orders_written": 0}
    prices = store.read("daily", filters=[("trade_date", ">=", min(j[2] for j in jobs)), ("trade_date", "<=", cutoff)])
    prices["trade_date"] = prices.trade_date.astype(str)
    marks = {(str(r.ts_code), str(r.trade_date)): float(r.close) for r in prices.itertuples()}
    # Only formally documented suspension permits carrying an earlier mark.
    suspended = set()
    for path in (root / "state/mature-multifactor/runtime/official").glob("*/suspend_d.parquet"):
        if (path.parent / "results.json").exists():
            f = pd.read_parquet(path)
            if not f.empty:
                suspended.update(zip(f.ts_code.astype(str), f.trade_date.astype(str)))
    updated, failures, repaired_equities = [], [], {}
    with sqlite3.connect(f"file:{(root / 'state/paper_orders.sqlite3').as_posix()}?mode=ro", uri=True) as c:
        c.execute("BEGIN")
        for table, account, day, history in jobs:
            try:
                cash, quantities = snapshot_as_of(c, account, day)
                market_value = 0.0
                for symbol, qty in quantities.items():
                    price = marks.get((symbol, day))
                    if price is None and (symbol, day) in suspended:
                        prior = store.read("daily", filters=[("ts_code", "==", symbol), ("trade_date", "<", day)])
                        if not prior.empty:
                            price = float(prior.sort_values("trade_date").iloc[-1].close)
                    if price is None or not math.isfinite(price) or not (price > 0):
                        raise ValueError("OFFICIAL_HELD_CLOSE_MISSING:" + symbol)
                    market_value += qty * price
                # Unsettled economic rights must not silently vanish from NAV.
                pending_rights = c.execute(
                    "SELECT action_key FROM corporate_entitlements WHERE account_id=? AND record_date<=? AND (cash_status='PENDING' OR stock_status='PENDING')",
                    (account, day),
                ).fetchall()
                if pending_rights:
                    raise ValueError("UNSETTLED_RIGHTS_REQUIRE_ECONOMIC_VALUATION")
                equity = cash + market_value
                prior = history.loc[history.account_id.eq(account) & history.trade_date.astype(str).lt(day)].sort_values("trade_date")
                repaired = [{"trade_date": d, "equity": value} for (t, a, d), value in repaired_equities.items()
                    if t == table and a == account and d < day]
                if repaired:
                    prior = pd.concat([prior[["trade_date", "equity"]], pd.DataFrame(repaired)], ignore_index=True).drop_duplicates("trade_date", keep="last").sort_values("trade_date")
                row = dict(trade_date=day, account_id=account, cash=cash, market_value=market_value, equity=equity,
                    gross_exposure=market_value/equity if equity else 0.0,
                    status="VALUATION_ONLY", valuation_source="OFFICIAL_CLOSE_LEDGER_ONLY",
                    observed_at=f"{day[:4]}-{day[4:6]}-{day[6:]}T15:00:00+08:00")
                if not prior.empty:
                    row["daily_return"] = equity / float(prior.iloc[-1].equity) - 1
                    row["drawdown"] = equity / max(equity, float(prior.equity.max())) - 1
                keys = ["trade_date", "account_id"] if "account_id" in pd.read_parquet(store.curated / (table+".parquet")).columns else ["trade_date"]
                store.merge_curated(table, [pd.DataFrame([row])], keys)
                # Subsequent repaired dates must use the repaired prior close.
                repaired_equities[(table, account, day)] = equity
                updated.append({"account": account, "date": day, "equity": equity})
            except (ValueError, TypeError) as error:
                failures.append({"account": account, "date": day, "error": str(error)})
    return {"status": "PARTIAL" if failures else "COMPLETED", "updated": updated, "failures": failures, "orders_written": 0}
