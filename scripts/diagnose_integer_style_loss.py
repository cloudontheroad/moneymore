"""Exact stock-level wealth difference; isolated, ex-post research only."""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.replay_attribution import reconstruct_positions, stock_wealth_change


def run(root, account="candidate_20261010"):
    folder = root / "state/enhanced-indexing-research"
    replay = json.loads((folder / "integer-continuous-replay.json").read_text(encoding="utf-8"))
    directory = Path(replay["output_directory"])
    store = ParquetStore(root / "data")
    daily = store.read(
        "daily",
        filters=[
            ("trade_date", ">=", replay["start_date"]),
            ("trade_date", "<=", replay["as_of_date"]),
        ],
    )
    closes = daily.pivot(index="trade_date", columns="ts_code", values="close")
    names = store.read("instruments").drop_duplicates("ts_code").set_index("ts_code")
    ca = json.loads((folder / "corporate-action-audit.json").read_text(encoding="utf-8"))
    raw_actions = pd.read_parquet(ca["action_file"])
    actions = official_replay_actions(
        raw_actions.loc[~raw_actions.ts_code.str.startswith(("200", "900"))]
    )
    initial = replay["config"]["initial_cash"]
    output = {
        "account": account,
        "source_directory": str(directory),
        "orders_enabled": False,
        "warning": "Exact realized wealth bookkeeping, not causal attribution or an executable alternative.",
        "variants": {},
    }
    for variant in ("native_pca", "style_feasible"):
        name = account + "-" + variant
        with sqlite3.connect(
            (directory / (name + ".sqlite3")).as_uri() + "?mode=ro", uri=True
        ) as conn:
            fills = pd.read_sql_query("SELECT * FROM fills", conn)
            ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", conn)
            entitlements = pd.read_sql_query("SELECT * FROM corporate_entitlements", conn)
        equity = pd.read_parquet(directory / (name + "-equity.parquet"))
        dates = equity.trade_date.astype(str).tolist()
        holdings = reconstruct_positions(fills, ledger, dates)
        prior_values, totals, days = {}, {}, []
        previous_equity = initial
        for date in dates:
            values = {s: q * closes.loc[date, s] for s, q in holdings[date].items()}
            for item in entitlements.itertuples():
                action = actions.loc[actions.action_key.eq(item.action_key)].iloc[0]
                if action.ex_date > pd.Timestamp(date):
                    continue
                settled = ledger.loc[
                    ledger.action_key.eq(item.action_key) & ledger.trade_date.astype(str).le(date)
                ]
                if action.cash_per_share > 0 and not settled.action_type.eq("CASH_DIVIDEND").any():
                    values[item.symbol] = (
                        values.get(item.symbol, 0.0)
                        + item.entitled_quantity * action.cash_per_share
                    )
                if action.stock_ratio > 0 and not settled.action_type.eq("STOCK_DIVIDEND").any():
                    values[item.symbol] = (
                        values.get(item.symbol, 0.0)
                        + int(item.entitled_quantity * action.stock_ratio)
                        * closes.loc[date, item.symbol]
                    )
            pnl = stock_wealth_change(
                prior_values,
                values,
                fills.loc[fills.trade_date.astype(str).eq(date)],
                ledger.loc[ledger.trade_date.astype(str).eq(date)],
            )
            current_equity = float(equity.set_index("trade_date").loc[date, "equity"])
            error = sum(pnl.values()) - (current_equity - previous_equity)
            if abs(error) > 0.01:
                raise ValueError("STOCK_WEALTH_NOT_RECONCILED:" + name + ":" + date)
            for s, value in pnl.items():
                totals[s] = totals.get(s, 0.0) + value
            days.append({"date": date, "pnl": pnl, "reconciliation_error": error})
            prior_values, previous_equity = values, current_equity
        output["variants"][variant] = {
            "stock_pnl": totals,
            "daily": days,
            "total_pnl": sum(totals.values()),
            "fees": float(fills.fee.sum()),
        }
    native, style = (output["variants"][v] for v in ("native_pca", "style_feasible"))
    output["stock_difference"] = sorted(
        [
            {
                "symbol": s,
                "name": names.loc[s, "name"] if s in names.index else s,
                "industry": names.loc[s, "industry"]
                if s in names.index and "industry" in names
                else None,
                "native_pnl": native["stock_pnl"].get(s, 0.0),
                "style_pnl": style["stock_pnl"].get(s, 0.0),
                "difference_pp": 100
                * (style["stock_pnl"].get(s, 0.0) - native["stock_pnl"].get(s, 0.0))
                / initial,
            }
            for s in set(native["stock_pnl"]) | set(style["stock_pnl"])
        ],
        key=lambda r: r["difference_pp"],
    )
    output["daily_difference"] = [
        {
            "date": a["date"],
            "difference_pp": 100 * (sum(b["pnl"].values()) - sum(a["pnl"].values())) / initial,
        }
        for a, b in zip(native["daily"], style["daily"], strict=True)
    ]
    write_diagnostic(folder / ("integer-loss-" + account + ".json"), output)
    print(
        json.dumps(
            {
                "worst": output["stock_difference"][:8],
                "best": output["stock_difference"][-5:],
                "daily": output["daily_difference"],
            },
            ensure_ascii=True,
        )
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default="candidate_20261010")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.account)
