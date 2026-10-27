"""Read-only, ledger-derived verification of isolated integer opening replays."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.replay_attribution import reconstruct_cash, reconstruct_positions


def run(root):
    folder = root / "state/enhanced-indexing-research"
    source = folder / "integer-continuous-replay.json"
    replay = json.loads(source.read_text(encoding="utf-8"))
    directory = Path(replay["output_directory"])
    daily = ParquetStore(root / "data").read(
        "daily",
        filters=[
            ("trade_date", ">=", replay["start_date"]),
            ("trade_date", "<=", replay["as_of_date"]),
        ],
    )
    opens = daily.pivot(index="trade_date", columns="ts_code", values="open")
    result = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_directory": str(directory),
        "orders_enabled": False,
        "mode": replay["mode"],
        "runs": [],
        "pairs": [],
    }
    for row in replay["runs"]:
        item = {"account": row["account"], "variant": row["variant"], "status": row["status"]}
        if row["status"] != "COMPLETED_DIAGNOSTIC_ONLY":
            item["reason"] = row.get("reason")
            result["runs"].append(item)
            continue
        name = row["account"] + "-" + row["variant"]
        with sqlite3.connect(
            (directory / (name + ".sqlite3")).as_uri() + "?mode=ro", uri=True
        ) as conn:
            fills = pd.read_sql_query("SELECT * FROM fills", conn)
            ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", conn)
        equity = pd.read_parquet(directory / (name + "-equity.parquet"))
        dates = equity.trade_date.astype(str).tolist()
        positions = reconstruct_positions(fills, ledger, dates)
        cash = reconstruct_cash(fills, ledger, dates, replay["config"]["initial_cash"])
        if not np.isfinite(list(cash.values())).all() or not np.isfinite(equity["cash"]).all():
            raise ValueError("LEDGER_NONFINITE_CASH:" + name)
        max_cash_error = max(
            abs(cash[d] - float(equity.set_index("trade_date").loc[d, "cash"])) for d in dates
        )
        if max_cash_error > 0.01:
            raise ValueError("LEDGER_CASH_MISMATCH:" + name)
        checks, blocked = [], []
        for event in row["summary"]["opening_plan_audits"]:
            plan, date = event["plan"], event["trade_date"]
            if plan["status"] == "NO_PRIOR_SIGNAL":
                continue
            if plan["status"] != "PASS_REFERENCE_QUOTES_ONLY":
                blocked.append({"date": date, "reason": plan.get("reason", plan["status"])})
                continue
            held = positions[date]
            expected = {s: int(q) for s, q in plan["target_quantities"].items() if q > 0}
            if held != expected or abs(cash[date] - plan["estimated_cash"]) > 0.01:
                raise ValueError("LEDGER_PLAN_MISMATCH:" + name + ":" + date)
            if plan["signal_date"] >= date:
                raise ValueError("NONPROSPECTIVE_PLAN_DATE")
            proof = plan["execution_style_proof"]
            z = pd.DataFrame.from_dict(proof["standardized"], orient="index")
            values = pd.Series({s: q * opens.loc[date, s] for s, q in held.items()})
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("LEDGER_INVALID_OPEN_VALUATION:" + name + ":" + date)
            active = z.reindex(values.index).T.dot(values / values.sum())
            excess = float((active.abs() - pd.Series(proof["limits"])).max())
            if not np.isfinite(active).all() or not np.isfinite(excess) or excess > 1e-5:
                raise ValueError("LEDGER_STYLE_BOUNDARY_FAILED:" + name + ":" + date)
            checks.append(
                {
                    "date": date,
                    "maximum_style_excess": excess,
                    "actual_exposures": active.to_dict(),
                    "initial_integer_bootstrap": not positions.get(dates[dates.index(date) - 1], {})
                    if dates.index(date)
                    else True,
                }
            )
        item.update(
            verified_opens=len(checks),
            blocked_opens=blocked,
            ledger_cash_max_error=max_cash_error,
            style_checks=checks,
            summary={k: v for k, v in row["summary"].items() if k != "opening_plan_audits"},
        )
        result["runs"].append(item)
    for account in sorted({r["account"] for r in result["runs"]}):
        pair = {
            r["variant"]: r for r in result["runs"] if r["account"] == account and "summary" in r
        }
        if set(pair) != {"native_pca", "style_feasible"}:
            continue
        native, style = pair["native_pca"], pair["style_feasible"]
        result["pairs"].append(
            {
                "account": account,
                "native_return": native["summary"]["total_return"],
                "style_return": style["summary"]["total_return"],
                "difference_pp": 100
                * (style["summary"]["total_return"] - native["summary"]["total_return"]),
                "native_paused_days": len(native["blocked_opens"]),
                "style_paused_days": len(style["blocked_opens"]),
                "warning": "Ex-post short-window mechanism test, includes explicit no-trade days; not OOS or promotion evidence.",
            }
        )
    result["protected_equity_tables_unchanged"] = replay["protected_equity_tables_unchanged"]
    write_diagnostic(folder / "integer-continuous-audit.json", result)
    print(json.dumps({"runs": len(result["runs"]), "pairs": result["pairs"]}, ensure_ascii=True))
    return result


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
