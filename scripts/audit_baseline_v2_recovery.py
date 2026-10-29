"""Read-only diagnosis of native risk recovery; never approves an account recovery."""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.research.baseline_v2_release import digest, read_json

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report_path = (
        root
        / "state/baseline-v2-validation/daily-inputs/stress-replays/platform_risk_reductions/latest.json"
    )
    report = read_json(report_path)
    directory = Path(report["output_directory"])
    runs, artifacts = [], [report_path]
    for variant in ("fixed", "learned"):
        equity_path = directory / f"{variant}-equity.parquet"
        plans_path = directory / f"{variant}-opening-audits.json"
        database = directory / f"{variant}.sqlite3"
        equity = pd.read_parquet(equity_path)
        plans = read_json(plans_path)["plans"]
        risks = [
            p
            for p in plans
            if p["plan"].get("risk_state", {}).get("effective_state", "NORMAL") != "NORMAL"
        ]
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            state = dict(
                connection.execute(
                    "SELECT * FROM account_risk_states WHERE account_id=?", (variant,)
                ).fetchone()
            )
        flat = equity.gross_exposure.abs().le(1e-9)
        active = equity.loc[~flat]
        terminal_drawdown = float(equity.equity.iloc[-1] / equity.equity.cummax().iloc[-1] - 1)
        runs.append(
            {
                "variant": variant,
                "risk_constrained_openings": len(risks),
                "first_risk_date": risks[0]["trade_date"] if risks else None,
                "last_stock_exposure_date": str(active.trade_date.iloc[-1])
                if len(active)
                else None,
                "cash_only_days": int(flat.sum()),
                "terminal_drawdown_from_high_watermark": terminal_drawdown,
                "terminal_effective_state": state["effective_state"],
                "terminal_proposed_state": state["proposed_state"],
                "native_manual_approval_would_change_state": bool(state["recovery_required"]),
                "automatic_reentry_validated": False,
            }
        )
        artifacts += [equity_path, plans_path, database]
    result = {
        "status": "BLOCKED_REENTRY_RISK_BUDGET_PROTOCOL_NOT_VALIDATED",
        "runs": runs,
        "artifact_hashes": {str(p): digest(p) for p in artifacts},
        "finding": "Realized losses preserve high-watermark drawdown while in cash; native approve_risk_recovery only adopts a less restrictive proposed state, not a new risk budget or peak.",
        "prohibited_shortcuts": [
            "ARBITRARY_COOLDOWN_AUTO_UNLOCK",
            "RESET_PEAK_TO_IMPROVE_BACKTEST",
            "DISABLE_SAFETY_TO_MATCH_82_PERCENT_RETURN",
        ],
        "account_recovery_performed": False,
        "real_orders_enabled": False,
    }
    write_diagnostic(root / "state/baseline-v2-release/recovery-diagnosis.json", result)
    print(json.dumps(result))
