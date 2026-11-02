"""Read-only frozen-basis drift decomposition and next-open repair evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.style_drift_audit import centered_exposure, decompose_drift, repair_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable repair evidence")
    report_path = args.source/"results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    obligation_namespace = report.get("obligation_namespace", report.get("resume_source_sha256", digest))
    independent = json.loads((args.source/"independent-audit.json").read_text(encoding="utf-8"))
    if independent["source_sha256"] != digest or independent["status"] != "PASS_ENGINEERING_LEDGER_ONLY":
        raise ValueError("SOURCE_LEDGER_NOT_VALIDATED")
    root = Path(__file__).resolve().parents[1]
    symbols = sorted({s for p in report["reconstructed_holdings"].values() for s in p})
    quotes = ParquetStore(root/"data").read("daily", columns=["trade_date", "ts_code", "open", "close"],
        filters=[("trade_date", ">=", report["start"]), ("trade_date", "<=", report["end"]), ("ts_code", "in", symbols)]).set_index(["trade_date", "ts_code"])
    opens = {e["trade_date"]: e for e in report["summary"]["opening_plan_audits"]}
    events = {e["date"]: e for e in report["events"] if e.get("actual_style", {}).get("active_controls")}
    dates = sorted(report["reconstructed_holdings"])
    calendar = trading_dates(ParquetStore(root/"data"), report["end"])
    with sqlite3.connect((args.source/"paper.sqlite3").resolve().as_uri()+"?mode=ro", uri=True) as connection:
        ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", connection)
    drift_rows = []; alert_rows = []
    for date, event in events.items():
        opening = opens[date]
        plan = opening["plan"]
        if plan["status"] != "PASS_REFERENCE_QUOTES_ONLY":
            raise ValueError("DRIFT_DECOMPOSITION_REQUIRES_VALIDATED_OPEN")
        frozen = pd.DataFrame.from_dict(plan["execution_style_proof"]["standardized"], orient="index")
        open_quantity = opening["execution_check"]["actual_quantities"]
        calculated_open = centered_exposure(open_quantity, quotes.loc[date].open, frozen)
        stored_open = pd.Series(opening["execution_check"]["actual_style_exposures"])
        if (abs(calculated_open-stored_open).max() > 1e-8):
            raise ValueError("OPEN_EXPOSURE_EVIDENCE_MISMATCH")
        closed = report["reconstructed_holdings"][date]
        close_frozen = centered_exposure(closed, quotes.loc[date].close, frozen)
        updated = pd.Series(event["actual_style"]["active_controls"])
        decomposition = decompose_drift(calculated_open, close_frozen, updated)
        for factor, row in decomposition.iterrows():
            limit = event.get("monitoring_limits", {}).get(factor,
                .01 if factor.startswith("unknown_") else .03 if factor.startswith(("industry_", "board_")) else .1)
            item = {"trade_date": date, "factor": factor, "limit": limit, **row.to_dict()}
            item["frozen_close_would_breach"] = abs(close_frozen[factor]) > limit+1e-6
            item["updated_close_breaches"] = abs(updated[factor]) > limit+1e-6
            drift_rows.append(item)
            if abs(updated[factor]) <= limit+1e-6:
                continue
            next_date = dates[dates.index(date)+1] if dates.index(date)+1 < len(dates) else None
            next_open = opens.get(next_date)
            before = after = None
            passed = bool(next_open and next_open["plan"]["status"] == "PASS_REFERENCE_QUOTES_ONLY")
            if passed:
                next_plan = next_open["plan"]
                next_frozen = pd.DataFrame.from_dict(next_plan["execution_style_proof"]["standardized"], orient="index")
                # Prior close shares plus actually settled stock entitlements.
                prior = pd.Series(closed, dtype=float)
                matured = ledger.loc[ledger.trade_date.astype(str).eq(next_date) & ledger.action_type.eq("STOCK_DIVIDEND")].groupby("symbol").share_quantity.sum()
                prior = prior.add(matured, fill_value=0.)
                pre_exposure = centered_exposure(prior, quotes.loc[next_date].open, next_frozen)
                post_exposure = centered_exposure(next_open["execution_check"]["actual_quantities"], quotes.loc[next_date].open, next_frozen)
                # This must be the same signal-date risk basis as the alert,
                # not a newer basis selected after observing the repair.
                if next_plan["signal_date"] != date:
                    raise ValueError("REPAIR_USES_WRONG_SIGNAL_DATE")
                same_basis_close = centered_exposure(closed, quotes.loc[date].close, next_frozen)
                if factor not in same_basis_close or abs(same_basis_close[factor]-updated[factor]) > 1e-8:
                    raise ValueError("REPAIR_BASIS_CHANGED_NOT_A_TRUE_REPAIR")
                before = pre_exposure.get(factor); after = post_exposure.get(factor)
            state = repair_state(before, after, limit, next_session=next_date is not None, plan_passed=passed)
            alert_rows.append({"alert_date": date, "factor": factor, "alert_exposure": float(updated[factor]), "limit": limit,
                "next_session": next_date, "pre_trade_exposure": before, "post_trade_exposure": after,
                "expected_next_session": calendar[calendar.index(date)+1] if calendar.index(date)+1 < len(calendar) else None,
                "obligation_key": hashlib.sha256((obligation_namespace+":"+date+":"+factor).encode()).hexdigest(),
                "block_reason": next_open["plan"].get("reason") if next_open and not passed else None,
                "state": state, "next_plan_status": next_open["plan"]["status"] if next_open else None})
    args.output.mkdir(parents=True)
    drift = pd.DataFrame(drift_rows)
    alerts = pd.DataFrame(alert_rows)
    drift.to_parquet(args.output/"exposure-drift-decomposition.parquet", index=False)
    alerts.to_parquet(args.output/"next-open-repair-chain.parquet", index=False)
    if abs(drift.identity_residual).max() > 1e-10:
        raise ValueError("DRIFT_ACCOUNTING_IDENTITY_FAILED")
    states = dict(Counter(a["state"] for a in alert_rows))
    result = {"status": "COMPLETED_OBSERVED_DRIFT_REPAIR_AUDIT_NOT_PREDICTIVE_PROOF", "orders_enabled": False,
        "source_report_sha256": digest, "source_variant": report.get("variant"), "drift_rows": len(drift),
        "obligation_namespace": obligation_namespace,
        "alert_factors": len(alerts), "alert_days": int(alerts.alert_date.nunique()), "repair_states": states,
        "max_identity_residual": float(abs(drift.identity_residual).max()),
        "effect_magnitudes_by_factor": drift.groupby("factor")[["within_day_weight_effect", "control_and_reference_update_effect"]].agg(lambda x: float(abs(x).sum())).to_dict("index"),
        "updated_only_alert_factors": int((drift.updated_close_breaches & ~drift.frozen_close_would_breach).sum()),
        "alerts_already_outside_in_frozen_basis": int((drift.updated_close_breaches & drift.frozen_close_would_breach).sum()),
        "pending_alerts": [a for a in alert_rows if a["state"].startswith("PENDING")],
        "state_labels": {"REPAIRED_AFTER_EXECUTED_TRADES": "次日成交后修复", "NATURAL_REENTRY_BEFORE_EXECUTION": "次日交易前自然回归",
            "PENDING_WINDOW_END": "窗口结束待验证", "PENDING_BLOCKED_EXECUTION": "交易受限待修复",
            "PENDING_MISSING_EVIDENCE": "缺少证据待核验", "STILL_OUTSIDE_AFTER_EXECUTION": "成交后仍越界"},
        "limitations": ["exposure identity, not named factor return attribution or causal learning proof",
            "control-update term includes scaling, reference and universe changes, not just new economic information",
            "repair is observed joint alpha/risk rebalancing, not an independently deployed emergency trader",
            "no proof beyond recorded window; terminal alerts remain pending", "no risk limit changes or live/paper orders submitted by this audit"]}
    (args.output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["alert_days", "alert_factors", "repair_states", "max_identity_residual", "updated_only_alert_factors", "alerts_already_outside_in_frozen_basis"]}))


if __name__ == "__main__":
    main()
