"""Same-window engineering comparisons with actual close exposure trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.data.store import ParquetStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable comparison")
    reports = {"named_style": json.loads((args.style/"results.json").read_text(encoding="utf-8")),
        "topk_daily_control": json.loads((args.control/"results.json").read_text(encoding="utf-8"))}
    style, control = reports.values()
    for field in ("start", "end", "slippage_bps", "config", "input_hashes"):
        if style[field] != control[field]:
            raise ValueError("COMPARISON_CONTRACT_MISMATCH:"+field)
    if style.get("integer_target_seconds", 30) != control.get("integer_target_seconds", 30):
        raise ValueError("COMPARISON_INTEGER_TIME_BUDGET_MISMATCH")
    for report in reports.values():
        for path, digest in report["input_hashes"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError("INPUT_CHANGED")
    signals = [{e["date"]: e["source_sha256"] for e in r["events"] if "source_sha256" in e} for r in reports.values()]
    if signals[0] != signals[1]:
        raise ValueError("NOT_IDENTICAL_ARCHIVED_SIGNALS")
    for date in [e["date"] for e in style["events"] if "actual_style" in e]:
        a = pd.read_parquet(args.style/f"controls-{date}.parquet")
        b = pd.read_parquet(args.control/f"controls-{date}.parquet")
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    args.output.mkdir(parents=True)
    store = ParquetStore(Path(__file__).resolve().parents[1]/"data")
    symbols = sorted({s for r in reports.values() for p in r["reconstructed_holdings"].values() for s in p})
    close = store.read("daily", columns=["trade_date", "ts_code", "close"], filters=[("trade_date", ">=", style["start"]),
        ("trade_date", "<=", style["end"]), ("ts_code", "in", symbols)]).set_index(["trade_date", "ts_code"]).close
    trajectory = []; comparison = []
    for variant, directory in (("named_style", args.style), ("topk_daily_control", args.control)):
        report = reports[variant]
        audit = json.loads((directory/"independent-audit.json").read_text(encoding="utf-8"))
        if (audit["status"] != "PASS_ENGINEERING_LEDGER_ONLY"
                or audit["source_sha256"] != hashlib.sha256((directory/"results.json").read_bytes()).hexdigest()):
            raise ValueError("LEDGER_AUDIT_NOT_PASSED")
        equity = pd.read_parquet(directory/"equity.parquet").set_index("trade_date")
        exposures = []
        for event in report["events"]:
            date = event["date"]
            if "actual_style" not in event:
                continue
            holdings = pd.Series(report["reconstructed_holdings"][date], dtype=float)
            row = {"variant": variant, "trade_date": date, "gross_exposure": float(equity.loc[date, "gross_exposure"]), "holdings": len(holdings)}
            if len(holdings):
                z = pd.read_parquet(directory/f"controls-{date}.parquet")
                values = holdings*close.loc[date].reindex(holdings.index)
                if values.isna().any() or set(holdings.index)-set(z.index):
                    raise ValueError("MISSING_ACTUAL_STYLE_OR_MARK")
                w = values/values.sum()
                row["effective_holdings"] = float(1/(w @ w))
                row["growth_board_weight"] = float(w @ z.reindex(w.index)[["board_STAR", "board_CHINEXT"]].sum(axis=1))
                tech_cols = ["industry_"+s for s in ["电子", "计算机", "通信"]]
                row["technology_industry_weight"] = float(w @ z.reindex(index=w.index, columns=tech_cols, fill_value=0.).sum(axis=1))
            else:
                row.update(growth_board_weight=None, technology_industry_weight=None, effective_holdings=0.)
            measured = event["actual_style"]
            row["active_tracking_volatility"] = measured["active_tracking_volatility"]
            for factor, value in (measured.get("active_controls") or {}).items():
                monitor_limit = event.get("monitoring_limits", {}).get(factor,
                    .01 if factor.startswith("unknown_") else .03 if factor.startswith(("industry_", "board_")) else .10)
                exposures.append({"trade_date": date, "factor": factor, "actual_active_exposure": value,
                    "monitoring_limit": monitor_limit, "close_breach": abs(value) > monitor_limit+1e-6,
                    "excess": max(0., abs(value)-monitor_limit)})
            trajectory.append(row)
        pd.DataFrame(exposures).to_parquet(args.output/(variant+"-active-controls.parquet"), index=False)
        blocked = [e for e in report["events"] if e["status"].startswith("BLOCKED")]
        blocked_opens = [e for e in report["summary"]["opening_plan_audits"] if e["plan"]["status"].startswith("BLOCKED")]
        comparison.append({"variant": variant, "summary": {k:v for k,v in report["summary"].items() if k != "opening_plan_audits"},
            "blocked_closes": blocked, "blocked_opens": len(blocked_opens), "measured_days": len({e["trade_date"] for e in exposures}),
            "closing_remeasured_breach_days": len({e["trade_date"] for e in exposures if e["close_breach"]}),
            "max_closing_remeasured_excess": max((e["excess"] for e in exposures), default=0.),
            "independent_audit_sha256": hashlib.sha256((directory/"independent-audit.json").read_bytes()).hexdigest()})
    pd.DataFrame(trajectory).to_parquet(args.output/"actual-style-trajectories.parquet", index=False)
    # Fractional, zero-cost daily cap-weight reference. This is not an
    # executable account and must not be described as a tradable benchmark.
    signal_dates = sorted(signals[0])
    all_dates = sorted(style["reconstructed_holdings"])
    reference_weights = {d: pd.read_parquet(args.style/f"weights-{d}.parquet").benchmark for d in signal_dates}
    reference_symbols = sorted({s for w in reference_weights.values() for s in w.index})
    filters = [("trade_date", ">=", style["start"]), ("trade_date", "<=", style["end"]), ("ts_code", "in", reference_symbols)]
    bars = store.read("daily", columns=["trade_date", "ts_code", "open", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["trade_date", "ts_code", "adj_factor"], filters=filters)
    bars = bars.merge(adj, on=["trade_date", "ts_code"], how="left", validate="one_to_one").set_index(["trade_date", "ts_code"])
    reference = [{"trade_date": all_dates[0], "daily_return": 0., "net_value": 1.}]
    nav = 1.
    reference_gaps = []
    for i,date in enumerate(all_dates[1:], 1):
        prior = all_dates[i-1]; w = reference_weights[prior]
        current = bars.loc[date].reindex(w.index)
        if i == 1:
            returns = current.close/current.open-1
        else:
            previous = bars.loc[prior].reindex(w.index)
            returns = current.close*current.adj_factor/(previous.close*previous.adj_factor)-1
        if not np.isfinite(returns).all():
            reference_gaps.append({"signal_date": prior, "trade_date": date,
                "symbols": returns.index[~np.isfinite(returns)].tolist()})
            reference.append({"trade_date": date, "daily_return": None, "net_value": None})
            continue
        daily_return = float(w @ returns)
        nav *= 1+daily_return
        reference.append({"trade_date": date, "daily_return": daily_return, "net_value": None if reference_gaps else nav})
    pd.DataFrame(reference).to_parquet(args.output/"fractional-zero-cost-reference.parquet", index=False)
    has_blocked = any(r["blocked_closes"] or r["blocked_opens"] for r in comparison)
    result = {"status": "COMPLETED_COMPARISON_WITH_BLOCKED_SESSIONS_NOT_ALPHA_PROOF" if has_blocked else "COMPLETED_MATCHED_EX_POST_COMPARISON_NOT_ALPHA_PROOF", "orders_enabled": False,
        "comparison": comparison, "signal_dates": sorted(signals[0]), "source_reports": {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.style/"results.json", args.control/"results.json"]},
        "fractional_risk_pool_cap_reference_return": None if reference_gaps else nav-1,
        "fractional_reference_status": "BLOCKED_PRICE_GAPS_NO_UNVERIFIED_FILL" if reference_gaps else "COMPLETE_THEORETICAL_ZERO_COST",
        "fractional_reference_gaps": reference_gaps,
        "reference_is_not_executable": True,
        "limitations": ["TopK is a daily matched ablation, not original five-session production schedule or account reconstruction",
            "portfolio methods differ in both style constraints and allocation, not a pure causal style-only intervention",
            "statistical PCA risk decomposition is not named-factor return attribution", "short known window and revised historical fundamentals do not establish predictive validity"]}
    (args.output/"comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps([{k:r["summary"][k] for k in ["total_return", "fill_count", "max_drawdown"]} for r in comparison]))


if __name__ == "__main__":
    main()
