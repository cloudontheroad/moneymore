"""All saved candidates: native Qlib named-style constraint feasibility ablation."""

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import prediction_reports, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.enhanced_indexing import EnhancedPolicy, optimize_snapshot


def run(root, as_of):
    store = ParquetStore(root / "data")
    accounts = {"main_gru": root / "state/qlib-challenger-shadow"}
    accounts.update({"candidate_" + p.name: p for p in (root / "state/qlib-candidate-shadow").iterdir() if p.is_dir()})
    reports = {name: [r for r in prediction_reports(path, as_of) if r["trade_date"] == as_of]
               for name, path in accounts.items()}
    # Reuse frozen snapshot evidence; do not recompute risk from future bars.
    import json
    snapshot = json.loads((root / "state/enhanced-indexing-research/latest.json").read_text(encoding="utf-8"))
    if snapshot["as_of_date"] != as_of:
        raise ValueError("RISK_EVIDENCE_DATE_MISMATCH")
    evidence = Path(snapshot["input_evidence"])
    returns = pd.read_parquet(evidence / "risk_returns.parquet")
    size = pd.read_parquet(evidence / "market_cap.parquet").total_mv
    close = store.read("daily", columns=["ts_code", "close"], filters=[("trade_date", "==", as_of)]).set_index("ts_code").close
    output = dict(as_of_date=as_of, orders_enabled=False, automatic_promotion=False,
                  experiment="FIXED_LAMBDA10_NAMED_STYLE_CONSTRAINT_ABLATION_NOT_WINNER_SELECTION",
                  risk_input_sha256=hashlib.sha256((evidence / "risk_returns.parquet").read_bytes()).hexdigest(),
                  benchmark="equal_weight_same_eligible_recorded_pool", results=[])
    for account, rs in reports.items():
        if not rs:
            output["results"].append(dict(account=account, status="NO_SAME_DATE_SAVED_PREDICTION"))
            continue
        report = rs[0]
        scores = pd.DataFrame(report["scores"]).set_index("instrument").score
        symbols = [s for s in scores.index if s in returns and np.isfinite(returns[s]).all()
                   and np.isfinite(size.get(s, np.nan)) and size.get(s, 0) > 0]
        risk = returns[symbols]
        market = risk.mean(axis=1)
        market_variance = market.var(ddof=1)
        if market_variance <= 0:
            raise ValueError("INVALID_MARKET_VARIANCE")
        styles = pd.DataFrame({"momentum_60": (1 + risk.tail(60)).prod() - 1,
                               "volatility_20": risk.tail(20).std(),
                               "market_beta_120": risk.apply(lambda x: x.cov(market) / market_variance),
                               "log_size": np.log(size.reindex(symbols))})
        quantities = {p["symbol"]: p["quantity"] for p in report["portfolio"]["positions"] if p["quantity"] > 0}
        missing = [s for s in quantities if s not in symbols or not np.isfinite(close.get(s, np.nan))]
        if missing or not quantities:
            output["results"].append(dict(account=account, status="BLOCKED_HELD_INPUT", missing=missing))
            continue
        w0 = pd.Series({s: q * close[s] for s, q in quantities.items()})
        w0 /= w0.sum()
        for variant, retention in (("native_pca", None), ("size_neutral", None),
                                   ("all_styles_retain_75pct", .75), ("all_styles_retain_50pct", .5),
                                   ("all_styles_neutral", 0.)):
            policy = EnhancedPolicy(lamb=10., size_neutral=variant == "size_neutral")
            try:
                result = optimize_snapshot(scores, risk, size, w0, policy,
                    style_controls=styles if retention is not None else None,
                    style_retention=retention if retention is not None else 0.)
            except ValueError as error:
                result = dict(status="BLOCKED", failures=[str(error)])
            output["results"].append({**result, "account": account, "variant": variant,
                "policy": asdict(policy), "source_sha256": report["source_sha256"],
                "risk_assets": len(symbols), "excluded_symbols": sorted(set(scores.index) - set(symbols))})
            print(account, variant, result["status"], flush=True)
    write_diagnostic(root / "state/enhanced-indexing-research/all-candidate-styles.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.as_of)
