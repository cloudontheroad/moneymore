"""Native Qlib current-snapshot ablation; not an executable trading plan or backtest."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.enhanced_indexing import (
    EnhancedPolicy,
    carry_confirmed_suspensions,
    optimize_snapshot,
)


def run(root, as_of):
    store = ParquetStore(root / "data")
    protected = [store.curated / (name + '.parquet') for name in ('fina_indicator','global_factor_recommendation','multi_sector_account_daily','qlib_challenger_account_daily','qlib_candidate_account_daily')]
    before_hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    config = yaml.safe_load(
        (root / "configs/enhanced_indexing_research.yaml").read_text(encoding="utf-8")
    )
    paths = {
        "main_gru": root / "state/qlib-challenger-shadow" / f"{as_of}.json",
        "candidate_20261010": root / "state/qlib-candidate-shadow/20261010" / f"{as_of}.json",
    }
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    symbols = sorted({str(row["instrument"]) for r in reports.values() for row in r["scores"]})
    calendar = store.read("trade_calendar")
    calendar = calendar.loc[
        (calendar.exchange == "SSE")
        & calendar.is_open.astype(int).eq(1)
        & (calendar.cal_date.astype(str) <= as_of)
    ]
    dates = sorted(calendar.cal_date.astype(str).unique())[-(config["risk_lookback"] + 1) :]
    if len(dates) != config["risk_lookback"] + 1:
        raise ValueError("INSUFFICIENT_EXCHANGE_CALENDAR")
    filters = [
        ("ts_code", "in", symbols),
        ("trade_date", ">=", dates[0]),
        ("trade_date", "<=", as_of),
    ]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", filters=filters)
    data = daily.merge(adj, on=["ts_code", "trade_date"], how="inner", validate="one_to_one")
    data["adjusted"] = data.close * data.adj_factor
    prices = data.pivot(index="trade_date", columns="ts_code", values="adjusted").reindex(dates)
    try:
        suspension = store.read("enhanced_indexing_suspension_proof")
    except FileNotFoundError:
        suspension = pd.DataFrame()
    prices, carried = carry_confirmed_suspensions(prices, suspension)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    basic = store.read(
        "daily_basic",
        columns=["ts_code", "trade_date", "total_mv"],
        filters=[
            ("ts_code", "in", symbols),
            ("trade_date", ">=", dates[-10]),
            ("trade_date", "<=", as_of),
        ],
    )
    size = basic.sort_values("trade_date").groupby("ts_code").tail(1).set_index("ts_code").total_mv
    close = daily.loc[daily.trade_date.eq(as_of)].set_index("ts_code").close
    output = {
        "as_of_date": as_of,
        "mode": config["mode"],
        "config": config,
        "results": [],
        "risk_start": dates[1],
        "risk_end": dates[-1],
        "risk_observations": len(returns),
        "returns_or_nav_rewritten": False,
        "orders_enabled": False,
        "automatic_promotion": False,
        "warning": "Snapshot only, no calibrated expected returns or performance proof. PCA factors are statistical, not industry/Barra factors.",
    }
    output["confirmed_suspension_carries"] = carried
    evidence = root / "state/enhanced-indexing-research/evidence" / as_of / uuid4().hex
    evidence.mkdir(parents=True, exist_ok=False)
    returns.to_parquet(evidence / "risk_returns.parquet")
    size.to_frame("total_mv").to_parquet(evidence / "market_cap.parquet")
    output["input_evidence"] = str(evidence)
    for name, report in reports.items():
        signal_dates = {str(row["datetime"]).replace("-", "")[:8] for row in report["scores"]}
        if signal_dates != {as_of}:
            output["results"].append(
                {
                    "account": name,
                    "status": "BLOCKED_STALE_OR_MIXED_SIGNALS",
                    "signal_dates": sorted(signal_dates),
                }
            )
            continue
        scores = pd.DataFrame(report["scores"]).set_index("instrument").score
        scored = sorted(scores.dropna().index)
        eligible = [
            s
            for s in scored
            if s in returns
            and np.isfinite(returns[s]).all()
            and np.isfinite(size.get(s, np.nan))
            and size.get(s, 0) > 0
        ]
        quantities = {
            p["symbol"]: p["quantity"]
            for p in report["portfolio"]["positions"]
            if p["quantity"] > 0
        }
        missing_held = [
            s for s in quantities if s not in eligible or not np.isfinite(close.get(s, np.nan))
        ]
        if missing_held:
            output["results"].append(
                {"account": name, "status": "BLOCKED_HELD_INPUT", "missing_holdings": missing_held}
            )
            continue
        current = pd.Series({s: q * close[s] for s, q in quantities.items()}, dtype=float)
        current /= current.sum()
        for variant in config["size_control_variants"]:
            for lamb in config["risk_aversion_grid"]:
                policy = EnhancedPolicy(
                    lamb=lamb,
                    delta=config["turnover_l1_limit"],
                    maximum_weight=config["maximum_weight"],
                    num_factors=config["num_statistical_factors"],
                    size_neutral=variant == "benchmark_neutral",
                )
                capture = {}
                result = optimize_snapshot(
                    scores, returns[eligible], size, current, policy, audit_capture=capture
                )
                risk_file = evidence / f"{name}-{variant}-{lamb}.npz"
                np.savez_compressed(risk_file, **capture)
                row = {
                    **result,
                    "account": name,
                    "size_variant": variant,
                    "policy": asdict(policy),
                    "scored_assets": len(scored),
                    "risk_assets": len(eligible),
                    "excluded_risk_symbols": sorted(set(scored) - set(eligible)),
                    "source_report_sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
                }
                row["risk_input_file"] = str(risk_file)
                row["risk_input_sha256"] = hashlib.sha256(risk_file.read_bytes()).hexdigest()
                if result["weights"]:
                    ranked = sorted(result["weights"].items(), key=lambda x: x[1], reverse=True)
                    row["top20_weights"] = [{"symbol": s, "weight": w} for s, w in ranked[:20]]
                    row["star_chinext_weight"] = sum(
                        w for s, w in ranked if s.startswith(("688", "689", "300", "301"))
                    )
                output["results"].append(row)
                print(name, variant, lamb, result["status"], result["failures"], flush=True)
    output['protected_tables_unchanged'] = before_hashes == {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    write_diagnostic(root / "state/enhanced-indexing-research/latest.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.as_of)
