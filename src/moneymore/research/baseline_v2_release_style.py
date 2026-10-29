"""Reuse lagged 3/5/10-peer diagnostics; preserve unmatchable return explicitly."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..challenger_attribution import matched_day
from .baseline_v2_release import read_json, save_stage, verify_manifest


def link_with_unknown(rows):
    components = dict.fromkeys(
        ["market", "style", "selection", "trading_and_other", "costs", "unmatched"], 0.0
    )
    wealth = 1.0
    for row in rows:
        if abs(sum(row.get(k, 0) for k in components) - row["actual_return"]) > 1e-10:
            raise ValueError("STYLE_DAILY_COMPONENTS_NOT_RECONCILED")
        for key in components:
            components[key] += wealth * row.get(key, 0)
        wealth *= 1 + row["actual_return"]
    return {
        **components,
        "total_return": wealth - 1,
        "reconciliation_error": sum(components.values()) - (wealth - 1),
    }


def run_release_style(manifest):
    verify_manifest(manifest)
    directory, source = Path(manifest["directory"]), Path(manifest["replay_directory"])
    frozen = read_json(source / "frozen-protocol.json")
    pool_path = (
        Path(frozen["scoring"]["identity_audit"]["output_directory"])
        / "identity-top1000-preview.parquet"
    )
    pool = pd.read_parquet(pool_path)
    pools = {str(d): f.set_index("security_id") for d, f in pool.groupby("signal_date")}
    frame = pd.read_parquet(directory / "full-union-native-prices.parquet")
    raw = frame.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    factors = frame.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex_like(
        raw
    )
    adjusted = raw * factors
    returns = adjusted.pct_change(fill_method=None)
    momentum = adjusted.shift(21) / adjusted.shift(252) - 1
    volatility = returns.rolling(60, min_periods=60).std(ddof=1)
    accounts = {}
    for variant in ("fixed", "learned"):
        equity = pd.read_parquet(source / f"{variant}-equity.parquet")
        held = pd.read_parquet(source / f"{variant}-reconstructed-holdings.parquet")
        fills = pd.read_parquet(source / f"{variant}-fills.parquet")
        accounts[variant] = (
            equity,
            {str(d): f for d, f in held.groupby("trade_date")},
            fills.groupby("trade_date").fee.sum(),
        )
    results, artifacts = [], []
    for variant, (equity, hold_groups, fees) in accounts.items():
        keys = equity.trade_date.astype(str).tolist()
        for neighbors in (3, 5, 10):
            rows = []
            for offset in range(1, len(keys)):
                previous, day = keys[offset - 1], keys[offset]
                sample = pools[previous]
                prior_hold = hold_groups.get(previous)
                positions = (
                    []
                    if prior_hold is None
                    else [
                        {"symbol": r.symbol, "quantity": r.economic_quantity}
                        for r in prior_hold.itertuples()
                    ]
                )
                names = sorted(set(sample.index) | {r["symbol"] for r in positions})
                features = pd.DataFrame(index=names)
                features["log_size"] = np.log(
                    pd.to_numeric(sample.total_mv, errors="coerce").where(sample.total_mv.gt(0))
                )
                features["book_to_price"] = 1 / pd.to_numeric(sample.pb, errors="coerce").where(
                    sample.pb.gt(0)
                )
                features["momentum"] = momentum.loc[previous].reindex(names)
                features["volatility"] = volatility.loc[previous].reindex(names)
                before = float(equity.equity.iloc[offset - 1])
                actual = float(equity.equity.iloc[offset] / before - 1)
                cost = float(fees.get(day, 0))
                matched = matched_day(
                    positions,
                    before,
                    actual,
                    cost,
                    raw.loc[previous],
                    returns.loc[day],
                    features,
                    sample.index.tolist(),
                    neighbors,
                )
                if matched["status"] != "MATCHED":
                    matched = {
                        **matched,
                        "actual_return": actual,
                        "costs": -cost / before,
                        "unmatched": actual + cost / before,
                    }
                else:
                    matched.pop(
                        "matches", None
                    )  # Keep bounded report size, not raw list of every peer.
                rows.append({"trade_date": day, **matched})
            annual = {
                year: link_with_unknown([r for r in rows if r["trade_date"].startswith(year)])
                for year in ("2024", "2025")
            }
            path = directory / f"{variant}-peer{neighbors}-daily.parquet"
            pd.DataFrame(rows).to_parquet(path, index=False)
            artifacts.append(path)
            results.append(
                {
                    "variant": variant,
                    "neighbors": neighbors,
                    "days": len(rows),
                    "matched_days": sum(r["status"] == "MATCHED" for r in rows),
                    "unmatched_status_counts": pd.Series(
                        [r["status"] for r in rows if r["status"] != "MATCHED"]
                    )
                    .value_counts()
                    .to_dict(),
                    "annual_linked": annual,
                }
            )
    original = read_json(directory / "attribution.json")
    result = {
        **original,
        "status": "ACCOUNTING_PASS_STYLE_DIAGNOSTIC_ONLY_SOURCE_GATE_BLOCKED",
        "peer_style_diagnostics": results,
        "style_limitations": [
            "NOT_CAUSAL_ALPHA",
            "FULL_UNION_ADJUSTMENT_ECONOMICS_NOT_VALIDATED",
            "UNMATCHED_DAYS_EXPLICITLY_LINKED_NOT_DROPPED",
            "LAGGED_FEATURES_AND_POOL_NO_PEER_SELECTION_ON_TODAY_RETURN",
        ],
    }
    artifacts += [Path(p) for p in original.get("artifact_hashes", {})]
    return save_stage(manifest, "attribution", result, artifacts)
