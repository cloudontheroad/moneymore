"""Bounded offline support scan to find later missing execution symbols upfront."""

import json
from pathlib import Path

import pandas as pd
from replay_corrected_v2 import variant_scores

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.portfolio_constructor import global_topk_portfolio, trailing_return_correlation
from moneymore.research.baseline_v2_release import digest, read_json


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-release"
    current = read_json(base / "corrected-daily-replay.json")
    frozen = read_json(Path(current["output_directory"]) / "frozen-inputs.json")
    paths = {Path(p).name: Path(p) for p in frozen["input_hashes"]}
    for path, checksum in frozen["input_hashes"].items():
        if digest(Path(path)) != checksum:
            raise ValueError("FROZEN_INPUT_CHANGED")
    overlay = pd.read_parquet(paths["native-with-derived-factors.parquet"])
    overlay["price"] = overlay.close * overlay.derived_adj_factor
    prices = overlay.pivot(index="trade_date", columns="ts_code", values="price")
    source = read_json(paths["frozen-protocol.json"])
    start = source["scoring"]["coverage_audit"]["identity_scope_start"]
    prices = prices.loc[prices.index >= start]
    prices.index = pd.to_datetime(prices.index)
    panel = pd.read_parquet(paths["daily-scores.parquet"])
    known = set(pd.read_parquet(paths["execution-bars.parquet"], columns=["symbol"]).symbol)
    for marker in (base / "corrected-execution-data/execution-extensions").glob("*/report.json"):
        report = read_json(marker)
        if report["score_panel_sha256"] == digest(paths["daily-scores.parquet"]):
            for name, checksum in report["artifact_hashes"].items():
                if digest(Path(report["output_directory"]) / name) != checksum:
                    raise ValueError("EXTENSION_HASH_MISMATCH")
            known.update(report["symbols"])
    p = frozen["policy"]["portfolio"]
    required = set()
    for count, (date, group) in enumerate(panel.groupby("date"), 1):
        correlation = trailing_return_correlation(
            prices.reindex(columns=group.symbol), date, lookback=int(p["correlation_lookback"])
        )
        for variant in ("fixed", "learned"):
            _, _, ranking = global_topk_portfolio(
                variant_scores(group, variant),
                set(),
                symbol_column="symbol",
                top_k=int(p["top_k"]),
                exit_rank=int(p["exit_rank"]),
                max_replacements=int(p["max_replacements"]),
                gross_exposure=float(p["gross_exposure"]),
                minimum_weight=float(p["minimum_weight"]),
                maximum_weight=float(p["maximum_weight"]),
                correlation=correlation,
                cluster_correlation_threshold=float(p["cluster_correlation_threshold"]),
                maximum_cluster_members=int(p["maximum_cluster_members"]),
                weighting_method=p["weighting_method"],
            )
            required.update(ranking.head(int(p["top_k"])).symbol)
        result = {
            "status": "COMPLETE" if count == panel.date.nunique() else "SCANNING",
            "completed_days": count,
            "required_symbols": sorted(required),
            "missing_symbols": sorted(required - known),
            "score_panel": str(paths["daily-scores.parquet"]),
            "score_panel_sha256": digest(paths["daily-scores.parquet"]),
            "scope": "OFFLINE_EXECUTION_SUPPORT_NOT_ADVANCE_TRADING_SIGNALS",
            "real_orders_enabled": False,
        }
        if count % 50 == 0 or result["status"] == "COMPLETE":
            write_diagnostic(base / "corrected-execution-support.json", result)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
