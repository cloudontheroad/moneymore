"""S0/S1 bounded read-only audit of completed prediction/label artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.style_foundation import economic_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit evidence is immutable")
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    folders = [root / "state/weekly-learning/isolated-research/candidates" / ("20261102_" + name)
        for name in ("sector_total_return", "absolute_total_return", "risk_adjusted_total_return")]
    reports = [json.loads((p / "research.json").read_text(encoding="utf-8")) for p in folders]
    base = pd.read_parquet(next(folders[0].glob("*-test-predictions.parquet")))
    label_column = "evaluation_label" if "evaluation_label" in base else "sector_relative_label"
    signal_dates = base.index.get_level_values("datetime")
    symbols = sorted(base.index.get_level_values("instrument").unique())
    cutoff = reports[0]["time_contract"]["data_end"].replace("-", "")
    calendar = trading_dates(store, cutoff)
    first = signal_dates.min().strftime("%Y%m%d")
    days = [d for d in calendar if first <= d <= cutoff]
    filters = [("ts_code", "in", symbols), ("trade_date", ">=", first), ("trade_date", "<=", cutoff)]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters)
    rebuilt = economic_labels(daily, adj, days)
    findings = []
    for folder, research in zip(folders, reports):
        files = []
        contract = research["effective_segments"]
        endpoints = {}
        for left, right in (("train", "valid"), ("valid", "blend"), ("blend", "test")):
            end = contract[left][1].replace("-", "")
            start = contract[right][0].replace("-", "")
            endpoints[left+"_to_"+right] = calendar.index(end)+5 < calendar.index(start)
        for path in sorted(folder.glob("*-test-predictions.parquet")):
            frame = pd.read_parquet(path)
            target = "evaluation_label" if "evaluation_label" in frame else "sector_relative_label"
            check = frame.join(rebuilt)
            diff = (check[target] - check.rebuilt).abs()
            daily_ic = check.groupby(level="datetime").apply(
                lambda x: x.score.corr(x.rebuilt, method="spearman"), include_groups=False)
            month = daily_ic.groupby(daily_ic.index.strftime("%Y-%m")).mean()
            files.append({"model": path.stem, "rows": len(check),
                "duplicates": int(frame.index.duplicated().sum()),
                "nonfinite_scores": int((~np.isfinite(frame.score)).sum()),
                "missing_reconstructed_labels": int(check.rebuilt.isna().sum()),
                "label_differences_over_1e_6": int((diff > 1e-6).sum()),
                "max_absolute_label_difference": float(diff.max()),
                "reconstructed_rank_ic": float(daily_ic.mean()),
                "monthly_rank_ic": month.to_dict(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        findings.append({"protocol": research["protocol"]["label_protocol"],
            "purged_boundaries": endpoints, "models": files,
            "blend_validation": research.get("blend_fit"),
            "research_sha256": hashlib.sha256((folder / "research.json").read_bytes()).hexdigest()})
    universe_coverage = []
    for table in ("strategy_universe", "universe_constituent_snapshots", "universe_membership"):
        df = store.read(table)
        date_columns = {col: {"min": str(df[col].min()), "max": str(df[col].max()),
            "unique": int(df[col].nunique())} for col in df.columns if col in ("date", "trade_date", "effective_date", "snapshot_date", "as_of_date")}
        universe_coverage.append({"table": table, "rows": len(df), "columns": list(df.columns), "dates": date_columns})
    result = {"status": "AUDIT_COMPLETED_NOT_STRATEGY_PASS", "orders_enabled": False,
        "protocol_sha256": hashlib.sha256((root / "configs/style_balance_research_v1.yaml").read_bytes()).hexdigest(),
        "label_reconstruction": "close * adj_factor, exchange calendar forward five sessions, no gap fill",
        "findings": findings, "universe_coverage": universe_coverage,
        "unresolved": ["current-pool training is not a historical dynamic-pool proof",
            "raw price features versus adjusted return targets require corporate-action sensitivity audit",
            "financial vintages and feature normalization not covered by this bounded audit",
            "economic label reconstruction cannot certify source adjustment factors",
            "negative IC does not prove a coding bug or establish inverse-signal alpha"],
        "benchmark_status": "PROPOSED_NOT_FROZEN_PENDING_PIT_COVERAGE",
        "risk_budget_status": "UNFROZEN_NO_PUBLICATION"}
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "protocols": len(findings),
        "models": sum(len(f["models"]) for f in findings),
        "label_mismatches": sum(m["label_differences_over_1e_6"] for f in findings for m in f["models"])}))


if __name__ == "__main__":
    main()
