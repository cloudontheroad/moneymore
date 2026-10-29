"""Map concrete symbol/field/period/publication dependencies; never certifies PIT."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.fundamental_research import research_financial_frame
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_scores import logical_hash
from moneymore.research.baseline_v2_financials import common_period_features, required_period
from moneymore.research.baseline_v2_release import digest, read_json
from moneymore.research.security_identity import SecurityIdentity

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    release = Path(read_json(root / "state/baseline-v2-release/latest.json")["directory"])
    manifest = read_json(release / "manifest.json")
    protocol = read_json(Path(manifest["replay_directory"]) / "frozen-protocol.json")
    identity = SecurityIdentity(protocol["scoring"]["coverage_audit"]["identity_policy"])
    store = ParquetStore(root / "data")
    finance = identity.market_rows(
        research_financial_frame(store),
        "ann_date",
        ["roe", "q_sales_yoy"],
        record_keys=["ann_date", "end_date", "growth_source"],
    )
    if (
        logical_hash(finance)
        != protocol["scoring"]["coverage_audit"]["source_logical_hashes"]["financial_records"]
    ):
        raise ValueError("FROZEN_FINANCIAL_INPUT_CHANGED")
    score_path = next(
        Path(p) for p in protocol["input_hashes"] if p.endswith("daily-scores.parquet")
    )
    daily = pd.read_parquet(score_path, columns=["date", "symbol"])
    daily["date"] = daily.date.dt.strftime("%Y%m%d")
    daily["scope"] = "FROZEN_DAILY_SCORED_INPUT"
    sample_path = (
        Path(read_json(root / "state/baseline-v2-validation/sample-audit.json")["output_directory"])
        / "sample-exclusions.parquet"
    )
    sample = pd.read_parquet(sample_path)
    sample = sample.loc[sample.reason.eq("ELIGIBLE"), ["date", "symbol"]].copy()
    sample["symbol"] = sample.symbol.map(identity.stable)
    sample["scope"] = "MONTHLY_TRAIN_VALID_TEST_MONITOR_REBUILD_REQUIREMENT"
    needed = pd.concat([daily, sample], ignore_index=True).drop_duplicates()
    rows, missing = [], []
    for date, group in needed.groupby("date", sort=True):
        selected = common_period_features(finance, date).set_index("ts_code")
        for scope, members in group.groupby("scope"):
            for field in ("roe", "q_sales_yoy"):
                frame = selected.reindex(members.symbol).copy()
                valid = frame[field].notna() & frame[field + "_end_date"].astype(str).eq(
                    required_period(date)
                )
                if not valid.all():
                    missing += [
                        {"date": str(date), "symbol": str(s), "field": field, "scope": scope}
                        for s in frame.index[~valid]
                    ]
                valid_frame = frame.loc[valid]
                rows += [
                    {
                        "symbol": str(s),
                        "field": field,
                        "end_date": str(r[field + "_end_date"]),
                        "ann_date": str(r[field + "_ann_date"]),
                        "source_value": float(r[field]),
                        "scope": scope,
                        "signal_date": str(date),
                        "source": "TUSHARE_FINA_INDICATOR"
                        if field == "roe"
                        else str(r["growth_source"]),
                    }
                    for s, r in valid_frame.iterrows()
                ]
    detail = pd.DataFrame(rows)
    keys = ["symbol", "field", "end_date", "ann_date", "source_value", "scope", "source"]
    dependency = (
        detail.groupby(keys, dropna=False).signal_date.agg(["min", "max", "count"]).reset_index()
    )
    dependency = dependency.rename(
        columns={
            "min": "first_signal_date",
            "max": "last_signal_date",
            "count": "affected_scored_stock_days",
        }
    )
    output = root / "state/baseline-v2-release/financial-vintage-requirements"
    output.mkdir(exist_ok=True)
    signature = hashlib.sha256(
        (digest(score_path) + digest(sample_path) + logical_hash(finance)).encode()
    ).hexdigest()
    path = output / (signature[:20] + "-requirements.parquet")
    dependency.to_parquet(path, index=False)
    report = {
        "status": "BLOCKED_ORIGINAL_VINTAGES_REQUIRED_WITH_EXACT_DEPENDENCIES",
        "dependency_rows": len(dependency),
        "unique_symbols": int(dependency.symbol.nunique()),
        "report_periods": sorted(dependency.end_date.unique()),
        "fields": ["roe", "q_sales_yoy"],
        "missing_reconstructed_field_rows": len(missing),
        "scope_counts": dependency.groupby("scope").size().to_dict(),
        "original_vintage_attested": False,
        "required_external_evidence": "As-published field values or original statements with original/revision publication timestamps for listed symbol/period/field dependencies.",
        "artifact_hashes": {str(p): digest(p) for p in (path, score_path, sample_path)},
        "original_training_input_provenance_claimed": False,
        "production_changed": False,
        "real_orders_enabled": False,
    }
    if missing:
        missing_path = path.with_name(path.stem + "-missing.parquet")
        pd.DataFrame(missing).to_parquet(missing_path, index=False)
        report["artifact_hashes"][str(missing_path)] = digest(missing_path)
    write_diagnostic(root / "state/baseline-v2-release/financial-vintage-requirements.json", report)
    print(json.dumps(report))
