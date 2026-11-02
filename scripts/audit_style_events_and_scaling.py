"""S1 event evidence plus native Qlib training-only normalization dry-run."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.qlib_challenger import build_challenger_dataset, challenger_universe
from moneymore.research.qlib_feature_preprocessing import fit_training_processor, transform_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new evidence directory")
    root = Path(__file__).resolve().parents[1]
    previous = root / "state/style-balance/s1-feature-paths-20261102-a/audit.json"
    audit = json.loads(previous.read_text(encoding="utf-8"))
    store = ParquetStore(root / "data")
    symbols = [r["symbol"] for r in audit["features"]]
    dividends = store.read("dividend", filters=[("ts_code", "in", symbols)])
    events = []
    for row in audit["features"]:
        for event in row["raw_vs_adjusted_large_jump_examples"]:
            date = str(event["date"]).replace("-", "")[:8]
            evidence = dividends.loc[dividends.ts_code.eq(row["symbol"]) & dividends.ex_date.astype(str).eq(date)]
            matching = evidence.loc[evidence.div_proc.astype(str).str.contains("实施", regex=False)]
            events.append({"symbol": row["symbol"], "date": date,
                "raw_close": event["raw_close"], "adjusted_close": event["close"],
                "classification": "PROVIDER_IMPLEMENTED_EX_DATE_MATCH" if len(matching) else "UNEXPLAINED_BY_DIVIDEND_TABLE",
                "provider_evidence": json.loads(matching.to_json(orient="records", force_ascii=False)),
                "not_original_document_attestation": True})
    universe = challenger_universe(root, store)
    dataset = build_challenger_dataset(store, {s: universe[s] for s in symbols}, require_label=False, feature_count=12)
    dates = dataset.index.get_level_values("datetime")
    dataset = dataset.loc[dates <= pd.Timestamp("2026-10-30")]
    artifact = fit_training_processor(dataset, "2015-01-05", "2025-09-17")
    normalized = transform_features(artifact, dataset)
    lag0 = dataset["feature"].filter(regex="lag00$")
    after = normalized["feature"][lag0.columns]
    stats = {name: {"raw_std": float(lag0[name].std()), "normalized_std": float(after[name].std()),
        "clip_fraction": float(after[name].abs().ge(3).mean())} for name in lag0}
    args.output.mkdir(parents=True)
    target = args.output / "feature-processor.pkl"
    target.write_bytes(pickle.dumps(artifact))
    payload = {"status": "ISOLATED_PREPROCESSING_VALIDATED_NOT_MODEL_VALIDATED", "orders_enabled": False,
        "sample_stocks": len(symbols), "sample_rows": len(dataset), "events": events, "feature_stats": stats,
        "processor": {k:v for k,v in artifact.items() if k != "processor"},
        "processor_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_audit_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "limitations": ["only eight stocks sampled", "event matching is provider evidence, not source truth",
            "scaling does not remove raw-price event shocks", "no model retraining or performance claim",
            "future production needs inference-side loading of the exact frozen processor"]}
    (args.output / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(dataset), "events": len(events),
        "ex_date_matches": sum(r["classification"] == "PROVIDER_IMPLEMENTED_EX_DATE_MATCH" for r in events)}))


if __name__ == "__main__":
    main()
