"""Rebuild isolated financial core fields from preserved raw responses. No network calls."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.financial_reconciliation import FIELDS, KEYS, reconcile_indicator_payload
from moneymore.data.fundamental_research import TABLE, financial_quality
from moneymore.data.store import ParquetStore
from moneymore.strategy_universe import active_strategy_universe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    protected = [
        store.curated / (name + ".parquet")
        for name in (
            "fina_indicator",
            "global_factor_recommendation",
            "multi_sector_account_daily",
            "qlib_challenger_account_daily",
            "qlib_candidate_account_daily",
        )
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    payloads = []
    for directory in (TABLE, "fina_indicator_payload_v2", "fina_indicator_deep_probe"):
        for path in sorted((store.raw / "tushare" / directory).glob("*.parquet")):
            from pyarrow import parquet as pq

            columns = (
                KEYS
                + FIELDS
                + (["update_flag"] if "update_flag" in pq.read_schema(path).names else [])
            )
            f = pd.read_parquet(path, columns=columns)
            f["_payload_capture"] = path.stem.split("_")[-1]
            payloads.append(f)
    raw = pd.concat(payloads, ignore_index=True)
    normalized = reconcile_indicator_payload(raw)
    existing = store.read(TABLE)
    extras = [c for c in existing if c not in normalized and c not in KEYS]
    if extras:
        normalized = normalized.merge(
            existing[KEYS + extras], on=KEYS, how="left", validate="one_to_one"
        )
    symbols = active_strategy_universe(store).symbol.tolist()
    report = {
        "as_of_date": args.as_of,
        "apply": args.apply,
        "payload_files": len(payloads),
        "raw_rows": len(raw),
        "normalized_rows": len(normalized),
        "conflicting_keys": {c: int(normalized[f"conflict__{c}"].sum()) for c in FIELDS},
        "before": financial_quality(existing, symbols, args.as_of, per_field=True),
        "after": financial_quality(normalized, symbols, args.as_of, per_field=True),
    }
    if args.apply:
        store.merge_curated(TABLE, [normalized], KEYS)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report["protected_tables_unchanged"] = before == after
    write_diagnostic(root / "state/fundamental-research-v2/duplicate-reconciliation.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("before", "after")}, ensure_ascii=False
        )
    )
    print(
        json.dumps(
            {
                k: {n: v for n, v in report[k].items() if not n.endswith("_symbols")}
                for k in ("before", "after")
            },
            ensure_ascii=False,
        )
    )
    if before != after:
        raise RuntimeError("Concurrent protected-table mutation detected; investigate separately")
