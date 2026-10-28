"""Audit daily research inputs; optional bounded, resumable official downloads."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_readiness import daily_schedule, validate_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-requests", type=int, default=25)
    args = parser.parse_args()
    if not 0 <= args.max_requests <= 100:
        parser.error("max-requests must be between 0 and 100")
    root = Path(__file__).resolve().parents[1]
    folder = root / "state/baseline-v2-validation/daily-inputs"
    folder.mkdir(parents=True, exist_ok=True)
    # Exclusive OS file handle: another invocation cannot download concurrently.
    import msvcrt

    with (folder / "sync.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        store = ParquetStore(root / "data")
        schedule = daily_schedule(trading_dates(store, "20261028"), "20240131", "20251230")
        base = store.read("baseline_v2_daily_basic")
        groups = {str(d): f for d, f in base.groupby("trade_date")}
        client, requests, rows, union = None, 0, [], set()
        failure = None
        for signal, source in schedule:
            path = folder / (source + ".parquet")
            marker = path.with_suffix(".json")
            frame = groups.get(source)
            origin = "EXISTING_ISOLATED_SNAPSHOT"
            if frame is None and path.exists() and marker.exists():
                proof = json.loads(marker.read_text(encoding="utf-8"))
                if proof["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("DAILY_CACHE_HASH_MISMATCH:" + source)
                frame, origin = pd.read_parquet(path), "HASH_VERIFIED_CACHE"
            if frame is None and args.download and requests < args.max_requests and failure is None:
                try:
                    if client is None:
                        from dotenv import load_dotenv

                        from moneymore.data.tushare_provider import TushareProvider

                        load_dotenv(root / ".env", override=False)
                        client = TushareProvider()._client
                    time.sleep(1.3)
                    requests += 1
                    frame = client.daily_basic(
                        trade_date=source,
                        fields="ts_code,trade_date,close,pb,pe_ttm,dv_ttm,total_mv",
                    )
                    validate_snapshot(frame, source)
                    frame.to_parquet(path, index=False)
                    write_diagnostic(
                        marker,
                        {
                            "date": source,
                            "rows": len(frame),
                            "captured_at": datetime.now(UTC).isoformat(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        },
                    )
                    origin = "OFFICIAL_FETCHED"
                except Exception as exc:  # noqa: BLE001 - stop batch; never leak provider secrets
                    # Do not persist provider text which may contain transport credentials.
                    failure = type(exc).__name__
                    frame = None
            if frame is None:
                rows.append({"signal_date": signal, "source_date": source, "status": "MISSING"})
                continue
            top = validate_snapshot(frame, source)
            union.update(top.ts_code)
            rows.append(
                {
                    "signal_date": signal,
                    "source_date": source,
                    "status": "READY",
                    "origin": origin,
                    "rows": len(frame),
                }
            )
        known = set(store.read("fina_indicator_research_v2", columns=["ts_code"]).ts_code)
        report = {
            "status": "INPUT_AUDIT_ONLY_NOT_DAILY_BACKTEST",
            "created_at": datetime.now(UTC).isoformat(),
            "input_references": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    root / "configs/baseline_v2.yaml",
                    root / "state/baseline-v2-validation/weight-validation.json",
                ]
            },
            "requests": requests,
            "provider_failure_type": failure,
            "signals": len(rows),
            "ready": sum(r["status"] == "READY" for r in rows),
            "missing": sum(r["status"] == "MISSING" for r in rows),
            "observed_union_symbols": len(union),
            "union_complete": all(r["status"] == "READY" for r in rows),
            "observed_union_missing_research_financials": sorted(union - known),
            "gates_remaining": [
                *(["EXACT_DAILY_SNAPSHOTS"] if any(r["status"] != "READY" for r in rows) else []),
                "FULL_UNION_PRICE_FINANCIAL_AUDIT",
                "DAILY_PIT_SCORING",
                "ACTUAL_INCUMBENT_CONSTRUCTOR_REPLAY",
                "EXECUTION_AND_LEDGER_AUDIT",
            ],
            "schedule": rows,
            "real_orders_enabled": False,
        }
        write_diagnostic(
            folder / "audits" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"), report
        )
        write_diagnostic(folder / "readiness.json", report)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in {"schedule", "observed_union_missing_research_financials"}
                }
            )
        )


if __name__ == "__main__":
    main()
