"""Two-symbol, bounded read-only minute comparison. Never touches trading state."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Shanghai")


def compare_minutes(intraday, history, started_ms):
    # Endpoint label semantics are not certified. Exclude the current minute
    # and one extra minute, rather than call an evolving endpoint a mismatch.
    cutoff = (started_ms // 60000 - 1) * 60000
    day = datetime.fromtimestamp(started_ms / 1000, TZ).date()
    fields = ("open", "high", "low", "close", "volume", "amount")

    def index(rows):
        result, duplicates = {}, set()
        for row in rows:
            stamp = int(row["timestamp"])
            if datetime.fromtimestamp(stamp / 1000, TZ).date() != day or stamp >= cutoff:
                continue
            if stamp in result:
                duplicates.add(stamp)
            result[stamp] = row
        return result, duplicates

    left, left_dupes = index(intraday)
    right, right_dupes = index(history)
    common = set(left) & set(right)
    mismatches = []
    invalid = []
    field_mismatches = {field: 0 for field in fields}
    max_field_deltas = {field: 0.0 for field in fields}
    for stamp in sorted(common):
        try:
            values = [(float(left[stamp][f]), float(right[stamp][f])) for f in fields]
            if not all(math.isfinite(x) and math.isfinite(y) for x, y in values):
                raise ValueError("nonfinite")
            if any(abs(x - y) > 1e-6 for x, y in values):
                mismatches.append(stamp)
            for field, (x, y) in zip(fields, values, strict=True):
                delta = abs(x - y)
                max_field_deltas[field] = max(max_field_deltas[field], delta)
                field_mismatches[field] += delta > 1e-6
        except (KeyError, TypeError, ValueError):
            invalid.append(stamp)
    missing = sorted(set(left) - set(right))
    missing_intraday = sorted(set(right) - set(left))
    expected = set()
    for hour, minutes in ((9, range(31, 60)), (10, range(60)), (11, range(30)),
                          (13, range(1, 60)), (14, range(60))):
        for minute in minutes:
            stamp = int(datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ).timestamp() * 1000)
            if stamp < cutoff:
                expected.add(stamp)
    return {"status": "SAMPLE_MATCH" if common and not (missing or missing_intraday or mismatches or invalid or left_dupes or right_dupes)
            else "BLOCKED_OR_INCOMPLETE", "intraday_rows": len(intraday), "history_rows": len(history),
            "conservative_cutoff_ms": cutoff, "compared_rows": len(common),
            "missing_history_slots": missing, "mismatch_slots": mismatches,
            "missing_intraday_slots": missing_intraday,
            "field_mismatch_counts": field_mismatches, "max_field_deltas": max_field_deltas,
            "missing_regular_session_slots": sorted(expected - common),
            "invalid_slots": invalid, "intraday_duplicate_slots": sorted(left_dupes),
            "history_duplicate_slots": sorted(right_dupes), "whole_pool_verified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-pool", action="store_true", help="Read actual held/order pool, at most 60 symbols")
    parser.add_argument("--summarize-existing", help="Private report filename; recompute field diagnostics without network")
    args = parser.parse_args()
    if args.summarize_existing:
        name = args.summarize_existing
        if Path(name).name != name or not name.startswith("alphafeed-minute-acceptance-") or not name.endswith(".json"):
            raise SystemExit("Invalid private report filename")
        report = json.loads((ROOT / "state/private" / name).read_text(encoding="utf-8"))
        for entry in report["results"].values():
            started = entry["summary"]["conservative_cutoff_ms"] + 60000
            entry["summary"] = compare_minutes(entry["intraday"], entry["history"], started)
        output = ROOT / "state/private" / name.replace(".json", "-diagnostics.json")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"mode": "OFFLINE_ONLY", "symbols": len(report["results"]), "output": output.name}))
        return
    sys.path.insert(0, str(ROOT / ".runtime/alphafeed-sdk"))
    from alphafeed import AlphaFeed
    from dotenv import dotenv_values

    key = dotenv_values(ROOT / "state/private/alphafeed.env").get("ALPHAFEED_API_KEY")
    if not key:
        raise SystemExit("Private key unavailable; not tested")
    symbols = ["600036.SH", "688785.SH"]
    if args.current_pool:
        with build_opener(ProxyHandler({})).open("http://127.0.0.1:8788/api/live-accounts", timeout=15) as response:
            payload = json.load(response)
        symbols = sorted({row["symbol"] for row in payload.get("quotes", [])})
        if payload.get("quote_error") or len(symbols) != payload.get("symbol_count") or not 1 <= len(symbols) <= 60:
            raise SystemExit("Current pool unavailable/incomplete or exceeds 60; no provider calls sent")
    report = {"mode": "READ_ONLY", "observed_at": datetime.now(TZ).isoformat(),
              "requested_symbols": symbols, "results": {}}
    started_run = time.monotonic()
    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    path = ROOT / f"state/private/alphafeed-minute-acceptance-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    client = AlphaFeed(api_key=key, timeout=10, max_retries=0,
                       cache_dir=str(ROOT / "state/private/alphafeed-cache"))
    try:
        for symbol in symbols:
            if time.monotonic() - started_run > 600:
                raise TimeoutError("Bounded acceptance deadline")
            started = int(datetime.now(TZ).timestamp() * 1000)
            left = client.klines.intraday(symbol, to_dataframe=True).to_dict("records")
            time.sleep(2)
            right = client.klines.get(symbol, period="1m", count=300, adjust="none",
                                      to_dataframe=True).to_dict("records")
            result = compare_minutes(left, right, started)
            report["results"][symbol] = {"summary": result, "intraday": left, "history": right}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(json.dumps({"symbol": symbol, **result}, ensure_ascii=False), flush=True)
            time.sleep(2)
    except Exception as error:  # noqa: BLE001 - sanitize SDK exceptions, never leak credential URLs
        # Do not print provider error strings or credential-bearing request URLs.
        report["error_type"] = type(error).__name__
        print("MINUTE_ACCEPTANCE_BLOCKED", type(error).__name__, flush=True)
    finally:
        client.close()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
