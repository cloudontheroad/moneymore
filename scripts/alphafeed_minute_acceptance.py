"""Two-symbol, bounded read-only minute comparison. Never touches trading state."""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
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
    for stamp in sorted(common):
        try:
            values = [(float(left[stamp][f]), float(right[stamp][f])) for f in fields]
            if not all(math.isfinite(x) and math.isfinite(y) for x, y in values):
                raise ValueError("nonfinite")
            if any(abs(x - y) > 1e-6 for x, y in values):
                mismatches.append(stamp)
        except (KeyError, TypeError, ValueError):
            invalid.append(stamp)
    missing = sorted(set(left) - set(right))
    return {"status": "SAMPLE_MATCH" if common and not (missing or mismatches or invalid or left_dupes or right_dupes)
            else "BLOCKED_OR_INCOMPLETE", "intraday_rows": len(intraday), "history_rows": len(history),
            "conservative_cutoff_ms": cutoff, "compared_rows": len(common),
            "missing_history_slots": missing, "mismatch_slots": mismatches,
            "invalid_slots": invalid, "intraday_duplicate_slots": sorted(left_dupes),
            "history_duplicate_slots": sorted(right_dupes), "whole_pool_verified": False}


def main():
    sys.path.insert(0, str(ROOT / ".runtime/alphafeed-sdk"))
    from alphafeed import AlphaFeed
    from dotenv import dotenv_values

    key = dotenv_values(ROOT / "state/private/alphafeed.env").get("ALPHAFEED_API_KEY")
    if not key:
        raise SystemExit("Private key unavailable; not tested")
    report = {"mode": "READ_ONLY", "observed_at": datetime.now(TZ).isoformat(), "results": {}}
    client = AlphaFeed(api_key=key, timeout=10, max_retries=0,
                       cache_dir=str(ROOT / "state/private/alphafeed-cache"))
    try:
        for symbol in ("600036.SH", "688785.SH"):
            started = int(datetime.now(TZ).timestamp() * 1000)
            left = client.klines.intraday(symbol, to_dataframe=True).to_dict("records")
            time.sleep(2)
            right = client.klines.get(symbol, period="1m", count=300, adjust="none",
                                      to_dataframe=True).to_dict("records")
            result = compare_minutes(left, right, started)
            report["results"][symbol] = {"summary": result, "intraday": left, "history": right}
            print(json.dumps({"symbol": symbol, **result}, ensure_ascii=False), flush=True)
            time.sleep(2)
    except Exception as error:  # noqa: BLE001 - sanitize SDK exceptions, never leak credential URLs
        # Do not print provider error strings or credential-bearing request URLs.
        report["error_type"] = type(error).__name__
        print("MINUTE_ACCEPTANCE_BLOCKED", type(error).__name__, flush=True)
    finally:
        client.close()
    stamp = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    path = ROOT / f"state/private/alphafeed-minute-acceptance-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
