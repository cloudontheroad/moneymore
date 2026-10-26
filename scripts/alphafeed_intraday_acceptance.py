"""Bounded read-only sampling of the running API; no execution or recovery calls."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    opener = build_opener(ProxyHandler({}))
    report = {"mode": "READ_ONLY", "samples": []}
    previous = {}
    for index in range(3):
        started = time.monotonic()
        try:
            with opener.open("http://127.0.0.1:8788/api/live-accounts", timeout=15) as response:
                payload = json.load(response)
            rows = payload.get("quotes", [])
            current = {row["symbol"]: row for row in rows}
            common = set(previous) & set(current)
            summary = {
                "observed_at": payload.get("observed_at"),
                "request_seconds": round(time.monotonic() - started, 3),
                "source": payload.get("quote_source"),
                "has_quote_error": bool(payload.get("quote_error")),
                "pool_size": payload.get("symbol_count"),
                "returned": len(current),
                "ready": sum(bool(row.get("execution_ready")) for row in rows),
                "blocked": [{"symbol": row["symbol"], "reasons": row.get("quality_reasons")}
                            for row in rows if not row.get("execution_ready")],
                "max_quote_age": max((row.get("quote_age_seconds", 0) for row in rows), default=None),
                "max_depth_age": max((row.get("depth_age_seconds") or 0 for row in rows), default=None),
                "max_skew_seconds": max((abs(row["timestamp"] - row["depth_timestamp"]) / 1000
                                         for row in rows if row.get("depth_timestamp")), default=None),
                "common_with_previous": len(common),
                "quote_timestamp_advanced": sum(current[s]["timestamp"] > previous[s]["timestamp"] for s in common),
                "depth_timestamp_advanced": sum(current[s]["depth_timestamp"] > previous[s]["depth_timestamp"] for s in common),
                "timestamp_regressions": [s for s in common if current[s]["timestamp"] < previous[s]["timestamp"]
                                          or current[s]["depth_timestamp"] < previous[s]["depth_timestamp"]],
                "volume_or_amount_regressions": [s for s in common if current[s]["volume"] < previous[s]["volume"]
                                                 or current[s]["amount"] < previous[s]["amount"]],
                "invalid_vwap": [row["symbol"] for row in rows if row.get("vwap") is None
                                 or not row["low"] - 1e-6 <= row["vwap"] <= row["high"] + 1e-6],
                "volume_conversion_errors": [row["symbol"] for row in rows
                                             if abs(row["volume_shares"] - row["volume"] * 100) > 1e-6],
            }
            report["samples"].append({"summary": summary, "quotes": rows})
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            previous = current
        except Exception as exc:
            # Never print arbitrary provider messages or credential-bearing URLs.
            report["error_type"] = type(exc).__name__
            print("READ_ONLY_CHECK_BLOCKED", type(exc).__name__, flush=True)
            break
        if index < 2:
            time.sleep(15)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    path = ROOT / f"state/private/alphafeed-intraday-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
