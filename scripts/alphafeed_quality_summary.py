"""Summarize persisted upstream observations; never makes network requests."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Shanghai")


def summarize(records, day):
    observed, reasons, http_statuses = set(), Counter(), Counter()
    total = active = ready = recoveries = malformed = 0
    unavailable = False
    for record in records:
        total += 1
        try:
            stamp = datetime.fromisoformat(record["observed_at"]).astimezone(TZ)
        except (KeyError, ValueError, TypeError):
            malformed += 1
            continue
        if stamp.strftime("%Y%m%d") != day:
            continue
        if record.get("status") == "UNAVAILABLE":
            unavailable = True
            http_statuses[str(record.get("http_status"))] += 1
            continue
        if not record.get("fresh_fetch"):
            continue
        if unavailable:
            recoveries += 1
            unavailable = False
        hm = stamp.hour * 60 + stamp.minute
        if 570 <= hm < 690 or 780 <= hm < 900:
            observed.add(hm)
            active += 1
            ready += record.get("status") == "AVAILABLE"
            reasons.update(record.get("quality_reasons", []))
    expected = set(range(570, 690)) | set(range(780, 900))
    return {"day": day, "records": total, "malformed_records": malformed,
            "active_upstream_samples": active, "all_symbols_ready_samples": ready,
            "observed_trading_minutes": len(observed),
            "missing_trading_minutes": sorted(expected - observed),
            "full_session_observed": not (expected - observed) and malformed == 0,
            "quality_reason_sample_counts": dict(reasons), "failure_http_counts": dict(http_statuses),
            "observed_recovery_transitions": recoveries,
            "real_429_observed": http_statuses["429"] > 0,
            "acceptance": "EVIDENCE_ONLY_NOT_AUTOMATIC_APPROVAL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(TZ).strftime("%Y%m%d"))
    args = parser.parse_args()
    datetime.strptime(args.date, "%Y%m%d").replace(tzinfo=TZ)
    path = ROOT / f"state/runtime/alphafeed-quality-{args.date}.jsonl"
    records = []
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({})
    result = summarize(records, args.date)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
