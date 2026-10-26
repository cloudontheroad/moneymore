"""Bounded local validation of the four manually authorized GUI exports.

No GUI, clipboard, broker/network or trade-ledger access. Do not use as an
execution daemon. Only aggregate evidence is written; source files stay private.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from moneymore.desktop_tables import TableValidationError, parse_huatai_export


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "state/private/huatai-acceptance"
    report = {"captured_at": datetime.now(UTC).isoformat(),
              "mode": "AUTHORIZED_FOREGROUND_READ_ONLY_EXPORT",
              "background_automation_verified": False,
              "broker_receipts_verified": False, "trading_ready": False,
              "ledger_updated": False, "exports": []}
    files = [("positions-20261022-2140.xls", "positions"),
             ("orders-20261022-2016.xls", "orders"),
             ("trades-20261022-2017.xls", "trades")]
    for filename, kind in files:
        path = folder / filename
        evidence = {"kind": kind, "source": filename}
        try:
            data = path.read_bytes()
            parsed = parse_huatai_export(data, kind=kind, account_ref="huatai_local",
                                         trade_date="20261022")
            evidence.update(status=parsed["status"], rows=len(parsed["rows"]),
                            encoding=parsed["encoding"], source_sha256=parsed["source_sha256"],
                            source_mtime=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat())
        except (OSError, TableValidationError):
            evidence.update(status="BLOCKED", rows=None)
        report["exports"].append(evidence)
    history_path = folder / "history-20261021-captured20261022.xls"
    try:
        data = history_path.read_bytes()
        if len(data) > 5_000_000:
            raise ValueError("oversize")
        rows = list(csv.DictReader(io.StringIO(data.decode("gbk"), newline=None), delimiter="\t"))
        report["history_observation"] = {
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(rows), "operation_counts": dict(Counter(r.get("操作", "") for r in rows)),
            "fill_imported": False, "status": "OBSERVED_ONLY_NOT_TRADE_RECEIPTS",
            "trade_time_column_present": "成交时间" in (rows[0] if rows else {}),
        }
    except (OSError, UnicodeError, ValueError, csv.Error):
        report["history_observation"] = {"status": "BLOCKED"}
    target = folder / "acceptance-20261022.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
