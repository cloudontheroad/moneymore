"""At most eight exact-symbol official probes for the unresolved daily price holes."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.tushare_provider import TushareProvider

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    report = json.loads((base / "gap-investigation.json").read_text(encoding="utf-8"))
    source = Path(report["output_directory"]) / "price-gaps.parquet"
    holes = pd.read_parquet(source)
    holes = holes.loc[holes.status.eq("UNEXPLAINED_PRICE_GAP")]
    if holes.symbol.nunique() > 2:
        raise ValueError("EXACT_PROBE_SCOPE_EXCEEDS_TWO_SYMBOLS")
    directory = Path(report["output_directory"]) / "price-rechecks"
    directory.mkdir(parents=True, exist_ok=True)
    load_dotenv(root / ".env", override=False)
    provider, calls, rows = TushareProvider(), 0, []
    for symbol, group in holes.groupby("symbol"):
        for kind in ("daily", "basic_L", "basic_D", "basic_P"):
            path = directory / (symbol + "-" + kind + ".parquet")
            marker = path.with_suffix(".json")
            if path.exists() and marker.exists():
                proof = json.loads(marker.read_text(encoding="utf-8"))
                if proof["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("EXACT_PROBE_HASH_MISMATCH")
                frame = pd.read_parquet(path)
            else:
                time.sleep(1.3)
                calls += 1
                try:
                    if kind == "daily":
                        frame = provider._client.daily(
                            ts_code=symbol,
                            start_date=group.trade_date.min(),
                            end_date=group.trade_date.max(),
                        )
                    else:
                        frame = provider._client.stock_basic(
                            ts_code=symbol,
                            list_status=kind[-1],
                            fields="ts_code,name,list_date,delist_date,list_status",
                        )
                except Exception as exc:  # noqa: BLE001 - no retry or credential-bearing provider text
                    raise RuntimeError("EXACT_PROBE_STOPPED:" + type(exc).__name__) from None
                if (
                    frame is None
                    or "ts_code" not in frame
                    or len(frame) >= 6000
                    or not frame.ts_code.eq(symbol).all()
                ):
                    raise ValueError("EXACT_PROBE_INVALID_SCOPE_OR_RESPONSE")
                frame.to_parquet(path, index=False)
                write_diagnostic(
                    marker,
                    {
                        "symbol": symbol,
                        "kind": kind,
                        "rows": len(frame),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "captured_at": datetime.now(UTC).isoformat(),
                    },
                )
            rows.append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "rows": len(frame),
                    "recovered_hole_dates": sorted(set(group.trade_date) & set(frame.trade_date))
                    if kind == "daily"
                    else [],
                    "listing_metadata": frame.to_dict("records") if kind != "daily" else [],
                }
            )
    result = {
        "status": "OFFICIAL_PROBES_ONLY_NOT_APPLIED",
        "requests": calls,
        "source_gap_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "results": rows,
        "production_changed": False,
        "real_orders_enabled": False,
    }
    write_diagnostic(directory / "report.json", result)
    print(json.dumps(result))
