"""Four bounded, read-only provider calls; timestamp evidence, not gate relaxation."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from alphafeed import AlphaFeed
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Shanghai")


def main():
    key = dotenv_values(ROOT / "state/private/alphafeed.env").get("ALPHAFEED_API_KEY")
    if not key:
        raise SystemExit("Private key not configured")
    symbols = ["688785.SH", "600036.SH"]
    report = {"observed_at": datetime.now(TZ).isoformat(), "mode": "READ_ONLY", "results": {}}
    client = AlphaFeed(api_key=key, timeout=10, max_retries=0,
                       cache_dir=str(ROOT / "state/private/alphafeed-cache"))
    try:
        quotes = {row["symbol"]: row for row in client.quotes.get(symbols=symbols)}
        time.sleep(2)
        depths = client.depth.batch(symbols, max_workers=1)
        for symbol in symbols:
            time.sleep(2)
            frame = client.klines.intraday(symbol, to_dataframe=True)
            quote, depth = quotes.get(symbol, {}), depths.get(symbol, {})
            active = frame.loc[frame["volume"] > 0].sort_values("timestamp")
            last = active.iloc[-1].to_dict() if len(active) else {}
            last_minute = int(last.get("timestamp", 0))
            quote_stamp = int(quote.get("timestamp", 0))
            summary = {
                "symbol": symbol, "quote_time": datetime.fromtimestamp(quote_stamp / 1000, TZ).isoformat(),
                "depth_time": datetime.fromtimestamp(int(depth.get("timestamp", 0)) / 1000, TZ).isoformat(),
                "minute_rows": len(frame), "nonzero_volume_minutes": len(active),
                "quote_volume": quote.get("volume"), "minute_volume_sum": float(frame["volume"].sum()),
                "quote_amount": quote.get("amount"), "minute_amount_sum": float(frame["amount"].sum()),
                "last_nonzero_minute": last.get("trade_time"),
                "quote_in_last_nonzero_minute": last_minute <= quote_stamp < last_minute + 60000,
                "quote_matches_last_minute_close": abs(float(quote.get("last_price", 0)) - float(last.get("close", 0))) < 1e-6,
            }
            report["results"][symbol] = {"summary": summary, "quote": quote, "depth": depth,
                                          "minutes": frame.to_dict("records")}
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as error:
        report["error_type"] = type(error).__name__
        print("TIMESTAMP_PROBE_BLOCKED", type(error).__name__, flush=True)
    finally:
        client.close()
    name = "alphafeed-timestamps-" + datetime.now(TZ).strftime("%Y%m%d-%H%M%S") + ".json"
    (ROOT / "state/private" / name).write_text(json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
