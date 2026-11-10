"""Resumable OpenD collection of daily US option-underlying statistics."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from futu import OpenQuoteContext, RET_OK

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = max((ROOT / "data/us_overnight_paper_only/universe").glob("top1000-*.parquet"))
OUT = ROOT / "data/us_overnight_event_warehouse/options_daily"


def pages(context, method: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    frames, key = [], None
    for _ in range(20):
        result = getattr(context, method)(symbol, begin_time=start, end_time=end, page_req_key=key)
        if result[0] != RET_OK:
            raise RuntimeError(str(result[1]))
        if len(result[1]):
            frames.append(result[1])
        key = result[2]
        if not key:
            break
        time.sleep(0.04)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=str(pd.Timestamp.now(tz="UTC").date()))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = pd.read_parquet(UNIVERSE)["code"].astype(str).tolist()
    if args.limit:
        symbols = symbols[: args.limit]
    failures, completed = [], 0
    context = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        for index, symbol in enumerate(symbols, 1):
            target = OUT / f"{symbol.replace('.', '_')}.parquet"
            if target.exists():
                completed += 1
                continue
            try:
                statistics = pages(context, "get_option_underlying_his_statistic", symbol, args.start, args.end)
                volatility = pages(context, "get_option_underlying_his_volatility", symbol, args.start, args.end)
                if statistics.empty and volatility.empty:
                    continue
                keys = [column for column in ["code", "time", "timestamp"] if column in statistics and column in volatility]
                if not statistics.empty and not volatility.empty:
                    drop = [column for column in ["name", "underlying_price"] if column in volatility]
                    frame = statistics.merge(volatility.drop(columns=drop), on=keys, how="outer")
                else:
                    frame = statistics if not statistics.empty else volatility
                for column in frame.columns:
                    if column not in {"code", "name", "time"}:
                        frame[column] = pd.to_numeric(frame[column], errors="coerce")
                frame["first_observed_at_utc"] = datetime.now(UTC).isoformat()
                frame.to_parquet(target, index=False)
                completed += 1
            except Exception as exc:
                failures.append({"symbol": symbol, "error": str(exc)})
            if index % 25 == 0:
                print(json.dumps({"processed": index, "saved": completed, "failures": len(failures)}), flush=True)
            time.sleep(0.06)
    finally:
        context.close()
    manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(), "range": [args.start, args.end],
        "requested": len(symbols), "saved": completed, "failures": failures,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
