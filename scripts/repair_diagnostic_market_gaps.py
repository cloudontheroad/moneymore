"""Bounded, missing-key-only market repair. Never reruns strategies or trades."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from moneymore.challenger_diagnostics import prediction_reports, trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    store = ParquetStore(root / "data")
    reports = prediction_reports(root / "state/qlib-challenger-shadow", args.as_of)
    symbols = sorted({str(score["instrument"]) for row in reports[-21:] for score in row["scores"]})
    dates = [day for day in trading_dates(store, args.as_of) if args.start <= day <= args.as_of]
    filters = [
        ("ts_code", "in", symbols),
        ("trade_date", ">=", args.start),
        ("trade_date", "<=", args.as_of),
    ]
    existing = {
        table: store.read(table, columns=["ts_code", "trade_date"], filters=filters)
        for table in ("daily", "adj_factor")
    }
    gaps = {}
    instruments = store.read("instruments").set_index("ts_code")
    for date in dates:
        eligible = {
            symbol
            for symbol in symbols
            if symbol in instruments.index and str(instruments.loc[symbol, "list_date"]) <= date
        }
        missing = eligible - set(
            existing["daily"].loc[existing["daily"].trade_date.astype(str) == date, "ts_code"]
        )
        # Broad holes only: don't query each isolated suspension or pre-listing date.
        if len(missing) >= 20:
            gaps[date] = sorted(missing)
    if len(gaps) > 20:
        raise RuntimeError("More than 20 broad missing dates; narrow the repair window")
    audit = {
        "as_of_date": args.as_of,
        "start_date": args.start,
        "apply": args.apply,
        "broad_gaps": {date: len(values) for date, values in gaps.items()},
        "requests": 0,
        "inserted": {},
    }
    if args.apply and gaps:
        provider = TushareProvider()
        batches: dict[str, list[tuple[str, pd.DataFrame]]] = {"daily": [], "adj_factor": []}
        for date in gaps:
            for table, fetch in (
                ("daily", provider.daily_bars),
                ("adj_factor", provider.adjustment_factors),
            ):
                audit["requests"] += 1
                frame = fetch(date)
                if frame is None or frame.empty:
                    raise RuntimeError(
                        f"{table}[{date}] returned empty; no synthetic bars inserted"
                    )
                if set(frame.trade_date.astype(str)) != {date} or frame.duplicated(["ts_code", "trade_date"]).any():
                    raise RuntimeError(f"{table}[{date}] returned mismatched dates or duplicate keys")
                frame = frame.loc[frame.ts_code.isin(symbols)].copy()
                present = set(
                    existing[table].loc[existing[table].trade_date.astype(str) == date, "ts_code"]
                )
                frame = frame.loc[~frame.ts_code.isin(present)]
                if not frame.empty:
                    batches[table].append((f"diagnostic-gap-{date}-{args.as_of}", frame))
                time.sleep(1)
        for table, snapshots in batches.items():
            if snapshots:
                store.save_snapshots(table, snapshots, "tushare", ["ts_code", "trade_date"])
            audit["inserted"][table] = sum(len(frame) for _, frame in snapshots)
    write_diagnostic(root / "state/challenger-diagnostics/market-gap-repair.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
