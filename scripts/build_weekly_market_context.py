from __future__ import annotations

import argparse
import json
from pathlib import Path

from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.qlib_challenger import challenger_universe
from moneymore.weekly_market_context import market_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True, help="last completed session, YYYYMMDD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists():
        raise FileExistsError("use a new output directory; snapshots are immutable")
    store = ParquetStore(root / "data")
    universe = challenger_universe(root, store)
    calendar = trading_dates(store, args.as_of)
    bars = store.read("daily", columns=["ts_code", "trade_date", "close"],
                      filters=[("ts_code", "in", list(universe)), ("trade_date", ">=", "20260101"), ("trade_date", "<=", args.as_of)])
    calendar = [d for d in calendar if "20260101" <= d <= args.as_of]
    summary, series = market_context(bars, calendar, universe, as_of=args.as_of)
    args.output.mkdir(parents=True)
    series.to_parquet(args.output / "market-history.parquet")
    (args.output / "context.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("expected_session", "latest_observed", "requested", "observed", "coverage")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
