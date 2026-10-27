"""Classify blocked risk histories using official listing and suspension proof."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.style_feasibility import classify_history_gaps


def run(root):
    folder = root / "state/enhanced-indexing-research"
    replay = json.loads((folder / "continuous-replay.json").read_text(encoding="utf-8"))
    missing = {}
    for row in replay["runs"]:
        for event in row.get("daily_optimizer_audit", []):
            for symbol, dates in event.get("missing_return_dates", {}).items():
                missing.setdefault(symbol, set()).update(dates)
    if len(missing) > 10:
        raise ValueError("BOUNDED_RESEARCH_LIMIT_10_SYMBOLS")
    store = ParquetStore(root / "data")
    instruments = store.read("instruments").set_index("ts_code")
    load_dotenv(root / ".env")
    from moneymore.data.tushare_provider import TushareProvider

    provider = TushareProvider()
    evidence = folder / "risk-history-evidence"
    evidence.mkdir(exist_ok=True)
    output = {
        "as_of_date": replay["as_of_date"],
        "orders_enabled": False,
        "source_listing_table_sha256": hashlib.sha256(
            (store.curated / "instruments.parquet").read_bytes()
        ).hexdigest(),
        "warning": "Missing return dates include the following day of a missing price. Pre-listing is never zero-filled.",
        "symbols": [],
    }
    for symbol, missing_dates in sorted(missing.items()):
        listing = str(instruments.loc[symbol, "list_date"]) if symbol in instruments.index else None
        path = evidence / (symbol + "-suspensions.parquet")
        dates = sorted(d for d in missing_dates if len(d) == 8 and d.isdigit())
        if not dates:
            output["symbols"].append({"symbol": symbol, "status": "NO_DATED_HISTORY"})
            continue
        if path.exists():
            proof = pd.read_parquet(path)
        else:
            proof = provider.suspensions(symbol, dates[0], replay["as_of_date"])
            proof.to_parquet(path, index=False)
            time.sleep(1.1)
        full = proof.loc[
            proof.suspend_type.eq("S") & (proof.suspend_timing.isna() | proof.suspend_timing.eq(""))
        ]
        confirmed = set(zip(full.ts_code, full.trade_date.astype(str)))
        # Audit prices at both ends, not only the return's end date.
        calendar = store.read("trade_calendar")
        days = sorted(
            calendar.loc[
                calendar.exchange.eq("SSE") & calendar.is_open.astype(int).eq(1), "cal_date"
            ]
            .astype(str)
            .unique()
        )
        price_dates = sorted({d for date in dates for d in [date, days[days.index(date) - 1]]})
        daily = store.read(
            "daily",
            filters=[
                ("ts_code", "==", symbol),
                ("trade_date", ">=", price_dates[0]),
                ("trade_date", "<=", price_dates[-1]),
            ],
        )
        adj = store.read(
            "adj_factor",
            filters=[
                ("ts_code", "==", symbol),
                ("trade_date", ">=", price_dates[0]),
                ("trade_date", "<=", price_dates[-1]),
            ],
        )
        good = set(daily.loc[daily.close.gt(0), "trade_date"]) & set(
            adj.loc[adj.adj_factor.gt(0), "trade_date"]
        )
        actual_missing = [d for d in price_dates if d not in good]
        output["symbols"].append(
            {
                "symbol": symbol,
                "official_list_date": listing,
                "missing_return_dates": dates,
                "missing_price_classification": classify_history_gaps(
                    symbol, actual_missing, listing, confirmed
                ),
                "suspension_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "suspension_source_file": str(path),
                "production_prices_changed": False,
            }
        )
    write_diagnostic(folder / "risk-history-audit.json", output)
    print(json.dumps(output["symbols"], ensure_ascii=True))
    return output


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
