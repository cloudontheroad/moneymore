from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SDK = Path(
    os.getenv("MONEYMORE_XTQUANT_SDK", ROOT / ".runtime" / "xtquant_250807")
)
sys.path.insert(0, str(SDK))

from moneymore.data.qmt_supplement import normalize_qmt_supplement
from moneymore.data.store import ParquetStore


def _active_symbols(as_of: str, limit: int | None) -> list[str]:
    universe = pd.read_parquet(
        ROOT / "data" / "processed" / "strategy_universe.parquet",
        columns=["effective_date", "rank", "symbol"],
    )
    universe["effective_date"] = universe["effective_date"].astype(str)
    eligible = universe.loc[universe["effective_date"] <= as_of]
    effective = eligible["effective_date"].max()
    symbols = eligible.loc[eligible["effective_date"] == effective].sort_values("rank")
    if limit is not None:
        symbols = symbols.head(limit)
    return symbols["symbol"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20150101")
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    from xtquant import xtdata

    xtdata.enable_hello = False
    with contextlib.redirect_stdout(io.StringIO()):
        client = xtdata.connect()
    if not client:
        raise RuntimeError("unable to connect to MiniQMT")

    symbols = _active_symbols(args.end, args.limit)
    collected: dict[str, dict[str, pd.DataFrame]] = {}
    for offset in range(0, len(symbols), args.batch_size):
        chunk = symbols[offset : offset + args.batch_size]
        xtdata.download_financial_data2(chunk, [], args.start, args.end)
        collected.update(xtdata.get_financial_data(chunk, [], args.start, args.end))
        print(f"qmt_financial_progress={min(offset + len(chunk), len(symbols))}/{len(symbols)}", flush=True)

    tables = normalize_qmt_supplement(collected)
    store = ParquetStore(args.data_dir)
    snapshot = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    keys = {
        "qmt_financial_indicator": ["ts_code", "ann_date", "end_date"],
        "qmt_holder_count": ["ts_code", "ann_date", "end_date"],
        "qmt_top10_concentration": [
            "ts_code",
            "ann_date",
            "end_date",
            "holder_type",
        ],
    }
    for table, frame in tables.items():
        if frame.empty:
            print(f"table={table} rows=0")
            continue
        store.save_snapshot(table, frame, "qmt", keys[table], snapshot)
        print(f"table={table} rows={len(frame)}")


if __name__ == "__main__":
    main()
