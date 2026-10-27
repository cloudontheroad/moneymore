"""Bounded financial probes; immutable isolated snapshots, no live-table replacement."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--problem-set", action="store_true")
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    store = ParquetStore(root / "data")
    provider = TushareProvider()
    if args.problem_set:
        from moneymore.data.fundamental_research import financial_quality
        from moneymore.strategy_universe import active_strategy_universe

        quality = financial_quality(
            store.read("fina_indicator_research_v2"),
            active_strategy_universe(store).symbol.tolist(),
            args.as_of,
            per_field=True,
        )
        args.symbols = sorted(
            set(quality["missing_symbols"] + quality["stale_symbols"] + ["600519.SH"])
        )
    if not args.symbols or len(args.symbols) > 40:
        raise ValueError("One to forty probe symbols required")
    for symbol in args.symbols:
        for endpoint in ("fina_indicator", "income"):
            previous = list(
                (store.raw / "tushare" / (endpoint + "_deep_probe")).glob(
                    f"{symbol.replace('.', '_')}_{args.as_of}*.parquet"
                )
            )
            if previous:
                print(
                    json.dumps(
                        {"symbol": symbol, "endpoint": endpoint, "status": "REUSED_CAPTURE"}
                    ),
                    flush=True,
                )
                continue
            try:
                if endpoint == "income":
                    frame = provider.income_statement(symbol, "20230101", args.as_of)
                    keys = ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type"]
                else:
                    frame = provider._client.fina_indicator(
                        ts_code=symbol,
                        start_date="20230101",
                        end_date=args.as_of,
                        fields="ts_code,ann_date,end_date,roe,q_sales_yoy,or_yoy,q_gr_yoy,update_flag",
                    )
                    keys = ["ts_code", "ann_date", "end_date"]
                captured = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                if not frame.empty:
                    if not frame.ts_code.eq(symbol).all():
                        raise ValueError("Unexpected returned symbol")
                    store.save_snapshot(
                        endpoint + "_deep_probe",
                        frame,
                        "tushare",
                        keys,
                        f"{symbol.replace('.', '_')}_{captured}",
                    )
                cols = [
                    c
                    for c in [
                        "end_date",
                        "ann_date",
                        "f_ann_date",
                        "report_type",
                        "comp_type",
                        "roe",
                        "q_sales_yoy",
                        "or_yoy",
                        "revenue",
                        "total_revenue",
                    ]
                    if c in frame
                ]
                print(
                    json.dumps(
                        {
                            "symbol": symbol,
                            "endpoint": endpoint,
                            "rows": len(frame),
                            "latest": frame.sort_values(["end_date", "ann_date"], ascending=False)[
                                cols
                            ]
                            .head(6)
                            .to_dict("records"),
                        },
                        default=str,
                    ),
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - bounded probe stops on first error
                print(
                    json.dumps(
                        {"symbol": symbol, "endpoint": endpoint, "error_type": type(error).__name__}
                    ),
                    flush=True,
                )
                raise SystemExit(1) from None
            time.sleep(1.2)
