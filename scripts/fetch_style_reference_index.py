"""Single bounded official-index request; isolated evidence, no store mutation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore

from moneymore.data.tushare_provider import TushareProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--start',default='20261012');parser.add_argument('--end',default='20261026');parser.add_argument('--episodes',type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable index evidence")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root/".env")
    try:
        calendar=trading_dates(ParquetStore(root/'data'),args.end);expected=[d for d in calendar if args.start<=d<=args.end]
        if not 1<len(expected)<=250:raise ValueError('BOUNDED_INDEX_WINDOW_REQUIRED')
        frame = TushareProvider().index_daily("000300.SH", args.start, args.end)
        if frame is None or set(frame.trade_date.astype(str))!=set(expected) or frame.trade_date.duplicated().any():
            raise ValueError("INCOMPLETE_INDEX_WINDOW")
        frame = frame.sort_values("trade_date")
        if not np.isfinite(frame[["open", "close"]].to_numpy()).all() or (frame[["open", "close"]] <= 0).any().any():
            raise ValueError("INVALID_INDEX_PRICES")
        path = args.output/"000300.SH.parquet"
        frame.to_parquet(path, index=False)
        # Match first investable open Oct13, not the cash-only initial close.
        index_return = float(frame.loc[frame.trade_date.eq(args.end), "close"].iloc[0]/frame.loc[frame.trade_date.eq(expected[1]), "open"].iloc[0]-1)
        report = {"status": "OFFICIAL_INDEX_PRICE_REFERENCE_AVAILABLE", "rows": len(frame), "orders_enabled": False,
            "price_return_from_first_investable_open": index_return, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "limitations": ["HS300 differs from study candidate pool", "price index, not dividend-total-return or an executable account", "not an alpha significance test"]}
        if args.episodes:
            indexed=frame.set_index('trade_date');report['episodes']=[{'fold':e['fold'],'start':e['execution_start'],'end':e['execution_end'],
                'open_to_close_price_return':float(indexed.loc[e['execution_end'],'close']/indexed.loc[e['execution_start'],'open']-1)} for e in json.loads(args.episodes.read_text())['episodes']]
    except Exception as error:
        report = {"status": "BLOCKED_OFFICIAL_INDEX_READ", "error_type": type(error).__name__, "orders_enabled": False,
            "provider_error_body_suppressed": True, "attempts": 1}
    (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
