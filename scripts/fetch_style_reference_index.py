"""Single bounded official-index request; isolated evidence, no store mutation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from moneymore.data.tushare_provider import TushareProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable index evidence")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root/".env")
    try:
        frame = TushareProvider().index_daily("000300.SH", "20261012", "20261026")
        if frame is None or len(frame) != 11 or frame.trade_date.duplicated().any():
            raise ValueError("INCOMPLETE_INDEX_WINDOW")
        frame = frame.sort_values("trade_date")
        if not np.isfinite(frame[["open", "close"]].to_numpy()).all() or (frame[["open", "close"]] <= 0).any().any():
            raise ValueError("INVALID_INDEX_PRICES")
        path = args.output/"000300.SH.parquet"
        frame.to_parquet(path, index=False)
        # Match first investable open Oct13, not the cash-only initial close.
        index_return = float(frame.loc[frame.trade_date.eq("20261026"), "close"].iloc[0]/frame.loc[frame.trade_date.eq("20261013"), "open"].iloc[0]-1)
        report = {"status": "OFFICIAL_INDEX_PRICE_REFERENCE_AVAILABLE", "rows": len(frame), "orders_enabled": False,
            "price_return_from_first_investable_open": index_return, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "limitations": ["HS300 differs from study candidate pool", "price index, not dividend-total-return or an executable account", "not an alpha significance test"]}
    except Exception as error:
        report = {"status": "BLOCKED_OFFICIAL_INDEX_READ", "error_type": type(error).__name__, "orders_enabled": False,
            "provider_error_body_suppressed": True, "attempts": 1}
    (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
