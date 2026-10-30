"""Bounded suspension evidence for a weekly context; never invent missing bars."""
import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from moneymore.data.tushare_provider import TushareProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    symbols = context["missing_symbols"]
    if len(symbols) > 10:
        raise ValueError("missing-symbol request budget exceeded")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    frames = []
    if symbols:
        provider = TushareProvider()
        for symbol in symbols:
            day = context["expected_session"]
            frames.append(provider.suspensions(symbol, day, day))
    proof = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ts_code", "trade_date", "suspend_type", "suspend_timing"])
    proof.to_parquet(args.output)


if __name__ == "__main__":
    main()
