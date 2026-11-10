"""Download split-adjusted Yahoo OHLCV plus dividends for the US research universe."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from moneymore.us_overnight.config import USOvernightConfig

ROOT = Path(__file__).resolve().parents[1]


def yahoo_symbol(code: str) -> str:
    return code.removeprefix("US.").replace(".", "-")


def download(symbol: str, start: str, end: str, timeout: int = 30) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    encoded = urllib.parse.quote(yahoo_symbol(symbol), safe="")
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?period1={period1}&period2={period2}&interval=1d&events=div%2Csplits"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 MoneyMore/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(
        "America/New_York"
    )
    dates = timestamps.strftime("%Y-%m-%d")
    dividends = {
        pd.Timestamp(int(item["date"]), unit="s", tz="UTC")
        .tz_convert("America/New_York")
        .strftime("%Y-%m-%d"): float(item["amount"])
        for item in result.get("events", {}).get("dividends", {}).values()
    }
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates).tz_localize("America/New_York").tz_convert("UTC"),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
            "cash_dividend": [dividends.get(day, 0.0) for day in dates],
            "symbol": symbol,
            "source": "YAHOO_SPLIT_ADJUSTED",
        }
    )
    return frame.dropna(subset=["open", "close"]).drop_duplicates("timestamp", keep="last")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--context-only", action="store_true")
    args = parser.parse_args()
    if args.context_only:
        from build_us_overnight_factor_panel import MARKET, SECTORS

        cache = ROOT / "data/us_overnight_corrected/features/context"
        cache.mkdir(parents=True, exist_ok=True)
        end = datetime.now(UTC).date()
        start = end - timedelta(days=730)
        for symbol in [*MARKET, *SECTORS]:
            frame = download(f"US.{symbol}", start.isoformat(), end.isoformat())
            if frame.empty:
                raise ValueError(f"Missing market context for {symbol}")
            path = cache / f"{symbol.replace('^', 'IDX_')}.parquet"
            temp = path.with_suffix(".parquet.tmp")
            frame.to_parquet(temp, index=False)
            temp.replace(path)
        return 0
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    base = ROOT / config.output_root
    universe_path = max((base / "universe").glob("top1000-*.parquet"))
    universe = pd.read_parquet(universe_path).sort_values("rank")
    if args.limit:
        universe = universe.head(args.limit)
    output = base / "daily_yahoo_adjusted"
    output.mkdir(parents=True, exist_ok=True)
    end = datetime.now(UTC).date()
    start = end - timedelta(days=365 * 3 + 60)
    pending = []
    skipped = 0
    for symbol in universe["code"].astype(str):
        target = output / f"{symbol.replace('.', '_')}.parquet"
        if target.exists() and not args.refresh:
            skipped += 1
        else:
            pending.append((symbol, target))
    failures: list[dict[str, str]] = []
    completed = 0

    def fetch(item: tuple[str, Path]) -> tuple[str, Path, pd.DataFrame]:
        symbol, target = item
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                return symbol, target, download(symbol, start.isoformat(), end.isoformat())
            except Exception as exc:  # noqa: BLE001 - bounded network retry
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"{symbol}: {last_error}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, item): item[0] for item in pending}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                _, target, frame = future.result()
                temporary = target.with_suffix(".parquet.tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(target)
                completed += 1
            except Exception as exc:  # noqa: BLE001 - retain per-symbol failure
                failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            processed = completed + len(failures)
            if processed % 50 == 0:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "downloaded": completed,
                            "failures": len(failures),
                        }
                    ),
                    flush=True,
                )
    manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source": "Yahoo chart split-adjusted OHLCV plus cash dividends",
        "source_universe": str(universe_path),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested": len(universe),
        "downloaded": completed,
        "skipped": skipped,
        "failures": failures,
        "feature_timing": "all model predictors lagged one completed session",
        "known_limitation": "current-universe membership; historical PIT universe pending",
        "real_orders_enabled": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
