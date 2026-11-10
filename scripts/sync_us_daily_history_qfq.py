"""Checkpointed QFQ US history download into an isolated corrected-data directory."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.data import FutuOpenDReadOnly

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=1.05)
    args = parser.parse_args()
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    base = ROOT / config.output_root
    universe_path = max((base / "universe").glob("top1000-*.parquet"))
    universe = pd.read_parquet(universe_path).sort_values("rank")
    if args.limit:
        universe = universe.head(args.limit)
    output = base / "daily_qfq"
    output.mkdir(parents=True, exist_ok=True)
    end = datetime.now(UTC).date()
    start = end - timedelta(days=365 * 3 + 30)
    adapter = FutuOpenDReadOnly(config.opend_host, config.opend_port)
    completed, skipped = 0, 0
    failures: list[dict[str, str]] = []
    for row in universe.to_dict("records"):
        symbol = str(row["code"])
        target = output / f"{symbol.replace('.', '_')}.parquet"
        if target.exists():
            skipped += 1
            continue
        try:
            frame = adapter.history_kline(
                symbol, start.isoformat(), end.isoformat(), "K_DAY", autype="qfq"
            )
            if frame.empty:
                raise RuntimeError("empty QFQ history")
            frame["symbol"] = symbol
            frame["adjustment"] = "QFQ"
            frame["captured_at_utc"] = datetime.now(UTC).isoformat()
            temporary = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(target)
            completed += 1
        except Exception as exc:  # noqa: BLE001 - checkpoint every symbol failure
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        processed = completed + skipped + len(failures)
        if processed % 25 == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "downloaded": completed,
                        "skipped": skipped,
                        "failures": len(failures),
                    }
                ),
                flush=True,
            )
        time.sleep(args.sleep)
    manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "adjustment": "QFQ",
        "source_universe": str(universe_path),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "requested": len(universe),
        "downloaded": completed,
        "skipped": skipped,
        "failures": failures,
        "known_limitation": "current-universe membership; historical PIT universe pending",
        "real_orders_enabled": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
