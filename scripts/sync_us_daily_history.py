"""Checkpointed OpenD daily-history download for the current US top-cap universe."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from moneymore.us_overnight.config import USOvernightConfig
from moneymore.us_overnight.data import FutuOpenDReadOnly

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = USOvernightConfig.from_yaml(ROOT / "configs/us_overnight_paper_only.yaml")
    universe_files = sorted((ROOT / config.output_root / "universe").glob("*.parquet"))
    if not universe_files:
        raise FileNotFoundError("run scripts/sync_us_top1000.py first")
    universe = pd.read_parquet(universe_files[-1]).sort_values("rank")
    output = ROOT / config.output_root / "daily"
    output.mkdir(parents=True, exist_ok=True)
    end = datetime.now(UTC).date()
    start = end - timedelta(days=365 * 3 + 30)
    adapter = FutuOpenDReadOnly(config.opend_host, config.opend_port)
    completed = 0
    failures: list[dict[str, str]] = []
    for row in universe.to_dict("records"):
        symbol = str(row["code"])
        target = output / f"{symbol.replace('.', '_')}.parquet"
        if target.exists():
            completed += 1
            continue
        try:
            frame = adapter.history_kline(symbol, start.isoformat(), end.isoformat(), "K_DAY")
            frame["symbol"] = symbol
            frame["captured_at_utc"] = datetime.now(UTC).isoformat()
            temporary = target.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(target)
            completed += 1
        except Exception as exc:  # noqa: BLE001 - record and continue the universe
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        if (completed + len(failures)) % 25 == 0:
            print(
                json.dumps(
                    {
                        "processed": completed + len(failures),
                        "saved": completed,
                        "failures": len(failures),
                    }
                ),
                flush=True,
            )
        time.sleep(1.05)
    manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "universe": universe_files[-1].name,
        "saved": completed,
        "failures": failures,
        "real_orders_enabled": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
