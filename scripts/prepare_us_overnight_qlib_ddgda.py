"""Prepare a local Qlib calendar and static 300-feature handler for DDG-DA."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_open_to_open/features/alpha360/panel.parquet"
FEATURES = ROOT / "data/us_overnight_open_to_open/features/alpha360/features.json"
OUT = ROOT / "state/us-overnight-open2open-v1/ddgda"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    provider = OUT / "provider"
    (provider / "calendars").mkdir(parents=True, exist_ok=True)
    (provider / "instruments").mkdir(parents=True, exist_ok=True)
    features = [
        name for name in json.loads(FEATURES.read_text(encoding="utf-8"))
        if not name.startswith("VWAP")
    ]
    panel = pd.read_parquet(PANEL, columns=["timestamp", "symbol", "label_next_open_net", *features])
    panel = panel.dropna(subset=["label_next_open_net"]).copy()
    panel["datetime"] = pd.to_datetime(panel["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    panel["instrument"] = panel["symbol"]
    panel = panel.set_index(["datetime", "instrument"]).sort_index()
    feature_frame = panel[features].astype("float32")
    ranked_label = panel.groupby(level="datetime")["label_next_open_net"].rank(pct=True)
    label_frame = ((ranked_label - 0.5) * 3.46).astype("float32").to_frame("LABEL0")
    data = pd.concat({"feature": feature_frame, "label": label_frame}, axis=1)
    static_path = OUT / "alpha300_static.pkl"
    data.to_pickle(static_path)
    handler = DataHandlerLP(
        data_loader={
            "class": "qlib.data.dataset.loader.StaticDataLoader",
            "kwargs": {"config": static_path},
        }
    )
    handler_path = OUT / "alpha300_handler.pkl"
    handler.to_pickle(handler_path, dump_all=True)
    dates = sorted(data.index.get_level_values("datetime").unique())
    # Qlib's official RollingGen truncates every non-test segment relative to
    # the test boundary and requires the training start to be after calendar
    # index zero.  Keep a small calendar-only prefix; it creates no samples.
    calendar_dates = sorted(
        set(pd.bdate_range(end=dates[0] - pd.Timedelta(days=1), periods=5)).union(dates)
    )
    symbols = sorted(data.index.get_level_values("instrument").unique())
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(date.strftime("%Y-%m-%d") for date in calendar_dates) + "\n", encoding="utf-8"
    )
    start, end = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    (provider / "instruments" / "all.txt").write_text(
        "\n".join(f"{symbol}\t{start}\t{end}" for symbol in symbols) + "\n",
        encoding="utf-8",
    )
    audit = {
        "rows": len(data), "symbols": len(symbols), "sessions": len(dates),
        "calendar_prefix_sessions": len(calendar_dates) - len(dates),
        "features": len(features), "removed_constant_vwap_features": 60,
        "training_label": "Qlib CSRankNorm; raw economic return retained in source panel for evaluation",
        "handler": str(handler_path), "provider": str(provider),
    }
    (OUT / "prepare_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
