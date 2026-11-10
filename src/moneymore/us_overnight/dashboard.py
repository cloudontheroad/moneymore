from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import exchange_calendars as xcals

from .config import USOvernightConfig
from .data import FutuOpenDReadOnly


def read_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def dashboard_snapshot(root: str | Path, session: str | None = None) -> dict[str, object]:
    root = Path(root)
    config = USOvernightConfig.from_yaml(root / "configs/us_overnight_paper_only.yaml")
    candidate_dir = root / config.output_root / "open2open_v1" / "plans"
    book_dir = candidate_dir.parent
    book = read_record(root / "configs/us_open2open_paper_book.json")
    universe_dir = root / config.output_root / "universe"
    now = pd.Timestamp.now(tz=config.timezone)
    calendar = xcals.get_calendar("XNYS")
    expected = calendar.date_to_session(now.date(), direction="next")
    if now >= calendar.session_close(expected):
        expected = calendar.next_session(expected)
    current = candidate_dir / f"{expected.date()}.json"
    latest = max(
        (path for path in candidate_dir.glob("*.json") if path.stem <= str(expected.date())),
        default=None,
    )
    plan = json.loads(latest.read_text(encoding="utf-8")) if latest else {}
    status = plan.get("status", "WAITING_FOR_FIRST_RUN")
    if latest and latest != current:
        status = "STALE"
    elif status == "READY" and now >= pd.Timestamp(plan["cutoff"]):
        status = "EXPIRED"
    recorded = {
        path.stem: path for path in candidate_dir.glob("*.json")
        if book.get("start_session", "9999") <= path.stem <= str(expected.date())
    }
    selected_day = session or max(recorded, default=str(expected.date()))
    historical_plan = read_record(recorded[selected_day]) if selected_day in recorded else {}
    ledger = None
    if historical_plan:
        entry = read_record(book_dir / "entries" / f"{selected_day}.json")
        settled = read_record(book_dir / "settled" / f"{selected_day}.json")
        ledger = {
            "trade_session": selected_day,
            "exit_session": historical_plan.get("exit_session"),
            "status": "SETTLED" if settled else (
                "OPEN" if entry and historical_plan["status"] == "READY" else historical_plan["status"]
            ),
            "positions": settled.get("realized", entry.get("positions", historical_plan.get("candidates", []))),
            "daily_return": settled.get("daily_return"),
            "pnl_usd": settled.get("pnl_usd"),
        }
    universe_files = sorted(universe_dir.glob("top1000-*.parquet"), reverse=True)
    universe = pd.read_parquet(universe_files[0]) if universe_files else pd.DataFrame()
    return {
        "mode": config.mode,
        "status": status,
        "latest_session": plan.get("trade_session"),
        "expected_session": str(expected.date()),
        "generated_at": plan.get("generated_at"),
        "reason": plan.get("reason"),
        "market_forecast_bps": plan.get("market_forecast_bps"),
        "paper_book": book | {
            "available_sessions": sorted(recorded, reverse=True), "selected_session": selected_day,
            "record": ledger,
        },
        "server_time": datetime.now(ZoneInfo(config.timezone)).isoformat(),
        "schedule": {
            "timezone": config.timezone,
            "signal_cutoff": (calendar.session_open(expected) - pd.Timedelta(minutes=5))
                .tz_convert(config.timezone).strftime("%H:%M"),
            "entry_window": ["09:30"],
            "exit_window": ["09:30"],
        },
        "policy": {
            "top_n": config.top_n,
            "round_trip_cost_bps": config.round_trip_cost_bps,
            "minimum_price": config.minimum_price,
            "minimum_median_dollar_volume_20d": config.minimum_median_dollar_volume_20d,
            "minimum_listing_days": config.minimum_listing_days,
            "real_orders_enabled": False,
        },
        "candidates": plan.get("candidates", []) if status == "READY" else [],
        "universe": {
            "size": len(universe),
            "minimum_market_cap": (
                float(universe["market_cap"].min()) if not universe.empty else None
            ),
            "snapshot": universe_files[0].stem if universe_files else None,
        },
        "opend": FutuOpenDReadOnly(config.opend_host, config.opend_port).probe(timeout=0.25),
    }
