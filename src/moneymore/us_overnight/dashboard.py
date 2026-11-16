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


def ledger_record(directory: Path, day: str, plan: dict) -> dict:
    entry = read_record(directory / "entries" / f"{day}.json")
    settled = read_record(directory / "settled" / f"{day}.json")
    cost = settled.get("cost_usd")
    if settled and cost is None:
        cost = sum(row.get("cost_usd", 0) for row in settled.get("realized", []))
    return {
        "trade_session": day,
        "exit_session": plan.get("exit_session"),
        "status": "SETTLED" if settled else (
            "OPEN" if entry and plan["status"] == "READY" else plan["status"]
        ),
        "positions": settled.get("realized", entry.get("positions", plan.get("candidates", []))),
        "daily_return": settled.get("daily_return"),
        "pnl_usd": settled.get("pnl_usd"),
        "cost_usd": cost,
        "fee_model": settled.get("fee_model", "legacy_bps" if settled else "futu_hk_us_fixed"),
    }


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
    comparison = None
    if historical_plan:
        ledger = ledger_record(book_dir, selected_day, historical_plan)
        review_dir = book_dir / "premarket"
        reviewed = read_record(review_dir / "plans" / f"{selected_day}.json")
        if reviewed:
            extra = (ledger_record(review_dir, selected_day, reviewed)
                     if reviewed["review_status"] == "CHANGED" else None)
            comparison = {
                "status": reviewed["review_status"], "added": reviewed["added"],
                "removed": reviewed["removed"], "quote_count": reviewed["quote_count"],
                "record": extra,
                "pnl_delta_usd": (extra["pnl_usd"] - ledger["pnl_usd"] if extra and
                                  extra["pnl_usd"] is not None and ledger["pnl_usd"] is not None else None),
                "return_delta": (extra["daily_return"] - ledger["daily_return"] if extra and
                                 extra["daily_return"] is not None and ledger["daily_return"] is not None else None),
            }
        elif historical_plan.get("review_pool"):
            decision = pd.Timestamp(historical_plan["cutoff"])
            comparison = {"status": "PENDING" if now < decision else "MISSED"}
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
            "record": ledger, "comparison": comparison,
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
