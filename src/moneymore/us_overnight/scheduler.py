"""Local service runner for the pre-open paper book and opening-price feedback."""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import exchange_calendars as xcals
import pandas as pd
from .dashboard import read_record


class OpenToOpenScheduler:
    def __init__(self, root: Path):
        self.root = root
        self.stop_event = Event()
        self.thread: Thread | None = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        if not (self.root / "configs/us_open2open_paper_book.json").is_file() or not (
            self.root / "data/us_overnight_corrected/features/historical_panel_bundle_investable_120d_20m_5.parquet"
        ).is_file():
            return
        self.stop_event.clear()
        self.thread = Thread(target=self.run, name="us-open2open-paper", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def command(self, script, *args, timeout=900):
        try:
            return subprocess.run(
                [sys.executable, str(self.root / "scripts" / script), *args],
                cwd=self.root, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def run(self):
        log = logging.getLogger(__name__)
        cal = xcals.get_calendar("XNYS")
        last_refresh = None
        last_attempt = None
        book = self.root / "data/us_overnight_paper_only/open2open_v1"
        while not self.stop_event.is_set():
            try:
                now = pd.Timestamp.now(tz="UTC")
                day = now.tz_convert("America/New_York").date()
                if cal.is_session(day):
                    opening = cal.session_open(day)
                    if now >= opening + pd.Timedelta(minutes=2):
                        due = False
                        for path in (book / "plans").glob("*.json"):
                            if path.stem > str(day) or (book / "settled" / path.name).is_file():
                                continue
                            plan = read_record(path)
                            if plan.get("status") in {"READY", "CASH"} and (
                                plan["exit_session"] <= str(day) or not (book / "entries" / path.name).is_file()
                            ):
                                due = True
                                break
                        if due:
                            self.command("run_us_open2open.py", "settle", timeout=240)
                    elif (opening - pd.Timedelta(hours=3) <= now < opening - pd.Timedelta(minutes=12)
                          and (last_attempt is None or now - last_attempt >= pd.Timedelta(minutes=20))):
                        last_attempt = now
                        plan = read_record(book / "plans" / f"{day}.json")
                        if plan.get("status") in {"READY", "CASH"}:
                            self.stop_event.wait(120)
                            continue
                        if last_refresh != day:
                            universe = self.root / "data/us_overnight_paper_only/universe"
                            snapshots = sorted(universe.glob("top1000-*.parquet"))
                            captured = pd.to_datetime(
                                snapshots[-1].stem.removeprefix("top1000-"),
                                format="%Y-%m-%dT%H%M%SZ", utc=True,
                            ) if snapshots else None
                            if captured is None or now - captured >= pd.Timedelta(days=5):
                                if not self.command("sync_us_top1000.py", timeout=180):
                                    log.warning("Could not refresh the US market-cap snapshot")
                            self.command("sync_us_yahoo_adjusted_history.py", "--refresh", "--workers", "16")
                            if self.command("sync_us_yahoo_adjusted_history.py", "--context-only"):
                                last_refresh = day
                        self.command("run_us_open2open.py", "score", timeout=480)
            except Exception:
                log.exception("US paper book scheduler tick failed")
            self.stop_event.wait(120)
