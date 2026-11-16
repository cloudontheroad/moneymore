"""Automatically refreshed V4 experiment, with an isolated read-only replay."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread

import exchange_calendars as xcals
import pandas as pd

from .leveraged_etf import INITIAL_CASH, NEW_YORK, LeveragedEtfPaper, YahooDailyClient
from .tqqq_v4 import SECTORS, v4_targets

V4_ID = "cycle_confirmed_exit_v4"
LOG = logging.getLogger(__name__)


def build_snapshot(end, next_session):
    symbols = ("QQQ", "TQQQ", "SQQQ", "HYG", "IEI", "^VIX", "^VIX3M", "^TNX", "SOXX", "SPY",
               "RSP", "RSPD", "RSPS", *SECTORS)
    client = YahooDailyClient()

    def fetch(symbol, adjusted):
        frame = client.history(symbol, "2010-02-11", end, adjusted=adjusted).set_index("trade_date")
        frame.index = pd.to_datetime(frame.index)
        return frame.sort_index()

    with ThreadPoolExecutor(max_workers=6) as pool:
        inputs = {s: pool.submit(fetch, s, not s.startswith("^")) for s in symbols if s != "SQQQ"}
        raw_inputs = {s: pool.submit(fetch, s, False) for s in ("QQQ", "TQQQ", "SQQQ")}
        adjusted = {s: item.result() for s, item in inputs.items()}
        raw = {s: item.result() for s, item in raw_inputs.items()}

    history = pd.concat([
        adjusted[s][["open", "close"]].rename(columns={"open": s.lower() + "Open", "close": s.lower() + "Close"})
        for s in ("QQQ", "TQQQ")
    ], axis=1, join="inner").loc[:end]
    if len(history) < 504 or history.index[-1] != pd.Timestamp(end):
        raise ValueError("The completed QQQ/TQQQ session is unavailable")
    close = pd.DataFrame({s: frame.close.reindex(history.index) for s, frame in adjusted.items()})
    for symbol in ("HYG", "IEI", "VIX", "VIX3M"):
        history[symbol] = close[symbol if not symbol.startswith("VIX") else "^" + symbol]
    if close.loc[end].isna().any():
        raise ValueError("V4 confirmation data for the latest completed session is unavailable")
    weights = v4_targets(history, close, close, close, next_session=next_session)
    # The shared replay takes close decisions. V4 allocates at the following open.
    targets = weights.shift(-1).reindex(history.index)
    decisions = pd.DataFrame({"tqqq": targets, "regime": targets.map({0.: "CASH", 1.: "TQQQ"})})
    market = {}
    for symbol, frame in raw.items():
        aligned = frame.reindex(history.index)
        aligned.index = aligned.index.strftime("%Y-%m-%d").rename("trade_date")
        market[symbol] = aligned.reset_index()
    year_start = f"{end[:4]}-01-01"
    indices = list(range(history.index.searchsorted(year_start), len(history)))
    for frame in market.values():
        if frame.loc[indices, ["open", "close"]].isna().any().any():
            raise ValueError("The V4 replay is missing execution prices")
    rows = LeveragedEtfPaper.replay(market, decisions, indices, year_start, drift_band=2.)
    target = float(weights.iloc[-1])
    equity = rows[-1]["equity"]
    return {
        "account_id": V4_ID, "strategy_id": V4_ID, "as_of": end,
        "equity": equity, "initial_cash": INITIAL_CASH, "backtest_history": rows,
        "latest_signal": {"signal_date": end, "regime": "TQQQ" if target else "CASH",
                          "tqqq": target, "cash": 1. - target},
    }


class LeveragedEtfV4Experiment:
    def __init__(self, cache: Path):
        self.cache = cache
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.payload = None

    def snapshot(self):
        return self.payload

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            self.payload = json.loads(self.cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.stop_event.clear()
        self.thread = Thread(target=self._run, name="tqqq-v4-experiment", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):
        calendar = xcals.get_calendar("XNYS")
        while not self.stop_event.is_set():
            try:
                now = datetime.now(NEW_YORK)
                session = calendar.date_to_session(now.date(), direction="previous")
                if now < calendar.session_close(session) + timedelta(minutes=15):
                    session = calendar.previous_session(session)
                day = session.strftime("%Y-%m-%d")
                if not self.payload or self.payload.get("as_of", "") < day:
                    payload = build_snapshot(day, calendar.next_session(session))
                    self.cache.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.cache.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
                    temporary.replace(self.cache)
                    self.payload = payload
            except Exception:
                LOG.exception("V4 research replay refresh failed")
            self.stop_event.wait(300)
