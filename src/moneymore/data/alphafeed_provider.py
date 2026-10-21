"""Read-only AlphaFeed quotes. No QMT connection, orders, or training-data writes."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime
from datetime import time as clock
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import dotenv_values

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _number(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite AlphaFeed value")
    return result


def normalize_quote(
    quote: dict,
    depth: dict | None,
    instrument: dict | None,
    now: datetime,
    *,
    volume_multiplier: float = 100,
) -> dict:
    """Retain source times independently; missing/stale depth never means zero spread."""
    symbol = str(quote["symbol"])
    depth = depth or {}
    instrument = instrument or {}
    ext = instrument.get("ext") or {}
    reasons = []
    price = _number(quote["last_price"])
    opening = _number(quote["open"])
    high, low = _number(quote["high"]), _number(quote["low"])
    volume, amount = _number(quote["volume"]), _number(quote["amount"])
    stamp = int(quote["timestamp"])
    depth_stamp = int(depth.get("timestamp") or 0)
    local = now.astimezone(SHANGHAI)
    observed = int(local.timestamp() * 1000)
    quote_time = datetime.fromtimestamp(stamp / 1000, SHANGHAI)
    age = (observed - stamp) / 1000
    depth_age = (observed - depth_stamp) / 1000 if depth_stamp else None
    bids, asks = depth.get("bid_prices") or [], depth.get("ask_prices") or []
    bid = _number(bids[0]) if bids else None
    ask = _number(asks[0]) if asks else None
    if price <= 0 or opening <= 0 or not low <= min(price, opening) <= max(price, opening) <= high:
        reasons.append("INVALID_OHLC")
    if volume <= 0 or amount <= 0 or volume_multiplier <= 0:
        reasons.append("INVALID_VOLUME_AMOUNT")
        vwap = None
    else:
        vwap = amount / (volume * volume_multiplier)
        if not low - 1e-6 <= vwap <= high + 1e-6:
            reasons.append("VWAP_OUTSIDE_RANGE")
    if depth.get("symbol") != symbol or instrument.get("symbol") != symbol:
        reasons.append("MISSING_OR_MISMATCHED_METADATA")
    if bid is None or ask is None or bid <= 0 or ask <= 0 or bid > ask:
        reasons.append("INVALID_BOOK")
    if not (0 <= age <= 30):
        reasons.append("QUOTE_NOT_FRESH")
    if depth_age is None or not (0 <= depth_age <= 30):
        reasons.append("DEPTH_NOT_FRESH")
    if not depth_stamp or abs(stamp - depth_stamp) > 5000:
        reasons.append("QUOTE_DEPTH_SKEW")
    if quote_time.date() != local.date():
        reasons.append("WRONG_SESSION_DATE")
    if local.weekday() >= 5 or not (
        clock(9, 30) <= local.time() <= clock(11, 30) or clock(13) <= local.time() < clock(15)
    ):
        reasons.append("OUTSIDE_CONTINUOUS_SESSION")
    up, down = float(ext.get("limit_up") or 0), float(ext.get("limit_down") or 0)
    if not (0 < down < up):
        reasons.append("MISSING_PRICE_LIMITS")
    return {
        "symbol": symbol,
        "name": instrument.get("name", ""),
        "timestamp": stamp,
        "depth_timestamp": depth_stamp,
        "observed_at": local.isoformat(),
        "last": price,
        "open": opening,
        "high": high,
        "low": low,
        "pre_close": _number(quote["prev_close"]),
        "volume": volume,
        "volume_unit": "lot" if volume_multiplier == 100 else "custom",
        "volume_shares": volume * volume_multiplier,
        "amount": amount,
        "vwap": vwap,
        "bid1": bid,
        "ask1": ask,
        "up_limit": up,
        "down_limit": down,
        "price_tick": ext.get("tick_size"),
        "source": "ALPHAFEED_REALTIME",
        "quote_age_seconds": age,
        "depth_age_seconds": depth_age,
        "execution_ready": not reasons,
        "quality_reasons": ",".join(reasons),
    }


class AlphaFeedProvider:
    name = "alphafeed"

    def __init__(self, root: str | Path | None = None, *, client=None, now=None):
        self.root = Path(root or Path(__file__).resolve().parents[3])
        self._now = now or (lambda: datetime.now(SHANGHAI))
        self._lock = threading.Lock()
        self._cached = None
        self._metadata = {}
        self._sample_count = 0
        if client is None:
            from alphafeed import AlphaFeed

            key = os.getenv("ALPHAFEED_API_KEY") or dotenv_values(
                self.root / "state/private/alphafeed.env"
            ).get("ALPHAFEED_API_KEY")
            if not key:
                raise RuntimeError("AlphaFeed private key is not configured")
            client = AlphaFeed(
                api_key=key,
                timeout=10,
                max_retries=0,
                cache_dir=str(self.root / "state/private/alphafeed-cache"),
            )
        self._client = client

    def intraday_quotes(self, symbols: list[str]) -> pd.DataFrame:
        symbols = sorted(set(symbols))
        if not symbols:
            return pd.DataFrame()
        if len(symbols) > 1000:
            raise ValueError("AlphaFeed real-time request exceeds 1000 symbols")
        with self._lock:
            if (
                self._cached
                and self._cached[0] == symbols
                and time.monotonic() - self._cached[1] < 2
            ):
                raw, depths = self._cached[2:]
            else:
                raw, depths = [], {}
                for start in range(0, len(symbols), 100):
                    batch = symbols[start : start + 100]
                    raw.extend(self._client.quotes.get(symbols=batch))
                    depths.update(self._client.depth.batch(batch, max_workers=1))
                    missing = [
                        s
                        for s in batch
                        if s not in self._metadata or time.monotonic() - self._metadata[s][0] > 300
                    ]
                    if missing:
                        for item in self._client.instruments.get(missing):
                            self._metadata[item["symbol"]] = (time.monotonic(), item)
                self._cached = (symbols, time.monotonic(), raw, depths)
            rows = []
            for item in raw:
                try:
                    rows.append(
                        normalize_quote(
                            item,
                            depths.get(item["symbol"]),
                            self._metadata.get(item["symbol"], (0, {}))[1],
                            self._now(),
                        )
                    )
                except (KeyError, ValueError, TypeError, OverflowError):
                    # Malformed rows must not participate in valuation or execution.
                    continue
            frame = pd.DataFrame(rows)
            self._sample_count += 1
            status = {
                "source": "ALPHAFEED_REALTIME",
                "observed_at": self._now().isoformat(),
                "sample_count_this_process": self._sample_count,
                "requested_symbols": len(symbols),
                "returned_symbols": len(frame),
                "missing_symbols": sorted(set(symbols) - {row["symbol"] for row in rows}),
                "execution_ready_count": sum(row["execution_ready"] for row in rows),
                "quality_reasons": sorted(
                    {
                        reason
                        for row in rows
                        for reason in row["quality_reasons"].split(",")
                        if reason
                    }
                ),
                "max_quote_age_seconds": max(
                    (row["quote_age_seconds"] for row in rows), default=None
                ),
                "max_depth_age_seconds": max(
                    (
                        row["depth_age_seconds"]
                        for row in rows
                        if row["depth_age_seconds"] is not None
                    ),
                    default=None,
                ),
            }
            path = self.root / "state/runtime/alphafeed-health.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            return frame

    def minute_history(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        start_dt, end_dt = (
            pd.Timestamp(start, tz="Asia/Shanghai"),
            pd.Timestamp(end, tz="Asia/Shanghai"),
        )
        if (
            end_dt < start_dt
            or (end_dt.to_pydatetime() - start_dt.to_pydatetime()).total_seconds() > 604800
            or len(symbols) > 100
        ):
            raise ValueError("Minute probe is limited to 7 days and 100 symbols")
        frames = []
        with self._lock:
            for symbol in sorted(set(symbols)):
                frame = self._client.klines.get(
                    symbol,
                    period="1m",
                    adjust="none",
                    count=2000,
                    start_time=int(start_dt.timestamp() * 1000),
                    end_time=int(end_dt.timestamp() * 1000),
                    to_dataframe=True,
                )
                if frame.empty:
                    continue
                frame = frame.copy()
                if len(frame) >= 2000:
                    raise ValueError("Minute result may be truncated; split the date interval")
                frame["symbol"] = symbol
                times = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert(
                    SHANGHAI
                )
                frame = frame.loc[(times >= start_dt) & (times <= end_dt)].copy()
                frame["source"] = "ALPHAFEED_MINUTE"
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
