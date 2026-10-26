"""Read-only AlphaFeed quotes. No QMT connection, orders, or training-data writes."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import Counter
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

    def __init__(self, root: str | Path | None = None, *, client=None, now=None, monotonic=None):
        self.root = Path(root or Path(__file__).resolve().parents[3])
        self._now = now or (lambda: datetime.now(SHANGHAI))
        self._lock = threading.Lock()
        self._cached = None
        self._metadata = {}
        self._sample_count = 0
        self._monotonic = monotonic or time.monotonic
        self._retry_after = 0.0
        self._fetch_timings = []
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

    def _write_health(self, status: dict) -> None:
        path = self.root / "state/runtime/alphafeed-health.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        # Persist actual upstream observations, not every cached dashboard read.
        if status.get("fresh_fetch") or status.get("status") == "UNAVAILABLE":
            journal = path.parent / ("alphafeed-quality-" + self._now().strftime("%Y%m%d") + ".jsonl")
            with journal.open("a", encoding="utf-8") as target:
                target.write(json.dumps(status, ensure_ascii=False) + "\n")

    def _provider_failure(self, error):
        code = getattr(error, "status_code", None)
        cooldown = 300 if code in (401, 403) else 60 if code == 429 else 5
        self._retry_after = self._monotonic() + cooldown
        self._cached = None
        self._write_health({
            "source": "ALPHAFEED_REALTIME", "observed_at": self._now().isoformat(),
            "status": "UNAVAILABLE", "execution_ready_count": 0,
            "error_type": type(error).__name__, "http_status": code if isinstance(code, int) else None,
            "retry_after_seconds": cooldown,
        })

    def _fetch_batch(self, batch: list[str]):
        try:
            missing = [s for s in batch
                       if s not in self._metadata or self._monotonic() - self._metadata[s][0] > 300]
            if missing:
                for item in self._client.instruments.get(missing):
                    self._metadata[item["symbol"]] = (self._monotonic(), item)
            started = self._monotonic()
            raw = self._client.quotes.get(symbols=batch)
            quote_received = self._monotonic()
            quote_received_at = self._now().isoformat()
            depths = self._client.depth.batch(batch, max_workers=1)
            self._fetch_timings.append({"symbols": len(batch),
                "quotes_request_seconds": round(quote_received - started, 4),
                "depth_request_seconds": round(self._monotonic() - quote_received, 4),
                "quote_received_at": quote_received_at,
                "depth_received_at": self._now().isoformat()})
            return raw, depths
        except Exception as error:
            # Never retry a failed provider call inside the polling operation.
            # SDK APIError exposes status_code; no credential-bearing text parsing.
            self._provider_failure(error)
            raise

    def intraday_quotes(self, symbols: list[str]) -> pd.DataFrame:
        symbols = sorted(set(symbols))
        if not symbols:
            return pd.DataFrame()
        if len(symbols) > 1000:
            raise ValueError("AlphaFeed real-time request exceeds 1000 symbols")
        with self._lock:
            if self._monotonic() < self._retry_after:
                raise RuntimeError("AlphaFeed cooling down after provider failure; no request sent")
            cached = bool(
                self._cached
                and self._cached[0] == symbols
                and self._monotonic() - self._cached[1] < 2
            )
            if cached:
                raw, depths = self._cached[2:]
            else:
                raw, depths = [], {}
                self._fetch_timings = []
                for start in range(0, len(symbols), 100):
                    batch = symbols[start : start + 100]
                    batch_quotes, batch_depths = self._fetch_batch(batch)
                    raw.extend(batch_quotes)
                    depths.update(batch_depths)
                self._cached = (symbols, self._monotonic(), raw, depths)
            rows = []
            counts = Counter(item.get("symbol") for item in raw if isinstance(item, dict))
            for item in raw:
                if not isinstance(item, dict) or item.get("symbol") not in symbols or counts[item.get("symbol")] != 1:
                    continue
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
                "status": "AVAILABLE" if all(row["execution_ready"] for row in rows) and len(rows) == len(symbols) else "QUALITY_BLOCKED",
                "observed_at": self._now().isoformat(),
                "sample_count_this_process": self._sample_count,
                "fresh_fetch": not cached,
                "fetch_timings": self._fetch_timings,
                "requested_symbols": len(symbols),
                "returned_symbols": len(frame),
                "missing_symbols": sorted(set(symbols) - {row["symbol"] for row in rows}),
                "duplicate_symbols": sorted(s for s in symbols if counts[s] > 1),
                "execution_ready_count": sum(row["execution_ready"] for row in rows),
                "blocked_symbols": [{"symbol": row["symbol"], "reasons": row["quality_reasons"]}
                                    for row in rows if not row["execution_ready"]],
                "max_quote_depth_skew_seconds": max(
                    (abs(row["timestamp"] - row["depth_timestamp"]) / 1000
                     for row in rows if row["depth_timestamp"]), default=None),
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
            self._write_health(status)
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
            if self._monotonic() < self._retry_after:
                raise RuntimeError("AlphaFeed cooling down after provider failure; no request sent")
            for symbol in sorted(set(symbols)):
                try:
                    frame = self._client.klines.get(
                        symbol, period="1m", adjust="none", count=2000,
                        start_time=int(start_dt.timestamp() * 1000),
                        end_time=int(end_dt.timestamp() * 1000), to_dataframe=True)
                except Exception as error:
                    self._provider_failure(error)
                    raise
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
