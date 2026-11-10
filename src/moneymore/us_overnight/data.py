from __future__ import annotations

import socket
from collections.abc import Mapping, Sequence
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Protocol

import pandas as pd

NEW_YORK = "America/New_York"


def normalize_futu_kline(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize OpenD K-lines into the timestamp/price/volume research contract."""
    if frame.empty:
        return frame.copy()
    required = {"time_key", "open", "close", "high", "low", "volume", "turnover"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Futu K-line is missing fields: {sorted(missing)}")
    result = frame.copy()
    local = pd.to_datetime(result["time_key"])
    result["timestamp"] = local.dt.tz_localize(
        NEW_YORK, ambiguous="infer", nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    volume = pd.to_numeric(result["volume"], errors="coerce")
    turnover = pd.to_numeric(result["turnover"], errors="coerce")
    reported = turnover.div(volume.where(volume > 0))
    ohlc4 = result[["open", "high", "low", "close"]].mean(axis=1)
    valid_reported = reported.between(result["low"], result["high"], inclusive="both")
    result["price"] = reported.where(valid_reported, ohlc4)
    result["price_source"] = valid_reported.map(
        {True: "TURNOVER_DIV_VOLUME", False: "OHLC4_FALLBACK"}
    )
    result["volume"] = volume
    return result


class ReadOnlyMarketData(Protocol):
    """Quote-only boundary. It intentionally has no order or account methods."""

    def stock_basicinfo(self) -> pd.DataFrame: ...
    def history_kline(
        self, symbol: str, start: str, end: str, ktype: str, autype: str = "None"
    ) -> pd.DataFrame: ...
    def trading_days(self, start: str, end: str) -> Sequence[str]: ...
    def probe(self, timeout: float = 1.0) -> dict[str, object]: ...


class FixtureMarketData:
    def __init__(self, tables: Mapping[str, pd.DataFrame], days: Sequence[str] = ()) -> None:
        self.tables = {key: value.copy() for key, value in tables.items()}
        self.days = tuple(days)

    @classmethod
    def from_directory(cls, root: str | Path) -> FixtureMarketData:
        return cls({path.stem: pd.read_parquet(path) for path in Path(root).glob("*.parquet")})

    def stock_basicinfo(self) -> pd.DataFrame:
        return self.tables.get("stock_basicinfo", pd.DataFrame()).copy()

    def history_kline(
        self, symbol: str, start: str, end: str, ktype: str, autype: str = "None"
    ) -> pd.DataFrame:
        frame = self.tables.get(f"{symbol}_{ktype}", pd.DataFrame()).copy()
        if frame.empty or "timestamp" not in frame:
            return frame
        stamp = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.loc[(stamp >= pd.Timestamp(start)) & (stamp <= pd.Timestamp(end))].copy()

    def trading_days(self, start: str, end: str) -> Sequence[str]:
        return tuple(day for day in self.days if start <= day <= end)


class FutuOpenDReadOnly:
    """Lazy Futu quote adapter; importing this module never requires futu-api."""

    def __init__(self, host: str = "127.0.0.1", port: int = 11111) -> None:
        self.host, self.port = host, port

    def _context(self):
        available, error = self._available(0.5)
        if not available:
            raise RuntimeError(f"Futu OpenD is not reachable at {self.host}:{self.port}: {error}")
        try:
            from futu import OpenQuoteContext, SysConfig
        except ImportError as exc:
            raise RuntimeError("futu-api is not installed; use FixtureMarketData offline") from exc
        SysConfig.set_all_thread_daemon(True)
        return OpenQuoteContext(host=self.host, port=self.port)

    def _available(self, timeout: float) -> tuple[bool, str | None]:
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout):
                return True, None
        except OSError as exc:
            return False, str(exc)

    @staticmethod
    def _unwrap(result) -> pd.DataFrame:
        code, data = result
        if code != 0:
            raise RuntimeError(f"Futu OpenD quote request failed: {data}")
        return data.copy()

    def stock_basicinfo(self) -> pd.DataFrame:
        try:
            from futu import Market, SecurityType
        except ImportError as exc:
            raise RuntimeError("futu-api is not installed; use FixtureMarketData offline") from exc
        quote = self._context()
        try:
            return self._unwrap(quote.get_stock_basicinfo(Market.US, SecurityType.STOCK))
        finally:
            quote.close()

    def history_kline(
        self, symbol: str, start: str, end: str, ktype: str, autype: str = "None"
    ) -> pd.DataFrame:
        from futu import AuType

        adjustment = {
            "none": AuType.NONE,
            "qfq": AuType.QFQ,
            "hfq": AuType.HFQ,
        }.get(str(autype).lower(), autype)
        quote = self._context()
        frames: list[pd.DataFrame] = []
        page_key = None
        try:
            while True:
                code, data, page_key = quote.request_history_kline(
                    symbol,
                    start=start,
                    end=end,
                    ktype=ktype,
                    autype=adjustment,
                    max_count=1000,
                    page_req_key=page_key,
                )
                if code != 0:
                    raise RuntimeError(f"Futu OpenD quote request failed: {data}")
                if data is not None and len(data):
                    frames.append(data.copy())
                if page_key is None:
                    break
        finally:
            quote.close()
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return normalize_futu_kline(raw)

    def trading_days(self, start: str, end: str) -> Sequence[str]:
        from futu import Market

        quote = self._context()
        try:
            frame = self._unwrap(quote.request_trading_days(Market.US, start=start, end=end))
        finally:
            quote.close()
        if isinstance(frame, pd.DataFrame):
            return tuple(frame["time"].astype(str))
        return tuple(str(row["time"]) for row in frame)

    def market_snapshots(self, symbols: Sequence[str], batch_size: int = 400) -> pd.DataFrame:
        """Fetch quote snapshots in OpenD's bounded batches using one quote-only context."""
        if not 1 <= batch_size <= 400:
            raise ValueError("batch_size must be between 1 and 400")
        quote = self._context()
        frames: list[pd.DataFrame] = []
        try:
            for offset in range(0, len(symbols), batch_size):
                batch = list(symbols[offset : offset + batch_size])
                frame = self._unwrap(quote.get_market_snapshot(batch))
                if not frame.empty:
                    frames.append(frame)
        finally:
            quote.close()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def probe(self, timeout: float = 1.0) -> dict[str, object]:
        available, error = self._available(timeout)
        result: dict[str, object] = {
            "host": self.host,
            "port": self.port,
            "reachable": available,
            "read_only": True,
            "status": "OFFLINE" if not available else "CONNECTING",
            "error": error,
        }
        if not available:
            return result
        responses: Queue[dict[str, object]] = Queue(maxsize=1)

        def connect() -> None:
            quote = None
            try:
                quote = self._context()
                global_state = self._unwrap(quote.get_global_state())
                if isinstance(global_state, pd.DataFrame):
                    global_state = global_state.to_dict("records")
                code, quota = quote.get_history_kl_quota(get_detail=True)
                if code != 0:
                    raise RuntimeError(str(quota))
                used, remaining, details = quota
                responses.put(
                    {
                        "status": "ONLINE",
                        "error": None,
                        "global_state": global_state,
                        "history_quota": {
                            "used": int(used),
                            "remaining": int(remaining),
                            "codes": [str(item.get("code")) for item in details or []],
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001 - sanitized diagnostics only
                responses.put({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                if quote is not None:
                    quote.close()

        worker = Thread(target=connect, name="us-opend-readonly-probe", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            result.update(status="STARTING", error="OpenD port is open but login is not ready")
            return result
        result.update(responses.get_nowait())
        return result
