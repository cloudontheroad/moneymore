from __future__ import annotations

import json
import math
import sqlite3
import threading
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .tqqq_cycle import CycleParameters, cycle_decisions, cycle_target, latest_cycle_decision
from .tqqq_execution import fill_at_open, share_delta

ACCOUNT_ID = "tqqq_sqqq_v5a_paper"
STRATEGY_ID = "cycle_rates_fear_v2"
INITIAL_CASH = 100_000.0
TRADE_SYMBOLS = ("TQQQ", "SQQQ")
DATA_SYMBOLS = ("QQQ", "TQQQ", "SQQQ", "^VIX", "^VIX3M", "^TNX")
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Allocation:
    model_version: str
    signal_date: str
    regime: str
    tqqq: float
    sqqq: float
    cash: float
    score: float
    qqq_close: float
    ma180: float
    vix: float
    vix_sma20: float
    annualized_volatility: float
    macd_histogram: float
    ma_slope: float
    base_tqqq: float
    estimated_position_volatility: float
    risk_scalar: float
    no_trade_band: float
    risk_on_step: float
    fear_rank: float
    rate_rank: float
    term_structure: float
    rate_change_60d: float
    ma250: float
    distance_from_ma250: float
    crash_guard_active: bool
    momentum: float | None
    relative_volatility: float | None
    rsi: float | None
    dip_buy: bool


class YahooDailyClient:
    """Small, dependency-free Yahoo chart client used only by this US ETF sleeve."""

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def history(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        period2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
        encoded = urllib.parse.quote(symbol, safe="")
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded}"
            f"?period1={period1}&period2={period2}&interval=1d&events=div%2Csplits"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 MoneyMore/1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(result["timestamp"], unit="s", utc=True)
                .tz_convert("America/New_York")
                .strftime("%Y-%m-%d"),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote["volume"],
            }
        )
        return frame.dropna(subset=["open", "close"]).drop_duplicates("trade_date", keep="last")


class LeveragedEtfPaper:
    """Fully isolated TQQQ/cash paper account using a cycle-aware strategy."""

    def __init__(
        self,
        database: str | Path,
        cache_dir: str | Path,
        *,
        account_id: str = ACCOUNT_ID,
        strategy_id: str = STRATEGY_ID,
        initial_cash: float = INITIAL_CASH,
    ) -> None:
        self.database = Path(database)
        self.cache_dir = Path(cache_dir)
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.initial_cash = initial_cash
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._display_backtest_cache: tuple[tuple, list[dict[str, object]], list[dict[str, object]]] | None = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS account (
                    account_id TEXT PRIMARY KEY, initial_cash REAL NOT NULL,
                    cash REAL NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY, quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS signals (
                    signal_date TEXT PRIMARY KEY, regime TEXT NOT NULL,
                    tqqq REAL NOT NULL, sqqq REAL NOT NULL, cash REAL NOT NULL,
                    score REAL NOT NULL, indicators_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, signal_date TEXT NOT NULL,
                    execution_date TEXT, symbol TEXT NOT NULL, side TEXT NOT NULL,
                    quantity INTEGER NOT NULL, status TEXT NOT NULL, target_weight REAL NOT NULL,
                    reason TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(signal_date, symbol, side)
                );
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL UNIQUE,
                    signal_date TEXT NOT NULL, trade_date TEXT NOT NULL, symbol TEXT NOT NULL,
                    side TEXT NOT NULL, quantity INTEGER NOT NULL, price REAL NOT NULL,
                    fee REAL NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity_daily (
                    trade_date TEXT PRIMARY KEY, cash REAL NOT NULL, market_value REAL NOT NULL,
                    equity REAL NOT NULL, tqqq_quantity INTEGER NOT NULL,
                    sqqq_quantity INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                """
            )
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT OR IGNORE INTO account VALUES (?, ?, ?, ?, ?)",
                (self.account_id, self.initial_cash, self.initial_cash, now, now),
            )

    def refresh_data(self, client: YahooDailyClient | None = None) -> dict[str, pd.DataFrame]:
        client = client or YahooDailyClient()
        end = datetime.now(UTC).date().isoformat()
        start = (datetime.now(UTC).date() - timedelta(days=3000)).isoformat()
        result: dict[str, pd.DataFrame] = {}
        for symbol in DATA_SYMBOLS:
            cache = self.cache_dir / f"{symbol.replace('^', '')}.csv"
            try:
                fresh = client.history(symbol, start, end)
                if fresh.empty:
                    raise RuntimeError(f"empty Yahoo history: {symbol}")
                fresh.to_csv(cache, index=False)
                result[symbol] = fresh
            except Exception:
                if not cache.exists():
                    raise
                result[symbol] = pd.read_csv(cache)
        return self._align(result)

    @staticmethod
    def _align(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        common = set(data["QQQ"]["trade_date"].astype(str))
        for symbol in ("TQQQ", "SQQQ"):
            common &= set(data[symbol]["trade_date"].astype(str))
        dates = sorted(common)
        if len(dates) < 504:
            raise RuntimeError("TQQQ cycle strategy requires at least 504 aligned sessions")
        result = {}
        for symbol, frame in data.items():
            normalized = (
                frame.assign(trade_date=frame["trade_date"].astype(str))
                .drop_duplicates("trade_date", keep="last")
                .set_index("trade_date")
                .sort_index()
            )
            if symbol in ("QQQ", "TQQQ", "SQQQ"):
                aligned = normalized.reindex(dates)
            else:
                aligned = normalized.reindex(normalized.index.union(dates)).sort_index().ffill().reindex(dates)
            result[symbol] = aligned.rename_axis("trade_date").reset_index()
        return result

    @staticmethod
    def allocation(
        data: dict[str, pd.DataFrame], index: int = -1, previous_tqqq: float | None = None
    ) -> Allocation:
        position = index if index >= 0 else len(data["QQQ"]) + index
        sliced = {symbol: frame.iloc[: position + 1].copy() for symbol, frame in data.items()}
        dates = pd.to_datetime(sliced["QQQ"]["trade_date"])

        def series(symbol: str) -> pd.Series:
            return pd.Series(
                sliced[symbol]["close"].astype(float).to_numpy(), index=dates
            )

        close = series("QQQ")
        vix = series("^VIX")
        rate = series("^TNX")
        vix3m = series("^VIX3M")
        decision = latest_cycle_decision(
            close, vix, rate, vix3m, previous_target=previous_tqqq
        )
        returns = close.pct_change()
        vol = float(returns.rolling(20).std().iloc[-1] * math.sqrt(252))
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        histogram = macd - macd.ewm(span=9, adjust=False).mean()
        hist = float(histogram.iloc[-1])
        ma_slope = float(close.rolling(180).mean().pct_change(20).iloc[-1] * 100)
        vix_value = float(vix.iloc[-1])
        vix_average = float(vix.rolling(20).mean().iloc[-1])
        tqqq = decision.target_tqqq
        sqqq = 0.0
        cash = 1.0 - tqqq
        risk_scalar = (
            decision.raw_target / max(
                0.45 + (decision.combined_rank / 100.0 - 0.5), 1e-8
            )
        )
        return Allocation(
            model_version=STRATEGY_ID,
            signal_date=str(sliced["QQQ"].iloc[-1]["trade_date"]), regime=decision.regime,
            tqqq=round(tqqq, 6), sqqq=round(sqqq, 6), cash=round(cash, 6),
            score=round(decision.combined_rank, 4), qqq_close=float(close.iloc[-1]),
            ma180=decision.ma250, vix=vix_value,
            vix_sma20=vix_average, annualized_volatility=vol,
            macd_histogram=hist, ma_slope=ma_slope,
            base_tqqq=round(decision.raw_target, 6),
            estimated_position_volatility=round(3.0 * vol, 6),
            risk_scalar=round(risk_scalar, 6), no_trade_band=decision.rebalance_band,
            risk_on_step=0.0, fear_rank=round(decision.fear_rank, 4),
            rate_rank=round(decision.rate_rank, 4),
            term_structure=round(decision.term_structure, 6),
            rate_change_60d=round(decision.rate_change_60d, 6),
            ma250=round(decision.ma250, 6),
            distance_from_ma250=round(decision.distance_from_ma250, 6),
            crash_guard_active=decision.crash_guard_active,
            momentum=None, relative_volatility=None, rsi=None, dip_buy=False,
        )

    def run(self, client: YahooDailyClient | None = None) -> dict[str, object]:
        data = self.refresh_data(client)
        now = datetime.now(NEW_YORK)
        today = now.date().isoformat()
        latest_date = str(data["QQQ"].iloc[-1]["trade_date"])
        # Yahoo exposes today's still-forming daily candle during the session.
        # Startup recovery must never turn that partial candle into a close signal.
        if latest_date == today and (now.hour, now.minute) < (16, 15):
            data = {
                symbol: frame.loc[frame["trade_date"].astype(str) < today]
                .reset_index(drop=True)
                for symbol, frame in data.items()
            }
        return self._run_aligned(data)

    def _run_aligned(self, data: dict[str, pd.DataFrame]) -> dict[str, object]:
        with self._connect() as connection:
            recent = connection.execute(
                "SELECT signal_date, tqqq, indicators_json FROM signals ORDER BY signal_date DESC LIMIT 2"
            ).fetchall()
            last_signal_date = None if not recent else str(recent[0]["signal_date"])
            latest_indicators = {} if not recent else json.loads(str(recent[0]["indicators_json"]))
            previous_tqqq = (
                float(recent[0]["tqqq"])
                if recent and latest_indicators.get("model_version") == self.strategy_id
                else None
            )
        indices = [
            index
            for index, trade_date in enumerate(data["QQQ"]["trade_date"].astype(str))
            if last_signal_date is None or trade_date > str(last_signal_date)
        ]
        # A new account starts from the latest completed session; an existing
        # account replays every missing session so orders, holdings and NAV do
        # not jump across an outage.
        if last_signal_date is None and indices:
            indices = indices[-1:]
        for index in indices:
            trade_date = str(data["QQQ"].iloc[index]["trade_date"])
            opens = {symbol: float(data[symbol].iloc[index]["open"]) for symbol in TRADE_SYMBOLS}
            closes = {symbol: float(data[symbol].iloc[index]["close"]) for symbol in TRADE_SYMBOLS}
            self._execute_pending(trade_date, opens)
            self._record_equity(trade_date, closes)
            allocation = self.allocation(data, index, previous_tqqq=previous_tqqq)
            self._queue(allocation, closes)
            previous_tqqq = allocation.tqqq
        closes = {symbol: float(data[symbol].iloc[-1]["close"]) for symbol in TRADE_SYMBOLS}
        if not indices and last_signal_date == str(data["QQQ"].iloc[-1]["trade_date"]):
            prior_indicators = (
                json.loads(str(recent[1]["indicators_json"])) if len(recent) > 1 else {}
            )
            prior_tqqq = (
                float(recent[1]["tqqq"])
                if len(recent) > 1 and prior_indicators.get("model_version") == self.strategy_id
                else None
            )
            self._queue(self.allocation(data, -1, previous_tqqq=prior_tqqq), closes)
        return self.snapshot(closes)

    def run_open(self, client: YahooDailyClient | None = None) -> dict[str, object]:
        """Execute prior-close orders using today's official session open.

        This deliberately does not calculate a new signal from the incomplete
        intraday daily bar returned shortly after the US open.
        """
        data = self.refresh_data(client)
        trade_date = str(data["QQQ"].iloc[-1]["trade_date"])
        expected_date = datetime.now(NEW_YORK).date().isoformat()
        if trade_date != expected_date:
            raise RuntimeError(
                f"US session open bar is not available yet: expected {expected_date}, got {trade_date}"
            )
        opens = {symbol: float(data[symbol].iloc[-1]["open"]) for symbol in TRADE_SYMBOLS}
        closes = {symbol: float(data[symbol].iloc[-1]["close"]) for symbol in TRADE_SYMBOLS}
        self._execute_pending(trade_date, opens)
        self._record_equity(trade_date, closes)
        return self.snapshot(closes)

    def run_close(self, client: YahooDailyClient | None = None) -> dict[str, object]:
        """Mark the account and create the next-session target after US close."""
        payload = self.run(client)
        expected_date = datetime.now(NEW_YORK).date().isoformat()
        latest_signal = payload.get("latest_signal")
        actual_date = None if latest_signal is None else latest_signal.get("signal_date")
        if actual_date != expected_date:
            raise RuntimeError(
                f"US session close bar is not available yet: expected {expected_date}, got {actual_date}"
            )
        return payload

    def _execute_pending(self, trade_date: str, opens: dict[str, float]) -> None:
        with self._connect() as connection:
            pending = connection.execute(
                "SELECT * FROM orders WHERE status = 'PENDING' AND signal_date < ? ORDER BY CASE side WHEN 'SELL' THEN 0 ELSE 1 END, id",
                (trade_date,),
            ).fetchall()
            for order in pending:
                symbol, side, requested = str(order["symbol"]), str(order["side"]), int(order["quantity"])
                account = connection.execute(
                    "SELECT cash FROM account WHERE account_id = ?", (self.account_id,)
                ).fetchone()
                position = connection.execute("SELECT quantity, avg_cost FROM positions WHERE symbol = ?", (symbol,)).fetchone()
                held = int(position["quantity"]) if position else 0
                fill = fill_at_open(side, requested, opens[symbol], float(account["cash"]), held)
                quantity, price, fee = fill.quantity, fill.price, fill.fee
                if quantity <= 0:
                    connection.execute("UPDATE orders SET status='CANCELLED', execution_date=? WHERE id=?", (trade_date, order["id"]))
                    continue
                new_quantity = held + quantity if side == "BUY" else held - quantity
                old_cost = float(position["avg_cost"]) if position else 0.0
                avg_cost = ((held * old_cost + quantity * price) / new_quantity) if side == "BUY" and new_quantity else (old_cost if new_quantity else 0.0)
                now = datetime.now(UTC).isoformat()
                connection.execute(
                    "UPDATE account SET cash=cash+?, updated_at=? WHERE account_id=?",
                    (fill.cash_delta, now, self.account_id),
                )
                connection.execute(
                    "INSERT INTO positions VALUES (?, ?, ?, ?) ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity, avg_cost=excluded.avg_cost, updated_at=excluded.updated_at",
                    (symbol, new_quantity, avg_cost, now),
                )
                connection.execute("UPDATE orders SET status='FILLED', execution_date=? WHERE id=?", (trade_date, order["id"]))
                connection.execute(
                    "INSERT INTO fills(order_id, signal_date, trade_date, symbol, side, quantity, price, fee, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (order["id"], order["signal_date"], trade_date, symbol, side, quantity, price, fee, now),
                )

    def _queue(self, allocation: Allocation, closes: dict[str, float]) -> None:
        weights = {"TQQQ": allocation.tqqq, "SQQQ": allocation.sqqq}
        with self._connect() as connection:
            now = datetime.now(UTC).isoformat()
            indicators = asdict(allocation)
            signal_exists = connection.execute(
                "SELECT 1 FROM signals WHERE signal_date=?", (allocation.signal_date,)
            ).fetchone()
            if signal_exists:
                connection.execute(
                    """
                    UPDATE signals
                    SET regime=?, tqqq=?, sqqq=?, cash=?, score=?, indicators_json=?
                    WHERE signal_date=?
                    """,
                    (
                        allocation.regime, allocation.tqqq, allocation.sqqq,
                        allocation.cash, allocation.score, json.dumps(indicators),
                        allocation.signal_date,
                    ),
                )
                connection.execute(
                    "DELETE FROM orders WHERE signal_date=? AND status='PENDING'",
                    (allocation.signal_date,),
                )
            account = connection.execute(
                "SELECT cash FROM account WHERE account_id=?", (self.account_id,)
            ).fetchone()
            positions = {row["symbol"]: int(row["quantity"]) for row in connection.execute("SELECT * FROM positions")}
            equity = float(account["cash"]) + sum(positions.get(symbol, 0) * closes[symbol] for symbol in TRADE_SYMBOLS)
            if not signal_exists:
                connection.execute(
                    "INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (allocation.signal_date, allocation.regime, allocation.tqqq, allocation.sqqq,
                     allocation.cash, allocation.score, json.dumps(indicators), now),
                )
            for symbol in TRADE_SYMBOLS:
                current = positions.get(symbol, 0)
                difference = share_delta(equity, weights[symbol], closes[symbol], current)
                if difference == 0:
                    continue
                side = "BUY" if difference > 0 else "SELL"
                connection.execute(
                    "INSERT OR IGNORE INTO orders(signal_date, symbol, side, quantity, status, target_weight, reason, created_at) VALUES (?, ?, ?, ?, 'PENDING', ?, ?, ?)",
                    (allocation.signal_date, symbol, side, abs(difference), weights[symbol], allocation.regime, now),
                )

    def _record_equity(self, trade_date: str, closes: dict[str, float]) -> None:
        with self._connect() as connection:
            cash = float(
                connection.execute(
                    "SELECT cash FROM account WHERE account_id=?", (self.account_id,)
                ).fetchone()[0]
            )
            positions = {row["symbol"]: int(row["quantity"]) for row in connection.execute("SELECT * FROM positions")}
            market_value = sum(positions.get(symbol, 0) * closes[symbol] for symbol in TRADE_SYMBOLS)
            connection.execute(
                "INSERT OR REPLACE INTO equity_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
                (trade_date, cash, market_value, cash + market_value, positions.get("TQQQ", 0),
                 positions.get("SQQQ", 0), datetime.now(UTC).isoformat()),
            )

    def snapshot(self, mark_prices: dict[str, float] | None = None) -> dict[str, object]:
        mark_prices = mark_prices or self._cached_marks()
        with self._connect() as connection:
            account = dict(
                connection.execute(
                    "SELECT * FROM account WHERE account_id=?", (self.account_id,)
                ).fetchone()
            )
            positions = [dict(row) for row in connection.execute("SELECT * FROM positions WHERE quantity != 0 ORDER BY symbol")]
            signals = [dict(row) for row in connection.execute("SELECT * FROM signals ORDER BY signal_date DESC LIMIT 30")]
            orders = [dict(row) for row in connection.execute("SELECT * FROM orders ORDER BY id DESC LIMIT 100")]
            fills = [dict(row) for row in connection.execute("SELECT * FROM fills ORDER BY id DESC LIMIT 100")]
            history = [dict(row) for row in connection.execute("SELECT * FROM equity_daily ORDER BY trade_date")]
            versions = connection.execute(
                """SELECT signal_date, json_extract(indicators_json, '$.model_version')
                   FROM signals ORDER BY signal_date"""
            ).fetchall()
        for position in positions:
            price = float(mark_prices.get(str(position["symbol"]), position["avg_cost"]))
            position["mark_price"] = price
            position["market_value"] = int(position["quantity"]) * price
            position["unrealized_pnl"] = int(position["quantity"]) * (price - float(position["avg_cost"]))
        market_value = sum(float(row["market_value"]) for row in positions)
        equity = float(account["cash"]) + market_value
        latest_signal = signals[0] if signals else None
        if latest_signal:
            latest_signal["indicators"] = json.loads(str(latest_signal.pop("indicators_json")))
        peak = max([self.initial_cash, *(float(row["equity"]) for row in history)])
        benchmark_history = self._buy_hold_benchmark(history)
        backtest_history, comparison_benchmark = self._year_backtest_history()
        version_segments = []
        equity_by_date = {row["trade_date"]: row["equity"] for row in history}
        for day, version in versions:
            if not version_segments or version_segments[-1]["version"] != version:
                version_segments.append({"version": version, "start": day, "end": day})
            version_segments[-1]["end"] = day
        for segment in version_segments:
            start = equity_by_date.get(segment["start"])
            end = equity_by_date.get(segment["end"])
            segment["return"] = end / start - 1 if start and end is not None else None
        return {
            "account_id": self.account_id, "strategy_id": self.strategy_id,
            "initial_cash": self.initial_cash, "currency": "USD", "status": "PAPER_ONLY",
            "cash": float(account["cash"]), "market_value": market_value, "equity": equity,
            "total_return": equity / self.initial_cash - 1, "drawdown": equity / peak - 1,
            "positions": positions, "latest_signal": latest_signal,
            "orders": orders, "fills": fills, "history": history,
            "backtest_history": backtest_history,
            "version_segments": version_segments,
            "benchmark_history": benchmark_history,
            "comparison_benchmark": comparison_benchmark,
            "data_source": "Yahoo Finance daily chart; cached independently",
            "execution_policy": "close signal / next session open / 5 bps slippage / 1 bp fee",
        }

    def _buy_hold_benchmark(self, history: list[dict[str, object]]) -> list[dict[str, object]]:
        """TQQQ buy-and-hold from the first market session of the latest year."""
        path = self.cache_dir / "TQQQ.csv"
        if not path.exists():
            return []
        prices = pd.read_csv(path, usecols=["trade_date", "close"])
        prices["trade_date"] = prices["trade_date"].astype(str)
        prices = prices.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        if prices.empty:
            return []
        latest_date = str(history[-1]["trade_date"]) if history else str(prices.iloc[-1]["trade_date"])
        year_start = f"{latest_date[:4]}-01-01"
        aligned = [
            (str(row.trade_date), float(row.close))
            for row in prices.loc[
                prices["trade_date"].between(year_start, latest_date, inclusive="both")
            ].itertuples(index=False)
        ]
        if not aligned:
            return []
        start_price = aligned[0][1]
        result = [
            {
                "trade_date": trade_date,
                "close": close,
                "normalized_nav": close / start_price,
                "equity": self.initial_cash * close / start_price,
            }
            for trade_date, close in aligned
        ]
        if result[0]["trade_date"] != year_start:
            result.insert(0, {
                "trade_date": year_start,
                "close": start_price,
                "normalized_nav": 1.0,
                "equity": self.initial_cash,
                "is_baseline": True,
            })
        return result

    def _year_backtest_history(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Display-only replay from Jan 1; it never writes to the paper ledger."""
        paths = [self.cache_dir / f"{symbol.replace('^', '')}.csv" for symbol in DATA_SYMBOLS]
        if any(not path.exists() for path in paths):
            return [], []
        now = datetime.now(NEW_YORK)
        fingerprint = (now.date(), (now.hour, now.minute) >= (16, 15),
                       *(path.stat().st_mtime_ns for path in paths))
        if self._display_backtest_cache and self._display_backtest_cache[0] == fingerprint:
            return self._display_backtest_cache[1], self._display_backtest_cache[2]
        data = self._align({
            symbol: pd.read_csv(path)
            for symbol, path in zip(DATA_SYMBOLS, paths, strict=True)
        })
        last_day = now.date().isoformat()
        dates = data["QQQ"]["trade_date"].astype(str)
        completed = dates <= last_day if (now.hour, now.minute) >= (16, 15) else dates < last_day
        if not completed.any():
            return [], []
        latest_date = dates[completed].iloc[-1]
        year_start = f"{latest_date[:4]}-01-01"
        indices = [
            index for index, value in enumerate(data["QQQ"]["trade_date"].astype(str))
            if year_start < value <= latest_date
        ]
        targets = self._backtest_targets(data, start_index=indices[0] if indices else len(dates))
        rows = self.replay(data, targets, indices, year_start, initial_cash=self.initial_cash)
        benchmark = self.replay(data, targets, indices, year_start,
                                initial_cash=self.initial_cash, buy_hold=True)
        self._display_backtest_cache = (fingerprint, rows, benchmark)
        return rows, benchmark

    @staticmethod
    def replay(
        data: dict[str, pd.DataFrame], targets: pd.DataFrame, indices: list[int],
        baseline_date: str, *, initial_cash: float = INITIAL_CASH,
        slippage: float = 0.0005, cash_yield: float = 0.0, buy_hold: bool = False,
        drift_band: float = 0.0,
    ) -> list[dict[str, object]]:
        """Recreate the close-size / next-open execution policy in memory."""
        cash = initial_cash
        quantities = {symbol: 0 for symbol in TRADE_SYMBOLS}
        opening_prices = {symbol: data[symbol]["open"].to_numpy(dtype=float) for symbol in TRADE_SYMBOLS}
        closing_prices = {symbol: data[symbol]["close"].to_numpy(dtype=float) for symbol in TRADE_SYMBOLS}
        dates = data["QQQ"]["trade_date"].astype(str).tolist()
        target_weights = targets["tqqq"].to_numpy(dtype=float)
        target_sqqq_weights = targets.get("sqqq", pd.Series(0.0, index=targets.index)).to_numpy(dtype=float)
        regimes = targets["regime"].tolist()
        pending: list[tuple[str, str, int]] = []
        rows: list[dict[str, object]] = []
        rows.append({
            "trade_date": baseline_date,
            "equity": initial_cash,
            "normalized_nav": 1.0,
            "target_tqqq": 0.0,
            "regime": "BASELINE",
            "is_baseline": True,
        })
        previous_equity = initial_cash
        previous_targets: dict[str, float] = {}
        for offset, index in enumerate(indices):
            cash *= (1 + cash_yield) ** (1 / 252)
            if buy_hold and offset == 0:
                pending = [("TQQQ", "BUY", int(cash // opening_prices["TQQQ"][index]))]
            fills = 0
            notional = 0.0
            cost = 0.0
            opens = {symbol: prices[index] for symbol, prices in opening_prices.items()}
            opening_equity = cash + sum(quantities[symbol] * opens[symbol] for symbol in TRADE_SYMBOLS)
            for symbol, side, requested in sorted(pending, key=lambda item: item[1] != "SELL"):
                held = quantities[symbol]
                fill = fill_at_open(side, requested, opens[symbol], cash, held, slippage=slippage)
                if not fill.quantity:
                    continue
                cash += fill.cash_delta
                quantities[symbol] += fill.quantity if side == "BUY" else -fill.quantity
                fills += 1
                notional += fill.price * fill.quantity
                cost += fill.fee + abs(fill.price - opens[symbol]) * fill.quantity
            closes = {symbol: prices[index] for symbol, prices in closing_prices.items()}
            equity = cash + sum(quantities[symbol] * closes[symbol] for symbol in TRADE_SYMBOLS)
            target_tqqq = 1.0 if buy_hold else target_weights[index]
            target_sqqq = 0.0 if buy_hold else target_sqqq_weights[index]
            regime = "BUY_AND_HOLD" if buy_hold else regimes[index]
            rows.append({
                "trade_date": dates[index],
                "equity": equity,
                "normalized_nav": equity / initial_cash,
                "target_tqqq": target_tqqq if math.isfinite(target_tqqq) else 0.0,
                "regime": regime if math.isfinite(target_tqqq) else "WARMUP",
                "actual_tqqq": quantities["TQQQ"] * closes["TQQQ"] / equity,
                "fill_count": fills,
                "turnover": notional / opening_equity,
                "cost": cost,
                "overnight_return": opening_equity / previous_equity - 1,
            })
            previous_equity = equity
            pending = []
            if buy_hold or not math.isfinite(target_tqqq):
                continue
            for symbol, weight in (("TQQQ", target_tqqq), ("SQQQ", target_sqqq)):
                actual_weight = quantities[symbol] * closes[symbol] / equity
                target_changed = previous_targets.get(symbol) != weight
                previous_targets[symbol] = weight
                if not target_changed and abs(weight - actual_weight) < drift_band:
                    continue
                difference = share_delta(equity, weight, closes[symbol], quantities[symbol])
                if difference:
                    pending.append((symbol, "BUY" if difference > 0 else "SELL", abs(difference)))
        return rows

    @staticmethod
    def _backtest_targets(data: dict[str, pd.DataFrame], start_index: int = 0) -> pd.DataFrame:
        dates = pd.to_datetime(data["QQQ"]["trade_date"])

        def series(symbol: str) -> pd.Series:
            return pd.Series(data[symbol]["close"].astype(float).to_numpy(), index=dates)

        decisions = cycle_decisions(
            series("QQQ"), series("^VIX"), series("^TNX"), series("^VIX3M")
        )
        months = dates.dt.to_period("M")
        month_opens = months.ne(months.shift())
        band = CycleParameters().rebalance_band
        targets = [math.nan] * len(decisions)
        prior = None
        for index, row in enumerate(decisions.itertuples()):
            if index < start_index or not math.isfinite(row.raw_target):
                continue
            prior = round(cycle_target(
                row.raw_target, prior, row.crash_guard_active, bool(month_opens.iloc[index]), band
            ), 6)
            targets[index] = prior
        return pd.DataFrame({
            "tqqq": targets,
            "sqqq": 0.0,
            "regime": decisions["regime"].to_numpy(),
        })

    def _cached_marks(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for symbol in TRADE_SYMBOLS:
            path = self.cache_dir / f"{symbol}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                if not frame.empty:
                    result[symbol] = float(frame.iloc[-1]["close"])
        return result


class LeveragedEtfScheduler:
    """US-session scheduler, independent from MoneyMore's A-share timetable."""

    def __init__(self, paper: LeveragedEtfPaper, poll_seconds: int = 30) -> None:
        self.paper = paper
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_open_run: str | None = None
        self.last_close_run: str | None = None
        self.last_startup_run: str | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="moneymore-us-etf-paper", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def status(self) -> dict[str, object]:
        now = datetime.now(NEW_YORK)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "timezone": "America/New_York",
            "now": now.isoformat(),
            "open_schedule": "09:35",
            "close_schedule": "16:15",
            "last_open_run": self.last_open_run,
            "last_close_run": self.last_close_run,
            "last_startup_run": self.last_startup_run,
            "last_error": self.last_error,
        }

    def _loop(self) -> None:
        # Every service start immediately repairs missed completed sessions.
        # The normal open/close clocks below then take over for the live session.
        try:
            self.paper.run()
            self.last_startup_run = datetime.now(NEW_YORK).isoformat()
            self.last_error = None
        except Exception as error:  # noqa: BLE001 - retry on the timed loop
            self.last_error = f"{type(error).__name__}: {error}"
        while not self._stop.is_set():
            now = datetime.now(NEW_YORK)
            session = now.date().isoformat()
            weekday = now.weekday() < 5
            try:
                if weekday and (now.hour, now.minute) >= (9, 35) and self.last_open_run != session:
                    self.paper.run_open()
                    self.last_open_run = session
                    self.last_error = None
                if weekday and (now.hour, now.minute) >= (16, 15) and self.last_close_run != session:
                    self.paper.run_close()
                    self.last_close_run = session
                    self.last_error = None
            except Exception as error:  # noqa: BLE001 - retry transient quote/feed failures
                self.last_error = f"{type(error).__name__}: {error}"
            self._stop.wait(self.poll_seconds)
