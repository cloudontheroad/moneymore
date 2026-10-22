from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from moneymore.data.alphafeed_provider import SHANGHAI, AlphaFeedProvider, normalize_quote

NOW = datetime(2026, 10, 21, 10, 0, tzinfo=SHANGHAI)


def sample():
    stamp = int(NOW.timestamp() * 1000)
    return (
        {
            "symbol": "600036.SH",
            "timestamp": stamp,
            "last_price": 42.28,
            "prev_close": 42.52,
            "open": 42.6,
            "high": 42.6,
            "low": 42,
            "volume": 1000,
            "amount": 4220000,
        },
        {"symbol": "600036.SH", "timestamp": stamp, "bid_prices": [42.27], "ask_prices": [42.28]},
        {
            "symbol": "600036.SH",
            "name": "招商银行",
            "ext": {"limit_up": 46.77, "limit_down": 38.27, "tick_size": 0.01},
        },
    )


def test_units_and_ready():
    row = normalize_quote(*sample(), NOW)
    assert row["vwap"] == 42.2
    assert row["volume_shares"] == 100000
    assert row["volume"] == 1000
    assert row["execution_ready"] is True


@pytest.mark.parametrize(
    "change,reason",
    [
        ("missing", "INVALID_BOOK"),
        ("crossed", "INVALID_BOOK"),
        ("stale", "QUOTE_NOT_FRESH"),
        ("future", "DEPTH_NOT_FRESH"),
        ("skew", "QUOTE_DEPTH_SKEW"),
        ("zero_volume", "INVALID_VOLUME_AMOUNT"),
        ("units", "VWAP_OUTSIDE_RANGE"),
        ("limits", "MISSING_PRICE_LIMITS"),
    ],
)
def test_fail_closed(change, reason):
    q, d, i = sample()
    if change == "missing":
        d = {}
    elif change == "crossed":
        d["bid_prices"] = [43]
    elif change == "stale":
        q["timestamp"] -= 31000
    elif change == "future":
        d["timestamp"] += 1000
    elif change == "skew":
        d["timestamp"] -= 6000
    elif change == "zero_volume":
        q["volume"] = 0
    elif change == "units":
        q["amount"] *= 100
    elif change == "limits":
        i["ext"] = {}
    row = normalize_quote(q, d, i, NOW)
    assert not row["execution_ready"]
    assert reason in row["quality_reasons"]
    if change == "missing":
        assert row["bid1"] is None


def test_close_quotes_usable_for_display_not_execution():
    row = normalize_quote(*sample(), NOW.replace(hour=16))
    assert row["last"] == 42.28
    assert not row["execution_ready"]


def test_cache_rechecks_age_and_read_only(tmp_path):
    q, d, i = sample()
    count = []

    def quotes(**kwargs):
        count.append(kwargs)
        return [q]

    client = SimpleNamespace(
        quotes=SimpleNamespace(get=quotes),
        depth=SimpleNamespace(batch=lambda *args, **kw: {q["symbol"]: d}),
        instruments=SimpleNamespace(get=lambda symbols: [i]),
    )
    current = [NOW]
    p = AlphaFeedProvider(tmp_path, client=client, now=lambda: current[0])
    assert p.intraday_quotes([q["symbol"]]).iloc[0].execution_ready
    current[0] += timedelta(seconds=40)
    assert not p.intraday_quotes([q["symbol"]]).iloc[0].execution_ready
    assert len(count) == 1


def test_permission_errors_not_silently_fallback(tmp_path):
    def denied(**kwargs):
        raise PermissionError("No permission")

    p = AlphaFeedProvider(tmp_path, client=SimpleNamespace(quotes=SimpleNamespace(get=denied)))
    with pytest.raises(PermissionError):
        p.intraday_quotes(["600036.SH"])


@pytest.mark.parametrize("error", [TimeoutError("isolated timeout"), RuntimeError("isolated 429")])
def test_isolated_failure_releases_lock_without_retry_and_next_call_recovers(tmp_path, error):
    q, d, i = sample()
    calls = []

    def quotes(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise error
        return [q]

    client = SimpleNamespace(
        quotes=SimpleNamespace(get=quotes),
        depth=SimpleNamespace(batch=lambda *args, **kw: {q["symbol"]: d}),
        instruments=SimpleNamespace(get=lambda symbols: [i]),
    )
    provider = AlphaFeedProvider(tmp_path, client=client, now=lambda: NOW)
    with pytest.raises(type(error)):
        provider.intraday_quotes([q["symbol"]])
    assert len(calls) == 1
    assert provider.intraday_quotes([q["symbol"]]).iloc[0].execution_ready
    assert len(calls) == 2


def test_bounded_minute_history(tmp_path):
    q, _, _ = sample()

    def minute(*args, **kwargs):
        assert kwargs["adjust"] == "none"
        return pd.DataFrame([{"timestamp": q["timestamp"], "close": 42.28}])

    p = AlphaFeedProvider(tmp_path, client=SimpleNamespace(klines=SimpleNamespace(get=minute)))
    frame = p.minute_history([q["symbol"]], "2026-10-21 09:30", "2026-10-21 11:30")
    assert len(frame) == 1
    with pytest.raises(ValueError):
        p.minute_history([q["symbol"]], "2026-10-01", "2026-10-21")


def test_same_inputs_keep_same_strategy_decision():
    from moneymore.intraday_execution import decide_intraday_execution

    row = normalize_quote(*sample(), NOW)
    for side in ("BUY", "SELL"):
        migrated = decide_intraday_execution(
            symbol=row["symbol"],
            side=side,
            price=row["last"],
            open_price=row["open"],
            vwap=row["vwap"],
            bid1=row["bid1"],
            ask1=row["ask1"],
            now_time=NOW.time(),
        )
        original = decide_intraday_execution(
            symbol="600036.SH",
            side=side,
            price=42.28,
            open_price=42.6,
            vwap=42.2,
            bid1=42.27,
            ask1=42.28,
            now_time=NOW.time(),
        )
        assert migrated == original


def test_factory_selects_alpha_without_qmt(monkeypatch, tmp_path):
    from moneymore.data import alphafeed_provider
    from moneymore.data.realtime import _provider, create_realtime_provider

    sentinel = object()
    monkeypatch.setenv("MONEYMORE_REALTIME_DATA_SOURCE", "alphafeed")
    monkeypatch.setattr(alphafeed_provider, "AlphaFeedProvider", lambda **kw: sentinel)
    _provider.cache_clear()
    assert create_realtime_provider(tmp_path) is sentinel
    _provider.cache_clear()
