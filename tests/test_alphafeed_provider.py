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

    p = AlphaFeedProvider(tmp_path, client=SimpleNamespace(quotes=SimpleNamespace(get=denied),
                         instruments=SimpleNamespace(get=lambda symbols: [])))
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
    elapsed = [0.0]
    provider = AlphaFeedProvider(tmp_path, client=client, now=lambda: NOW, monotonic=lambda: elapsed[0])
    with pytest.raises(type(error)):
        provider.intraday_quotes([q["symbol"]])
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="cooling down"):
        provider.intraday_quotes([q["symbol"]])
    assert len(calls) == 1
    elapsed[0] += 6
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


@pytest.mark.parametrize("code,cooldown", [(429, 60), (401, 300), (403, 300)])
def test_provider_status_cooldown_prevents_repeated_requests(tmp_path, code, cooldown):
    import json

    class ProviderFailure(Exception):
        status_code = code

    q, d, i = sample()
    elapsed, calls = [0.0], []

    def quotes(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ProviderFailure("private diagnostic not for health output")
        return [q]

    provider = AlphaFeedProvider(tmp_path, monotonic=lambda: elapsed[0], now=lambda: NOW,
        client=SimpleNamespace(
            quotes=SimpleNamespace(get=quotes),
            depth=SimpleNamespace(batch=lambda *args, **kw: {q["symbol"]: d}),
            instruments=SimpleNamespace(get=lambda symbols: [i]),
        ))
    with pytest.raises(ProviderFailure):
        provider.intraday_quotes([q["symbol"]])
    health_path = tmp_path / "state/runtime/alphafeed-health.json"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert health["status"] == "UNAVAILABLE"
    assert health["retry_after_seconds"] == cooldown
    assert "private diagnostic" not in health_path.read_text(encoding="utf-8")
    elapsed[0] = cooldown - 0.01
    with pytest.raises(RuntimeError, match="cooling down"):
        provider.intraday_quotes([q["symbol"]])
    assert len(calls) == 1
    elapsed[0] = cooldown
    assert provider.intraday_quotes([q["symbol"]]).iloc[0].execution_ready
    assert json.loads(health_path.read_text(encoding="utf-8"))["status"] == "AVAILABLE"


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


def test_metadata_before_quotes_and_cached_reads_not_journaled_as_upstream(tmp_path):
    import json
    q, d, i = sample()
    calls = []
    elapsed = [0.0]
    client = SimpleNamespace(
        instruments=SimpleNamespace(get=lambda symbols: calls.append("metadata") or [i]),
        quotes=SimpleNamespace(get=lambda **kw: calls.append("quotes") or [q]),
        depth=SimpleNamespace(batch=lambda *args, **kw: calls.append("depth") or {q["symbol"]: d}))
    p = AlphaFeedProvider(tmp_path, client=client, now=lambda: NOW, monotonic=lambda: elapsed[0])
    p.intraday_quotes([q["symbol"]])
    p.intraday_quotes([q["symbol"]])
    assert calls == ["metadata", "quotes", "depth"]
    journal = tmp_path / "state/runtime/alphafeed-quality-20261021.jsonl"
    assert len(journal.read_text().splitlines()) == 1
    assert json.loads(journal.read_text())["fresh_fetch"]
    elapsed[0] = 3
    p.intraday_quotes([q["symbol"]])
    assert calls == ["metadata", "quotes", "depth", "quotes", "depth"]
    assert len(journal.read_text().splitlines()) == 2


def test_minute_429_freezes_quotes_and_minutes_until_cooldown(tmp_path):
    class Limited(Exception):
        status_code = 429

    calls, elapsed = [], [0.0]

    def minutes(*args, **kwargs):
        calls.append("minute")
        if len(calls) == 1:
            raise Limited("private text")
        return pd.DataFrame()

    p = AlphaFeedProvider(tmp_path, client=SimpleNamespace(klines=SimpleNamespace(get=minutes)),
                          now=lambda: NOW, monotonic=lambda: elapsed[0])
    args = (["600036.SH"], "2026-10-21 09:30", "2026-10-21 11:30")
    with pytest.raises(Limited):
        p.minute_history(*args)
    with pytest.raises(RuntimeError, match="cooling down"):
        p.minute_history(*args)
    with pytest.raises(RuntimeError, match="cooling down"):
        p.intraday_quotes(["600036.SH"])
    assert len(calls) == 1
    elapsed[0] = 60
    assert p.minute_history(*args).empty
    assert len(calls) == 2


def test_duplicate_or_unrequested_rows_never_enter_execution(tmp_path):
    q, d, i = sample()
    wrong = {**q, "symbol": "000001.SZ"}
    p = AlphaFeedProvider(tmp_path, now=lambda: NOW, client=SimpleNamespace(
        instruments=SimpleNamespace(get=lambda symbols: [i]),
        quotes=SimpleNamespace(get=lambda **kw: [q, q.copy(), wrong]),
        depth=SimpleNamespace(batch=lambda *args, **kw: {q["symbol"]: d})))
    assert p.intraday_quotes([q["symbol"]]).empty
    import json
    health = json.loads((tmp_path / "state/runtime/alphafeed-health.json").read_text())
    assert health["status"] == "QUALITY_BLOCKED"
    assert health["duplicate_symbols"] == [q["symbol"]]
    assert health["missing_symbols"] == [q["symbol"]]
