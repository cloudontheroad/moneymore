import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from moneymore.research.baseline_v2_union_verification import verify_union_batch


def fixture_source(tmp_path):
    directory = tmp_path / "release"
    directory.mkdir()
    native = pd.DataFrame(
        {"ts_code": ["A", "A"], "trade_date": ["20181228", "20190102"], "adj_factor": [1.0, 1.01]}
    )
    native.to_parquet(directory / "full-union-native-prices.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "A",
                "trade_date": "20190102",
                "adj_factor": 1.01,
                "pre_close": 10.0,
                "previous_close": 10.0,
                "action_alignment": "NO_IMPLEMENTED_EVENT_IN_CACHE_UNRESOLVED",
            }
        ]
    ).to_parquet(directory / "full-union-action-alignment.parquet", index=False)
    (directory / "frozen-protocol.json").write_text(
        json.dumps({"scoring": {"coverage_audit": {"identity_policy": []}}})
    )
    official = pd.DataFrame(
        {
            "ts_code": ["A", "A"],
            "trade_date": ["20181228", "20190102"],
            "pre_close": [10, 10],
            "open": [10, 11],
            "close": [10, 11],
            "high": [10, 11],
            "low": [10, 11],
        }
    )
    return {
        "directory": str(directory),
        "replay_directory": str(directory),
        "input_hashes": {},
    }, official


def test_bounded_proof_does_not_normalize_and_is_cached(tmp_path, monkeypatch):
    manifest, official = fixture_source(tmp_path)
    monkeypatch.setattr(
        "moneymore.research.baseline_v2_union_verification.time.sleep", lambda _: None
    )
    calls = []

    def get(symbol, **kwargs):
        calls.append(kwargs)
        return official.rename(columns={"ts_code": "symbol"}).assign(
            trade_date=pd.to_datetime(official.trade_date)
        )

    store = SimpleNamespace(read=lambda *_args, **_kwargs: official.copy())
    client = SimpleNamespace(klines=SimpleNamespace(get=get))
    result = verify_union_batch(tmp_path, store, client, manifest, max_pairs=8)
    assert len(calls) == 2
    assert result["verified_non_economic_pairs"] == 1
    assert result["source_normalized"] is False
    again = verify_union_batch(tmp_path, store, client, manifest)
    assert again["requests"] == 0
    assert len(calls) == 2


def test_provider_failure_stops_and_cools_down_without_retry(tmp_path, monkeypatch):
    manifest, official = fixture_source(tmp_path)
    monkeypatch.setattr(
        "moneymore.research.baseline_v2_union_verification.time.sleep", lambda _: None
    )

    def get(*_args, **_kwargs):
        raise RuntimeError("must never persist provider credentials or messages")

    client = SimpleNamespace(klines=SimpleNamespace(get=get))
    store = SimpleNamespace(read=lambda *_args, **_kwargs: official.copy())
    result = verify_union_batch(tmp_path, store, client, manifest)
    assert result["status"] == "STOPPED_PROVIDER_ERROR"
    assert result["requests"] == 1
    assert "must never" not in json.dumps(result)
    assert verify_union_batch(tmp_path, store, client, manifest)["status"] == "PROVIDER_COOLDOWN"


def test_changed_input_fails_before_network(tmp_path):
    manifest, _ = fixture_source(tmp_path)
    manifest["input_hashes"] = {str(tmp_path / "missing"): "wrong"}
    with pytest.raises(FileNotFoundError):
        verify_union_batch(tmp_path, None, None, manifest)


def test_same_year_events_share_query_but_receive_separate_proofs(tmp_path, monkeypatch):
    manifest, official = fixture_source(tmp_path)
    directory = Path(manifest["directory"])
    native = pd.read_parquet(directory / "full-union-native-prices.parquet")
    pd.concat(
        [native, pd.DataFrame([{"ts_code": "A", "trade_date": "20190103", "adj_factor": 1.02}])]
    ).to_parquet(directory / "full-union-native-prices.parquet", index=False)
    issues = pd.read_parquet(directory / "full-union-action-alignment.parquet")
    pd.concat(
        [
            issues,
            pd.DataFrame(
                [
                    {
                        "ts_code": "A",
                        "trade_date": "20190103",
                        "adj_factor": 1.02,
                        "pre_close": 11.0,
                        "previous_close": 11.0,
                        "action_alignment": "NO_IMPLEMENTED_EVENT_IN_CACHE_UNRESOLVED",
                    }
                ]
            ),
        ]
    ).to_parquet(directory / "full-union-action-alignment.parquet", index=False)
    official = pd.concat(
        [
            official,
            pd.DataFrame(
                [
                    {
                        "ts_code": "A",
                        "trade_date": "20190103",
                        "pre_close": 11,
                        "open": 12,
                        "high": 12,
                        "low": 12,
                        "close": 12,
                    }
                ]
            ),
        ]
    )
    monkeypatch.setattr(
        "moneymore.research.baseline_v2_union_verification.time.sleep", lambda _: None
    )
    calls = []

    def get(symbol, **kwargs):
        calls.append(kwargs)
        return official.rename(columns={"ts_code": "symbol"}).assign(
            trade_date=pd.to_datetime(official.trade_date)
        )

    def read(*_args, **kwargs):
        return official.loc[official.trade_date.isin(kwargs["filters"][-1][2])].copy()

    result = verify_union_batch(
        tmp_path,
        SimpleNamespace(read=read),
        SimpleNamespace(klines=SimpleNamespace(get=get)),
        manifest,
        max_pairs=64,
    )
    assert len(calls) == 2
    assert result["reviewed_pairs"] == result["verified_non_economic_pairs"] == 2
    assert result["unresolved_pairs"] == 0
    assert result["max_http_requests_per_batch"] == 16


def test_http_limit_remains_sixteen_when_pair_budget_is_larger(tmp_path, monkeypatch):
    manifest, official = fixture_source(tmp_path)
    directory = Path(manifest["directory"])
    for name in ("full-union-native-prices", "full-union-action-alignment"):
        frame = pd.read_parquet(directory / (name + ".parquet"))
        pd.concat([frame.assign(ts_code=f"A{i}") for i in range(9)]).to_parquet(
            directory / (name + ".parquet"), index=False
        )
    monkeypatch.setattr(
        "moneymore.research.baseline_v2_union_verification.time.sleep", lambda _: None
    )
    calls = []

    def get(symbol, **kwargs):
        calls.append(symbol)
        return official.rename(columns={"ts_code": "symbol"}).assign(
            symbol=symbol, trade_date=pd.to_datetime(official.trade_date)
        )

    def read(*_args, **kwargs):
        return official.assign(ts_code=kwargs["filters"][0][2])

    result = verify_union_batch(
        tmp_path,
        SimpleNamespace(read=read),
        SimpleNamespace(klines=SimpleNamespace(get=get)),
        manifest,
        max_pairs=64,
    )
    assert len(calls) == 16
    assert result["status"] == "BOUNDED_INDEPENDENT_BATCH_COMPLETE"
    assert result["new_pairs"] == 8
    assert result["remaining_pairs"] == 1
