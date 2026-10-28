import json

import pandas as pd
import pytest

from moneymore.research.baseline_v2_execution_data import (
    cache_entry,
    execution_evidence,
    sync_execution_data,
    validate_response,
)


def scope(root):
    f = root / "state/baseline-v2-validation"
    f.mkdir(parents=True)
    p = {
        "source_weight_run": "test",
        "source_panel_sha256": "p",
        "start_date": "20240101",
        "end_date": "20251231",
        "required_data_symbols": ["A"],
    }
    (f / "execution-preflight.json").write_text(json.dumps(p), encoding="utf-8")
    return p


class Provider:
    calls = 0

    def stock_limits(self, *args):
        self.calls += 1
        return pd.DataFrame(
            {"ts_code": ["A"], "trade_date": ["20240102"], "up_limit": [11.0], "down_limit": [9.0]}
        )

    def dividends(self, *args):
        self.calls += 1
        from moneymore.research.baseline_v2_execution_data import FIELDS

        return pd.DataFrame(columns=FIELDS["dividends"])

    def suspensions(self, *args):
        self.calls += 1
        from moneymore.research.baseline_v2_execution_data import FIELDS

        return pd.DataFrame(columns=FIELDS["suspensions"])

    def historical_adjustments(self, *args):
        self.calls += 1
        return pd.DataFrame({"ts_code": ["A"], "trade_date": ["20240102"], "adj_factor": [1.0]})


def test_budget_resume_and_empty_responses_are_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("moneymore.research.baseline_v2_execution_data.time.sleep", lambda _: None)
    p = scope(tmp_path)
    provider = Provider()
    first = sync_execution_data(tmp_path, provider, max_requests=1)
    assert first["requests"] == 1 and first["complete"] == 1
    result = sync_execution_data(tmp_path, provider, max_requests=3)
    assert result["status"] == "DOWNLOAD_COMPLETE_PENDING_AUDIT"
    assert provider.calls == 4
    third = sync_execution_data(tmp_path, provider, max_requests=10)
    assert third["requests"] == 0 and third["cached"] == 4
    frames, missing = execution_evidence(tmp_path, p)
    assert not missing and frames["dividends"].empty
    assert not (tmp_path / "data").exists()


def test_provider_failure_stops_and_cools_down_without_secret(tmp_path):
    scope(tmp_path)

    class Bad:
        def stock_limits(self, *args):
            raise RuntimeError("SENSITIVE_TOKEN_DO_NOT_PERSIST")

    result = sync_execution_data(tmp_path, Bad(), max_requests=5)
    assert result["requests"] == 1 and result["status"] == "STOPPED_ERROR"
    assert "SENSITIVE" not in json.dumps(result)
    assert sync_execution_data(tmp_path, Bad())["status"] == "PROVIDER_COOLDOWN"


def test_invalid_data_and_truncation_never_mark_complete():
    f = Provider().stock_limits()
    with pytest.raises(ValueError, match="SYMBOL"):
        validate_response(f, "limits", "B", "20240101", "20251231")
    with pytest.raises(ValueError, match="TRUNCATION"):
        validate_response(pd.concat([f] * 5800), "limits", "A", "20240101", "20251231")
    with pytest.raises(ValueError, match="INVALID_PRICE"):
        validate_response(f.assign(up_limit=8), "limits", "A", "20240101", "20251231")


def test_cache_tampering_blocks_instead_of_refetching(tmp_path):
    p = scope(tmp_path)
    provider = Provider()
    sync_execution_data(tmp_path, provider, max_requests=1)
    request = {"kind": "limits", "symbol": "A", "start": p["start_date"], "end": p["end_date"]}
    folder = tmp_path / "state/baseline-v2-validation/execution-data/test"
    cache_entry(folder, request).write_bytes(b"corrupted")
    result = sync_execution_data(tmp_path, provider, max_requests=5)
    assert result["status"] == "STOPPED_ERROR" and result["requests"] == 0
    assert provider.calls == 1


def test_adjustment_recheck_is_cached_and_isolated(tmp_path):
    from moneymore.research.baseline_v2_execution_data import sync_unexplained_adjustments

    p = scope(tmp_path)
    folder = tmp_path / "state/baseline-v2-validation"
    evidence = folder / "preflight"
    evidence.mkdir()
    p["output_directory"] = str(evidence)
    (folder / "execution-preflight.json").write_text(json.dumps(p), encoding="utf-8")
    pd.DataFrame(
        {"trade_date": ["20240625"], "official_dividend_ex_date_found": [False]}
    ).to_parquet(evidence / "adjustment-jump-audit.parquet")
    (folder / "execution-data/test").mkdir(parents=True)

    class Adj:
        calls = 0

        def adjustment_factors(self, date):
            self.calls += 1
            return pd.DataFrame({"ts_code": ["A"], "trade_date": [date], "adj_factor": [1.01]})

    provider = Adj()
    assert sync_unexplained_adjustments(tmp_path, provider)["requests"] == 1
    assert sync_unexplained_adjustments(tmp_path, provider)["requests"] == 0
    assert provider.calls == 1
    assert not (tmp_path / "data").exists()
