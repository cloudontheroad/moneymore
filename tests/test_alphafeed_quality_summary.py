import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("quality_summary", Path(__file__).parents[1] / "scripts/alphafeed_quality_summary.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empty_and_cached_evidence_cannot_claim_full_day():
    assert not module.summarize([], "20261026")["full_session_observed"]
    records = [{"observed_at": "2026-10-26T10:00:00+08:00", "status": "AVAILABLE", "fresh_fetch": False}]
    assert module.summarize(records, "20261026")["observed_trading_minutes"] == 0


def test_real_status_evidence_separates_recovery_from_approval():
    records = [
        {"observed_at": "2026-10-26T10:00:00+08:00", "status": "UNAVAILABLE", "http_status": 429},
        {"observed_at": "2026-10-26T10:01:00+08:00", "status": "QUALITY_BLOCKED", "fresh_fetch": True,
         "quality_reasons": ["QUOTE_DEPTH_SKEW"]},
        {"observed_at": "2026-10-26T17:00:00+08:00", "status": "AVAILABLE", "fresh_fetch": True}, {}]
    result = module.summarize(records, "20261026")
    assert result["real_429_observed"] and result["observed_recovery_transitions"] == 1
    assert result["active_upstream_samples"] == 1
    assert result["malformed_records"] == 1 and not result["full_session_observed"]
