import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("replay_candidate_styles", Path(__file__).resolve().parents[1] / "scripts/replay_candidate_styles.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_as_recorded_post_close_before_next_open_predictions():
    calendar = ["20261012", "20261013", "20261014"]
    assert module.prospective(dict(trade_date="20261012", created_at="2026-10-12T10:30:00Z"), calendar)
    assert not module.prospective(dict(trade_date="20261012", created_at="2026-10-12T06:59:00Z"), calendar)
    assert not module.prospective(dict(trade_date="20261012", created_at="2026-10-13T01:30:00Z"), calendar)
    assert not module.prospective(dict(trade_date="20261012", created_at=None), calendar)
