import json
from moneymore.execution.paper import PaperBroker


def test_retired_primary_gru_blocks_only_that_account(tmp_path):
    state = tmp_path / "state"
    broker = PaperBroker(state / "paper_orders.sqlite3")
    deployment = state / "qlib-governance/deployment.json"
    deployment.parent.mkdir()
    deployment.write_text(json.dumps({"lifecycle": "RETIRED"}))
    assert broker._candidate_frozen("qlib_gru_shadow")
    assert not broker._candidate_frozen("multi_sector_shadow")
    assert not broker._candidate_frozen("multi_sector_intraday_shadow")
    deployment.write_text("broken")
    assert broker._candidate_frozen("qlib_gru_shadow")
