import json

import pandas as pd
import pytest
import yaml

from moneymore import qlib_governance as governance


def test_readonly_drift_ignores_future_labels_and_never_changes_orders(tmp_path, monkeypatch):
    config = {
        "drift_monitoring": {
            "reference_window": 20,
            "recent_window": 20,
            "minimum_observations": 20,
            "minimum_rolling_rank_ic": -0.02,
            "maximum_score_psi": 0.25,
            "maximum_feature_psi": 0.25,
            "maximum_turnover": 0.5,
            "maximum_cost_ratio": 0.005,
        },
        "training_policy": {"frequency": "weekly"},
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/qlib_challenger.yaml").write_text(yaml.safe_dump(config))
    path = tmp_path / "state/qlib-challenger/forward-evaluation-history/20261026.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "daily": [{"rank_ic": -0.1, "maturity_date": "20261026"}] * 20
                + [{"rank_ic": 99, "maturity_date": "20261027"}]
            }
        )
    )
    deployment = {"execution_mode": "PAPER_TRADING", "release_id": "test"}
    monkeypatch.setattr(governance, "bootstrap_qlib_release", lambda *_: deployment)
    monkeypatch.setattr(
        governance,
        "_latest_reports",
        lambda *_: [{"trade_date": "20261026", "scores": [], "selected": []}] * 40,
    )
    monkeypatch.setattr(governance, "_psi", lambda *_: 0)
    monkeypatch.setattr(governance, "_feature_drift", lambda *_: 0)
    monkeypatch.setattr(
        governance,
        "transition_release",
        lambda *_args, **_kwargs: pytest.fail("diagnostic changed deployment"),
    )

    class Broker:
        def fills(self, *_):
            return []

        def cancel_pending(self, *_args, **_kwargs):
            pytest.fail("diagnostic cancelled orders")

    class Store:
        def read(self, *_):
            return pd.DataFrame({"trade_date": ["20261026"], "equity": [1000000]})

    result = governance.evaluate_qlib_drift(
        tmp_path, Store(), Broker(), "20261026", apply_protection=False
    )
    assert result["matured_forward_days"] == 20
    assert result["rolling_rank_ic"] == pytest.approx(-0.1)
    assert result["status"] == "BREACH"
    assert result["protection_mode"] == "DIAGNOSTIC_ONLY"
