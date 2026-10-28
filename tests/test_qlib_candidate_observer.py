import json
from pathlib import Path

from moneymore.qlib_candidate_observer import (
    candidate_account_id,
    candidate_catalog,
    latest_candidate,
    run_candidate_queue_daily,
)


def _write_config(root: Path) -> None:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "qlib_challenger.yaml").write_text(
        """
model_id: qlib_gru_alpha360_v1
research_gate:
  minimum_samples: 10000
  minimum_rank_ic: 0.01
  minimum_rank_ic_ir: 0.08
  minimum_cost_adjusted_excess_return: 0.0
  minimum_seed_count: 5
  minimum_positive_seed_ratio: 0.8
""".strip(),
        encoding="utf-8",
    )


def _write_candidate(root: Path, tag: str, rank_ic: float) -> None:
    target = root / "state" / "qlib-challenger" / "candidates" / tag
    target.mkdir(parents=True)
    payload = {
        "candidate_tag": tag,
        "created_at": f"{tag[:4]}-{tag[4:6]}-{tag[6:]}T09:00:00+08:00",
        "data_cutoff": tag,
        "effective_segments": {"test": ["2026-01-01", "2026-07-01"]},
        "metrics": [
            {
                "model_id": "qlib_gru_alpha360_v1",
                "samples": 12000,
                "rank_ic": rank_ic,
                "rank_ic_ir": 0.12,
                "cost_adjusted_top_k_excess_return": 0.01,
                "average_turnover": 0.1,
            }
        ],
        "stability": {"seed_count": 5, "positive_seed_ratio": 0.8},
    }
    (target / "research.json").write_text(json.dumps(payload), encoding="utf-8")
    (target / "_TRAINING_COMPLETE").write_text("test=true\n", encoding="utf-8")


def test_candidate_catalog_ranks_versions_and_evaluates_gate(tmp_path: Path) -> None:
    _write_config(tmp_path)
    _write_candidate(tmp_path, "20260805", -0.02)
    _write_candidate(tmp_path, "20260812", 0.03)

    rows = candidate_catalog(tmp_path)

    assert [row["candidate_tag"] for row in rows] == ["20260812", "20260805"]
    assert rows[0]["research_gate_passed"] is True
    assert rows[1]["research_gate_passed"] is False
    assert latest_candidate(tmp_path) == rows[0]
    assert rows[0]["account_id"] == candidate_account_id("20260812")


def test_factor_catalog_does_not_require_gru_training_or_fabricate_metrics(tmp_path):
    _write_config(tmp_path)
    folder = tmp_path / "state/factor-baseline-candidates"
    folder.mkdir(parents=True)
    payload = {"candidate_tag": "20261028", "account_id": "qlib_candidate_20261028",
               "portfolio_variant": "factor_baseline_v2", "strategy_family": "factor_baseline",
               "display_name": "因子V2", "model_retrained": False, "promotion_eligible": False}
    (folder / "20261028.json").write_text(json.dumps(payload), encoding="utf-8")
    rows = candidate_catalog(tmp_path)
    assert len(rows) == 1
    assert rows[0]["research_path"] == str(folder / "20261028.json")
    assert rows[0]["rank_ic"] is None
    assert not rows[0]["promotion_eligible"]
    assert "model_dir" not in rows[0]


def test_factor_bootstrap_report_does_not_skip_actual_close_plan(tmp_path, monkeypatch):
    _write_config(tmp_path)
    folder = tmp_path / "state/factor-baseline-candidates"
    folder.mkdir(parents=True)
    (folder / "20261028.json").write_text(json.dumps({"candidate_tag": "20261028",
        "account_id": "qlib_candidate_20261028", "portfolio_variant": "factor_baseline_v2"}))
    reports = tmp_path / "state/qlib-candidate-shadow/20261028"
    reports.mkdir(parents=True)
    (reports / "20261028.json").write_text(json.dumps({"trade_date": "20261028", "signal_date": "20261027"}))
    calls = []
    monkeypatch.setattr("moneymore.qlib_candidate_observer.run_candidate_daily",
                        lambda **kw: calls.append(kw["trade_date"]))
    run_candidate_queue_daily(root=tmp_path, store=object(), broker=object(), config=object(), trade_date="20261028")
    assert calls == ["20261028"]


def test_portfolio_candidate_reuses_model_but_not_history_or_training_marker(tmp_path):
    _write_config(tmp_path)
    _write_candidate(tmp_path, "20261010", .03)
    folder = tmp_path / "state/qlib-portfolio-candidates"
    folder.mkdir()
    (folder / "20261027.json").write_text(json.dumps({"candidate_tag": "20261027",
        "source_candidate_tag": "20261010", "portfolio_variant": "fixed_budget_v1",
        "display_name": "固定预算实验", "model_retrained": False}), encoding="utf-8")
    rows = candidate_catalog(tmp_path)
    assert rows[0]["candidate_tag"] == "20261027"
    assert rows[0]["account_id"] != rows[1]["account_id"]
    assert rows[0]["model_dir"] == rows[1]["model_dir"]
    assert rows[0]["observation_days"] == 0
    assert not rows[0]["model_retrained"]
    assert not (tmp_path / "state/qlib-challenger/candidates/20261027/_TRAINING_COMPLETE").exists()


def test_candidate_queue_keeps_older_versions_running(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path)
    with (tmp_path / "configs" / "qlib_challenger.yaml").open("a", encoding="utf-8") as handle:
        handle.write("\ncandidate_observation_policy:\n  formal_review_days: 60\n")
    _write_candidate(tmp_path, "20260805", 0.02)
    _write_candidate(tmp_path, "20260812", 0.03)
    calls = []
    monkeypatch.setattr(
        "moneymore.qlib_candidate_observer.run_candidate_daily",
        lambda **kwargs: calls.append(kwargs["candidate"]["candidate_tag"]),
    )

    run_candidate_queue_daily(
        root=tmp_path,
        store=object(),
        broker=object(),
        config=object(),
        trade_date="20260812",
    )

    assert calls == ["20260805", "20260812"]


def test_archived_candidate_remains_in_catalog_but_queue_cannot_restart_it(monkeypatch, tmp_path):
    import sqlite3

    _write_config(tmp_path)
    _write_candidate(tmp_path, "20260812", 0.03)
    state = tmp_path / "state"
    with sqlite3.connect(state / "candidate-lifecycle.sqlite3") as connection:
        connection.execute("CREATE TABLE candidate_state(candidate_tag TEXT PRIMARY KEY, payload TEXT)")
        connection.execute("INSERT INTO candidate_state VALUES (?, ?)",
                           ("20260812", json.dumps({"status": "GATE_REJECTED"})))
    assert candidate_catalog(tmp_path)[0]["archived"]
    assert latest_candidate(tmp_path) is None

    calls = []
    class Broker:
        def cancel_pending(self, *args):
            calls.append(args)
    def forbidden(**kwargs):
        raise AssertionError("retired candidate was run")
    monkeypatch.setattr("moneymore.qlib_candidate_observer.run_candidate_daily", forbidden)
    assert run_candidate_queue_daily(root=tmp_path, store=object(), broker=Broker(),
                                     config=object(), trade_date="20261026") == []
    assert calls
