import json

import pandas as pd
import pytest

from moneymore.challenger_diagnostics import evaluate_forward_reports, prediction_reports


def fixture():
    dates = pd.bdate_range("2026-01-05", periods=8).strftime("%Y%m%d").tolist()
    report = {
        "trade_date": dates[0],
        "created_at": "2026-01-05T08:00:00+00:00",
        "selected": ["D"],
        "scores": [
            {"instrument": symbol, "datetime": dates[0], "sector": "x", "score": n}
            for n, symbol in enumerate("ABCD")
        ],
    }
    daily = pd.DataFrame(
        [
            {"ts_code": symbol, "trade_date": date, "close": 100 + n * j}
            for j, date in enumerate(dates)
            for n, symbol in enumerate("ABCD")
        ]
    )
    return dates, report, daily


def test_calendar_maturity_and_no_future_labels():
    dates, report, daily = fixture()
    early = evaluate_forward_reports([report], daily, dates, as_of=dates[1], horizon=2)
    assert early["pending_days"] == 1
    assert early["matured_days"] == 0
    result = evaluate_forward_reports([report], daily, dates, as_of=dates[2], horizon=2)
    assert result["matured_days"] == 1
    assert result["daily"][0]["maturity_date"] == dates[2]
    assert result["rank_ic"] == pytest.approx(1)
    assert result["daily"][0]["selected_excess_return"] == pytest.approx(0.03)


def test_missing_symbol_is_reported_not_silently_skip_entire_day():
    dates, report, daily = fixture()
    daily = daily.loc[~((daily.trade_date == dates[2]) & (daily.ts_code == "B"))]
    result = evaluate_forward_reports([report], daily, dates, as_of=dates[3], horizon=2)
    assert result["incomplete_days"] == 1
    row = result["all_observations"][0]
    assert row["missing_symbols"] == ["B"]
    assert row["sample_count"] == 3
    assert row["coverage"] == 0.75


def test_replayed_or_early_prediction_not_promotion_evidence():
    dates, report, daily = fixture()
    for capture in ["2026-01-12T08:00:00+00:00", "2026-01-05T01:00:00+00:00", None]:
        result = evaluate_forward_reports(
            [{**report, "created_at": capture}], daily, dates, as_of=dates[-1], horizon=2
        )
        assert result["diagnostic_matured_days"] == 1
        assert result["matured_days"] == 0


def test_degenerate_score_is_not_zero_ic():
    dates, report, daily = fixture()
    for row in report["scores"]:
        row["score"] = 1
    result = evaluate_forward_reports([report], daily, dates, as_of=dates[-1], horizon=2)
    assert result["rank_ic"] is None
    assert result["all_observations"][0]["status"] == "INCOMPLETE_LABELS"


def test_wrong_score_date_cannot_mature():
    dates, report, daily = fixture()
    report["scores"][0]["datetime"] = dates[1]
    result = evaluate_forward_reports([report], daily, dates, as_of=dates[-1], horizon=2)
    assert result["all_observations"][0]["status"] == "SCORE_DATE_MISMATCH"


def test_first_recorded_prediction_wins_over_later_repair(tmp_path):
    dates, report, _ = fixture()
    (tmp_path / "original.json").write_text(json.dumps(report))
    (tmp_path / "repair.json").write_text(
        json.dumps({**report, "created_at": "2026-01-12T12:00:00Z"})
    )
    result = prediction_reports(tmp_path, dates[-1])
    assert len(result) == 1
    assert result[0]["source_file"] == "original.json"
    assert len(result[0]["source_sha256"]) == 64


def test_peer_matching_and_linked_contributions_reconcile_exactly():
    from moneymore.challenger_attribution import linked_attribution, matched_day

    style = pd.DataFrame(
        {"size": [1, 1.1, 3], "momentum": [1, 1.1, 3], "volatility": [1, 1.1, 3]}, index=list("ABC")
    )
    result = matched_day(
        [{"symbol": "A", "quantity": 500}],
        100000,
        0.01,
        100,
        pd.Series({"A": 100}),
        pd.Series({"A": 0.04, "B": 0.02, "C": 0.01}),
        style,
        list("ABC"),
        neighbors=1,
    )
    assert result["status"] == "MATCHED"
    assert result["matches"][0]["peers"] == ["B"]
    assert result["selection"] == pytest.approx(0.01)
    assert result["costs"] == pytest.approx(-0.001)
    linked = linked_attribution(
        [
            result,
            {
                **result,
                "actual_return": 0.02,
                "trading_and_other": result["trading_and_other"] + 0.01,
            },
        ]
    )
    assert linked["total_return"] == pytest.approx(1.01 * 1.02 - 1)
    assert linked["reconciliation_error"] == pytest.approx(0, abs=1e-12)


def test_missing_peer_return_never_causes_future_based_rematch():
    from moneymore.challenger_attribution import matched_day

    style = pd.DataFrame(
        {"size": [1, 1.1, 3], "momentum": [1, 1.1, 3], "volatility": [1, 1.1, 3]}, index=list("ABC")
    )
    result = matched_day(
        [{"symbol": "A", "quantity": 500}],
        100000,
        0.01,
        0,
        pd.Series({"A": 100}),
        pd.Series({"A": 0.04, "C": 0.01}),
        style,
        list("ABC"),
        neighbors=1,
    )
    assert result["status"] == "MISSING_MATCHED_PEER_RETURN"


def test_board_matching_never_crosses_main_star_chinext():
    from moneymore.challenger_attribution import board, matched_day

    assert board("688001.SH") == "STAR"
    assert board("301001.SZ") == "CHINEXT"
    assert board("600036.SH") == "MAIN"
    symbols = ["688001.SH", "600036.SH"]
    result = matched_day(
        [{"symbol": symbols[0], "quantity": 1}],
        1000,
        0,
        0,
        pd.Series(dict.fromkeys(symbols, 100)),
        pd.Series(dict.fromkeys(symbols, 0.01)),
        pd.DataFrame({"size": [1, 1], "momentum": [1, 1], "volatility": [1, 1]}, index=symbols),
        symbols,
        neighbors=1,
    )
    assert result["status"] == "INSUFFICIENT_STYLE_PEERS"


def test_input_continuity_preserves_missing_sessions_instead_of_compressing():
    from moneymore.challenger_diagnostics import align_trading_observations

    result = align_trading_observations(
        pd.DataFrame({"date": ["20260105", "20260107"], "raw_close": [10, 20]}),
        ["20260105", "20260106", "20260107"],
    )
    assert len(result) == 3
    assert pd.isna(result.iloc[1].raw_close)
    assert pd.isna(result.raw_close.pct_change(fill_method=None).iloc[2])


def test_review_runs_forward_then_style_and_never_applies_recommendations(tmp_path, monkeypatch):
    from moneymore.challenger_diagnostics import run_challenger_review

    calls = []
    monkeypatch.setattr(
        "moneymore.challenger_diagnostics.run_forward_evaluations",
        lambda *_: (
            calls.append("forward")
            or {"accounts": [{"account_id": "test", "matured_days": 25, "rolling_rank_ic": -0.1}]}
        ),
    )
    style = {
        "account_id": "test",
        "status": "COMPLETE",
        "expected_return_days": 10,
        "contributions": {"selection": -0.03, "style": -0.04},
        "sensitivity": {"3": {"matched_days": 10, "contributions": {"selection": -0.02}}},
    }
    monkeypatch.setattr(
        "moneymore.challenger_attribution.run_style_attribution",
        lambda *_: calls.append("style") or {"accounts": [style]},
    )
    monkeypatch.setattr("moneymore.qlib_candidate_observer.candidate_catalog", lambda *_: [])
    result = run_challenger_review(tmp_path, object(), "20261026")
    assert calls == ["forward", "style"]
    assert result["recommendations"][0]["recommendation"] == "RESEARCH_REDESIGN"
    assert not result["recommendations"][0]["automatic_action"]
    assert not result["writes_orders"]
    assert not result["rewrites_account_history"]
    style["sensitivity"]["3"]["contributions"]["selection"] = 0.01
    result = run_challenger_review(tmp_path, object(), "20261026")
    assert result["recommendations"][0]["recommendation"] != "RESEARCH_REDESIGN"


def test_old_recovery_does_not_overwrite_latest_evidence(tmp_path):
    from moneymore.challenger_diagnostics import write_diagnostic

    target = tmp_path / "forward-evaluation.json"
    write_diagnostic(target, {"as_of_date": "20261026", "rank_ic": -0.1})
    write_diagnostic(target, {"as_of_date": "20261012", "rank_ic": 0.1})
    assert json.loads(target.read_text())["as_of_date"] == "20261026"
    assert (
        json.loads((tmp_path / "forward-evaluation-history/20261012.json").read_text())["rank_ic"]
        == 0.1
    )
