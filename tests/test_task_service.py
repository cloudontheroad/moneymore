from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from moneymore.data.store import ParquetStore
from moneymore.web import TaskService, _next_open_trade_date


def test_task_configuration_is_persistent(tmp_path: Path):
    database = tmp_path / "service.sqlite3"
    service = TaskService(database)

    initial = service.config()
    assert initial["enabled"] is True
    assert initial["schedule"] == "每日 18:30"
    assert [row["symbol"] for row in service.candidates()] == [
        "600036.SH",
        "600900.SH",
    ]

    updated = service.update_config(False, 19, 5)
    reloaded = TaskService(database).config()

    assert updated["schedule"] == "每日 19:05"
    assert reloaded["enabled"] is False
    assert reloaded["hour"] == 19
    assert reloaded["minute"] == 5


def test_task_step_retries_and_resumes_completed_work(tmp_path: Path):
    service = TaskService(tmp_path / "service.sqlite3")
    calls = 0

    def flaky_operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return "ready"

    status, result = service._run_step(1, "20260729", "research", flaky_operation)

    assert status == "COMPLETED"
    assert result == "ready"
    assert [row["status"] for row in reversed(service.steps(run_id=1))] == [
        "FAILED",
        "COMPLETED",
    ]

    status, result = service._run_step(
        2,
        "20260729",
        "research",
        lambda: (_ for _ in ()).throw(AssertionError("must not run again")),
    )
    assert status == "SKIPPED_COMPLETED"
    assert result is None
    assert service.steps(run_id=2)[0]["attempt"] == 0


def test_notifications_are_deduplicated_and_acknowledged(tmp_path: Path):
    service = TaskService(tmp_path / "service.sqlite3")
    for _ in range(2):
        service._notify(
            "ERROR",
            "PIPELINE_FAILED",
            "failed",
            "failure detail",
            trade_date="20260729",
            run_id=7,
        )

    notifications = service.notifications()
    assert len(notifications) == 1
    assert notifications[0]["acknowledged"] is False

    acknowledged = service.acknowledge_notification(notifications[0]["id"])
    assert acknowledged["acknowledged"] is True
    assert acknowledged["acknowledged_at"]


def test_daily_run_preview_is_read_only_and_marks_resume_steps(tmp_path: Path):
    service = TaskService(tmp_path / "service.sqlite3")
    service._run_step(11, "20260729", "bank_pipeline", lambda: "done")

    preview = service.preview("20260729")

    assert preview["mutates_state"] is False
    assert preview["prior_run_count"] == 0
    assert preview["steps"] == [
        {"step_name": "bank_pipeline", "action": "SKIP_COMPLETED"},
        {"step_name": "strategy_universe_refresh", "action": "RUN"},
        {"step_name": "composite_daily_basic", "action": "RUN"},
        {"step_name": "sector_research", "action": "RUN"},
        {"step_name": "selected_fundamental_backfill", "action": "RUN"},
        {"step_name": "multi_sector_execution", "action": "RUN"},
        {"step_name": "qlib_challenger_execution", "action": "RUN"},
        {"step_name": "qlib_candidate_observation", "action": "RUN"},
        {"step_name": "research_financial_pool_refresh", "action": "RUN"},
        {"step_name": "qlib_forward_and_style_review", "action": "RUN"},
    ]


def test_recovery_dates_fill_every_missing_open_session(tmp_path: Path):
    service = TaskService(tmp_path / "service.sqlite3")
    store = ParquetStore(tmp_path / "data")
    store.merge_curated(
        "daily",
        [
            pd.DataFrame(
                {"ts_code": ["A", "A"], "trade_date": ["20260727", "20260730"]}
            )
        ],
        ["ts_code", "trade_date"],
    )
    store.merge_curated(
        "trade_calendar",
        [
            pd.DataFrame(
                {
                    "cal_date": ["20260727", "20260728", "20260729", "20260730"],
                    "is_open": [1, 0, 1, 1],
                }
            )
        ],
        ["cal_date"],
    )
    completed = service._create_run("daily_pipeline", "20260727", "TEST")
    service._finish(completed, "COMPLETED")

    assert service._recovery_dates(store, "20260730") == ["20260729", "20260730"]


def test_recovery_dates_include_completed_day_with_missing_intraday_audit(tmp_path: Path):
    service = TaskService(tmp_path / "service.sqlite3")
    store = ParquetStore(tmp_path / "data")
    store.merge_curated(
        "daily",
        [pd.DataFrame({"ts_code": ["A"], "trade_date": ["20260918"]})],
        ["ts_code", "trade_date"],
    )
    store.merge_curated(
        "trade_calendar",
        [pd.DataFrame({"cal_date": ["20260917", "20260918"], "is_open": [1, 1]})],
        ["cal_date"],
    )
    for day in ("20260917", "20260918"):
        completed = service._create_run("daily_pipeline", day, "RECOVERY")
        service._finish(completed, "COMPLETED")
    for directory in (
        "intraday-execution",
        "baseline-pysystemtrade-intraday-execution",
    ):
        target = tmp_path / directory
        target.mkdir()
        (target / "20260917.json").write_text("{}", encoding="utf-8")

    assert service._recovery_dates(store, "20260918") == ["20260918"]

    for directory in (
        "intraday-execution",
        "baseline-pysystemtrade-intraday-execution",
    ):
        (tmp_path / directory / "20260918.json").write_text("{}", encoding="utf-8")
    assert service._recovery_dates(store, "20260918") == []


def test_controlled_intraday_audit_is_not_required_before_activation(tmp_path: Path):
    service = TaskService(tmp_path / "service.sqlite3")
    store = ParquetStore(tmp_path / "data")
    store.merge_curated(
        "daily",
        [pd.DataFrame({"ts_code": ["A"], "trade_date": ["20260908"]})],
        ["ts_code", "trade_date"],
    )
    store.merge_curated(
        "trade_calendar",
        [pd.DataFrame({"cal_date": ["20260908"], "is_open": [1]})],
        ["cal_date"],
    )
    completed = service._create_run("daily_pipeline", "20260908", "RECOVERY")
    service._finish(completed, "COMPLETED")
    audit = tmp_path / "intraday-execution"
    audit.mkdir()
    (audit / "20260908.json").write_text("{}", encoding="utf-8")

    assert service._recovery_dates(store, "20260908") == []


def test_scheduler_accepts_completed_recovery_and_throttles_waiting(tmp_path: Path):
    service = TaskService(tmp_path / "service.sqlite3")
    timezone = ZoneInfo("Asia/Shanghai")
    now = datetime.now(timezone)

    waiting = service._create_run("recovery_pipeline", "20260810", "SCHEDULED")
    service._finish(waiting, "WAITING_MARKET_DATA")
    assert not service._scheduler_should_trigger("20260810", now)
    assert service._scheduler_should_trigger(
        "20260810", now + timedelta(minutes=11)
    )

    completed = service._create_run("daily_pipeline", "20260810", "RECOVERY")
    service._finish(completed, "COMPLETED")
    assert not service._scheduler_should_trigger(
        "20260810", now + timedelta(hours=1)
    )


def test_next_open_trade_date_uses_official_calendar(tmp_path: Path):
    store = ParquetStore(tmp_path / "data")
    store.merge_curated(
        "trade_calendar",
        [
            pd.DataFrame(
                {
                    "cal_date": ["20260810", "20260811", "20260812"],
                    "is_open": [1, 1, 0],
                }
            )
        ],
        ["cal_date"],
    )
    assert _next_open_trade_date(store, "20260810") == "20260811"
