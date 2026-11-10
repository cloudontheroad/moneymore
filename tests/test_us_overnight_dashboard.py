from moneymore.us_overnight.dashboard import dashboard_snapshot


def test_dashboard_is_read_only_and_reports_opend_state(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "us_overnight_paper_only.yaml").write_text(
        "mode: US_PAPER_ONLY\noutput_root: data/us_overnight_paper_only\n",
        encoding="utf-8",
    )
    result = dashboard_snapshot(tmp_path)
    assert result["status"] == "WAITING_FOR_FIRST_RUN"
    assert result["policy"]["real_orders_enabled"] is False
    assert result["opend"]["read_only"] is True
    assert result["opend"]["status"] in {"OFFLINE", "STARTING", "ERROR", "ONLINE"}
    assert not (tmp_path / "data").exists()
