import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "disclosure_index",
    Path(__file__).resolve().parents[1] / "scripts/collect_public_disclosure_index.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Session:
    def __init__(self, pages):
        self.headers = {}
        self.pages = iter(pages)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        data = next(self.pages)
        if isinstance(data, Exception):
            raise data

        class Response:
            content = b"{}"

            def raise_for_status(self):
                pass

            def json(self):
                return data

        return Response()


def entry(identifier="1", code="600036"):
    return {"announcementId": identifier, "secCode": code, "announcementTime": 1698422400000}


def run(tmp_path, session):
    return module.collect(
        tmp_path, "600036.SH", "20231027", "20231029", session=session, pause=lambda _: None
    )


def test_complete_index_and_cache_resume_without_network(tmp_path):
    session = Session(
        [
            [{"code": "600036", "orgId": "org"}],
            {"announcements": [entry()], "totalRecordNum": 1, "hasMore": False},
        ]
    )
    result = run(tmp_path, session)
    assert result["status"] == "INDEX_COMPLETE_FACTS_NOT_REVIEWED"
    assert result["financial_gate_passed"] is False
    assert result["requests"] == 2
    assert run(tmp_path, Session([]))["requests"] == 0


def test_budget_exhaustion_never_means_complete(tmp_path):
    pages = [
        {"announcements": [entry(str(i))], "totalRecordNum": 4, "hasMore": True} for i in range(3)
    ]
    result = run(tmp_path, Session([[{"code": "600036", "orgId": "org"}], *pages]))
    assert result["status"] == "PAGE_BUDGET_EXHAUSTED_INCOMPLETE"
    assert result["requests"] == 4


@pytest.mark.parametrize(
    "pages",
    [
        [RuntimeError("offline")],
        [
            [{"code": "600036", "orgId": "org"}],
            {"announcements": [entry(code="600000")], "totalRecordNum": 1},
        ],
        [
            [{"code": "600036", "orgId": "org"}],
            {"announcements": [entry(), entry()], "totalRecordNum": 2},
        ],
    ],
)
def test_failure_stops_without_retry(tmp_path, pages):
    session = Session(pages)
    result = run(tmp_path, session)
    assert result["status"] == "STOPPED_NO_RETRY"
    assert session.calls <= 2


def test_window_bound_is_enforced_before_network(tmp_path):
    with pytest.raises(ValueError, match="WINDOW"):
        module.collect(tmp_path, "600036.SH", "20230101", "20231231", session=Session([]))
