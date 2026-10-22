import io
import json
from urllib.error import HTTPError, URLError

import pytest

from moneymore.desktop_vm import HEALTH_URL, _NoRedirect, check_health


class FakeOpener:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return io.BytesIO(self.payload)


def test_health_is_not_trading_readiness():
    client = FakeOpener(json.dumps({
        "success": True, "data": {"status": "healthy", "automator": "connected"}
    }).encode())
    result = check_health(token="test-token", opener=client)
    assert result["status"] == "PROCESS_HEALTHY_ONLY"
    assert not result["live_submission_supported"]
    assert not result["account_verified"]
    assert not result["receipts_verified"]
    request, timeout = client.calls[0]
    assert request.full_url == HEALTH_URL
    assert request.get_method() == "GET"
    assert request.data is None
    assert timeout == 3


@pytest.mark.parametrize("token", ["", "bad\ntoken", "bad\rtoken"])
def test_missing_or_invalid_token_never_calls_network(token):
    client = FakeOpener(b"{}")
    assert check_health(token=token, opener=client)["status"] == "BLOCKED"
    assert not client.calls


@pytest.mark.parametrize("raw", [b"{}", b"[]", b"null", b"not json", b"x" * 65537,
    b'{"success":false,"data":{"status":"healthy","automator":"connected"}}',
    b'{"success":true,"data":{"status":"healthy","automator":"disconnected"}}'],
    ids=["empty", "array", "null", "invalid-json", "oversize", "failed", "disconnected"])
def test_bad_or_unhealthy_response_blocks(raw):
    assert check_health(token="test", opener=FakeOpener(raw))["status"] == "BLOCKED"


def test_network_failure_is_not_retried_or_leaked():
    class Broken:
        calls = 0

        def open(self, request, timeout):
            self.calls += 1
            raise URLError("sensitive detail")

    client = Broken()
    result = check_health(token="secret", opener=client)
    assert client.calls == 1
    assert "secret" not in str(result)
    assert "sensitive" not in str(result)


def test_redirect_is_not_followed():
    assert _NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com") is None


def test_http_failure_is_sanitized():
    class Unauthorized:
        def open(self, request, timeout):
            raise HTTPError(HEALTH_URL, 401, "secret", {}, None)

    assert check_health(token="test", opener=Unauthorized()) == {
        "status": "BLOCKED", "live_submission_supported": False,
        "reason": "VM_HTTP_ERROR", "http_status": 401,
    }
