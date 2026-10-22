"""Read-only EasyTHS health check over a separately established loopback tunnel.

No GUI dependency, POST, order operation, reconnect or automatic retry lives here.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

HEALTH_URL = "http://127.0.0.1:17648/api/v1/system/health"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def check_health(*, token: str = "", opener=None) -> dict:
    """Report process health only: never imply correct account or trade readiness."""
    blocked = {"status": "BLOCKED", "live_submission_supported": False}
    if not token or "\n" in token or "\r" in token:
        return {**blocked, "reason": "VM_API_TOKEN_REQUIRED"}
    request = Request(
        HEALTH_URL,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    # No environment proxy or redirects: credentials stay on the fixed loopback endpoint.
    client = opener or build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with client.open(request, timeout=3) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            return {**blocked, "reason": "INVALID_HEALTH_RESPONSE"}
        payload = json.loads(raw)
        data = payload.get("data") if isinstance(payload, dict) else None
        if (
            not isinstance(data, dict)
            or payload.get("success") is not True
            or data.get("status") != "healthy"
            or data.get("automator") != "connected"
        ):
            return {**blocked, "reason": "VM_CLIENT_NOT_HEALTHY"}
        return {
            "status": "PROCESS_HEALTHY_ONLY",
            "account_verified": False,
            "receipts_verified": False,
            "live_submission_supported": False,
        }
    except HTTPError as error:
        return {**blocked, "reason": "VM_HTTP_ERROR", "http_status": error.code}
    except (OSError, URLError, ValueError, TypeError):
        # Do not echo responses, exceptions, credentials or account information.
        return {**blocked, "reason": "VM_HEALTH_UNAVAILABLE"}


if __name__ == "__main__":
    print(json.dumps(check_health(token=os.environ.get("MONEYMORE_VM_API_TOKEN", ""))))
