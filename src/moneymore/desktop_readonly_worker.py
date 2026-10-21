"""MoneyMore-owned, on-demand UIA inventory. No Codex dependencies or input APIs.

Only inspect an explicitly configured executable. No coordinate/OCR fallback.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def probe(root: Path):
    config = root / "state/private/desktop-client.json"
    if not config.exists():
        return {"status": "BLOCKED", "reason": "尚未配置普通交易客户端；请先确定客户端安装路径"}
    profile = json.loads(config.read_text(encoding="utf-8"))
    executable = Path(profile.get("executable", ""))
    if (
        not executable.is_absolute()
        or executable.suffix.lower() != ".exe"
        or not executable.is_file()
    ):
        return {"status": "BLOCKED", "reason": "客户端路径无效；不会启动任何程序"}
    if sys.platform != "win32":
        return {"status": "BLOCKED", "reason": "桌面代理仅支持Windows交互会话"}
    try:
        import psutil
        from pywinauto import Desktop
    except ImportError:
        return {"status": "BLOCKED", "reason": "未安装desktop可选依赖"}
    pids = set()
    for process in psutil.process_iter(["pid", "exe"]):
        try:
            if process.info["exe"] and Path(process.info["exe"]).resolve() == executable.resolve():
                pids.add(process.info["pid"])
        except (OSError, psutil.Error):
            continue
    if not pids:
        return {"status": "BLOCKED", "reason": "目标客户端未运行或进程不可读；请自行启动和登录"}
    windows = []
    for window in Desktop(backend="uia").windows(visible_only=False):
        if window.process_id() not in pids:
            continue
        title = window.window_text()
        if any(word in title.lower() for word in ("登录", "密码", "验证码", "login", "password")):
            windows.append({"title": "认证窗口（跳过读取）", "status": "USER_LOGIN_REQUIRED"})
            continue
        controls = []
        for item in window.descendants()[:300]:
            if item.element_info.control_type in {"Edit", "Document", "Password"}:
                continue
            # Inventory is not a verified broker-order or trade parser.
            controls.append(
                {
                    "type": item.element_info.control_type,
                    "name": re.sub(r"\d{5,}", "***", item.window_text())[:100],
                }
            )
        windows.append(
            {
                "title": re.sub(r"\d{5,}", "***", title),
                "controls": controls,
                "status": "READ_ONLY_INVENTORY",
            }
        )
    return {
        "status": "INSPECTED" if windows else "BLOCKED",
        "windows": windows,
        "foreground_activated": False,
        "mouse_input_sent": False,
        "background_read_acceptance": "UNVERIFIED",
        "submission_supported": False,
        "receipt_parser_supported": False,
    }


def main():
    try:
        result = probe(Path(__file__).resolve().parents[2])
    except Exception:  # noqa: BLE001 - do not leak desktop contents or credentials
        result = {"status": "BLOCKED", "reason": "后台控件读取失败；未执行前台回退"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
