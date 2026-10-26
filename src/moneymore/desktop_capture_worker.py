"""Explicitly granted, one-shot Huatai read-only foreground export worker.

No login, credentials, buy/sell/cancel menus, clipboard or coordinate fallback.
Only called by the local capture service with a fixed job ID, never a command
or executable supplied through request arguments. Runtime acceptance required.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from .desktop_tables import parse_huatai_export

PAGES = (("positions", "资金股票"), ("orders", "当日委托"), ("trades", "当日成交"))


class CaptureBlocked(RuntimeError):
    pass


def inert_broker_helper(window):
    """Ignore only the observed zero-area IE helper, never a usable dialog."""
    if window.class_name() != "Internet Explorer_Hidden" or window.window_text():
        return False
    rect = window.rectangle()
    return rect.width() == 0 or rect.height() == 0


def capture(root: Path, job_id: str, adapter, *, clock=time.monotonic, sleep=time.sleep,
            lease_check=lambda: None):
    """Deterministic, dependency-injected workflow; failures import nothing."""
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise CaptureBlocked("INVALID_JOB")
    folder = root / "state/private/huatai-acceptance"
    folder.mkdir(parents=True, exist_ok=True)
    stop = root / f"state/private/desktop-capture/stop-{job_id}"
    started = clock()
    trade_date = datetime.now(UTC).astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d")

    def check():
        lease_check()
        if stop.exists():
            raise CaptureBlocked("STOP_REQUESTED")
        if clock() - started > 65:
            raise CaptureBlocked("CAPTURE_DEADLINE")
        if datetime.now(UTC).astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d") != trade_date:
            raise CaptureBlocked("DATE_ROLLOVER")

    check()
    adapter.attach()
    results = []
    for kind, page in PAGES:
        check()
        staging = folder / f"capture-{job_id}-{kind}.xls"
        final = folder / f"{kind}-{trade_date}-{job_id}.xls"
        if staging.exists() or final.exists():
            raise CaptureBlocked("CAPTURE_FILE_EXISTS")
        adapter.export(page, staging, check)
        deadline = clock() + 5
        previous = None
        while True:
            check()
            if clock() > deadline:
                raise CaptureBlocked("FILE_NOT_STABLE")
            if staging.is_file():
                stat = staging.stat()
                if 0 < stat.st_size <= 5_000_000:
                    state = (stat.st_size, stat.st_mtime_ns)
                    if state == previous:
                        break
                    previous = state
                elif stat.st_size > 5_000_000:
                    raise CaptureBlocked("EXPORT_TOO_LARGE")
            sleep(0.25)
        with staging.open("rb") as source:
            data = source.read(5_000_001)
        parsed = parse_huatai_export(data, kind=kind, account_ref="huatai_local", trade_date=trade_date)
        # File names are unique per job. Do not overwrite or recycle old exports.
        staging.rename(final)
        results.append({"kind": kind, "filename": final.name,
                        "sha256": parsed["source_sha256"], "rows": len(parsed["rows"])})
    adapter.finish(check)
    return {"status": "CAPTURED_UNVERIFIED", "job_id": job_id, "trade_date": trade_date,
            "exports": results, "foreground_used": True, "account_verified": False,
            "submitted": False, "broker_receipts_verified": False}


class HuataiForegroundAdapter:
    """Win32 navigation plus UIA common Save As; no blind Enter key presses."""

    def __init__(self, executable: Path):
        self.executable = executable
        self.phase = "INIT"
        self.diagnostics = {}

    def _unlocked(self):
        # Do not create, switch, unlock or change the input desktop.
        self.user32.OpenInputDesktop.restype = ctypes.c_void_p
        handle = self.user32.OpenInputDesktop(0, False, 0x0100)
        if not handle:
            raise CaptureBlocked("DESKTOP_LOCKED_OR_UNAVAILABLE")
        self.user32.CloseDesktop.argtypes = [ctypes.c_void_p]
        self.user32.GetUserObjectInformationW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                        ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
        name = ctypes.create_unicode_buffer(256)
        needed = ctypes.c_uint()
        try:
            if not self.user32.GetUserObjectInformationW(handle, 2, name, ctypes.sizeof(name), ctypes.byref(needed)) or name.value != "Default":
                raise CaptureBlocked("DESKTOP_LOCKED_OR_UNAVAILABLE")
        finally:
            self.user32.CloseDesktop(handle)

    def _guard(self, expected_handle=None):
        self._unlocked()
        expected = self.main.handle if expected_handle is None else expected_handle
        foreground = self.user32.GetForegroundWindow()
        if foreground != expected:
            self.diagnostics["foreground_matches_main"] = foreground == self.main.handle
            self.diagnostics["expected_foreground_role"] = "MAIN" if expected_handle is None else "SAVE_DIALOG"
            raise CaptureBlocked("FOREGROUND_CHANGED")
        if any(self.user32.GetAsyncKeyState(key) & 0x8000 for key in (1, 2, 16, 17, 18, 91, 92)):
            raise CaptureBlocked("USER_INPUT_ACTIVE")
        # Only our main window or the explicitly checked common Save As dialog.
        windows = self.Desktop(backend="win32").windows(process=self.pid, visible_only=True)
        unexpected = [w for w in windows if w.handle not in {self.main.handle, expected}
                      and w.class_name() not in {"tooltips_class32", "IME", "MSCTFIME UI"}
                      and not inert_broker_helper(w)]
        if unexpected:
            self.diagnostics["unexpected_window_classes"] = sorted({
                w.class_name() for w in unexpected
                if re.fullmatch(r"[A-Za-z0-9_#:. -]{1,100}", w.class_name())})
            self.diagnostics["zero_area_helpers"] = [
                {"empty_title": not w.window_text(), "width": w.rectangle().width(),
                 "height": w.rectangle().height()}
                for w in unexpected if w.class_name() == "Internet Explorer_Hidden"]
            raise CaptureBlocked("UNEXPECTED_DIALOG")
        if expected_handle is None and self._session_hash() != self.session_hash:
            raise CaptureBlocked("SELECTED_ACCOUNT_CHANGED")
        if expected_handle is None and self.expected_page:
            trees = [w for w in self.main.descendants(class_name="SysTreeView32") if w.control_id() == 129]
            if len(trees) != 1 or not trees[0].get_item(["查询[F4]", self.expected_page], exact=True).is_selected():
                raise CaptureBlocked("QUERY_PAGE_CHANGED")

    def _session_hash(self):
        # Huatai's themed account selector exposes ComboBox through UIA, not
        # necessarily a native window class named ComboBox. Never open/change it.
        ui = self.Desktop(backend="uia").window(handle=self.main.handle).wrapper_object()
        controls = [w for w in ui.descendants(control_type="ComboBox")
                    if w.element_info.automation_id == "1004"]
        if len(controls) != 1:
            raise CaptureBlocked("ACCOUNT_SELECTOR_UNAVAILABLE")
        value = (controls[0].selected_text() or "").strip()
        if not value or "*" in value:
            raise CaptureBlocked("ACCOUNT_SELECTOR_MASKED")
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def attach(self):
        self.phase = "ATTACH"
        if sys.platform != "win32":
            raise CaptureBlocked("WINDOWS_REQUIRED")
        import psutil
        from pywinauto import Desktop

        self.Desktop = Desktop
        self.user32 = ctypes.windll.user32
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p
        self.expected_page = None
        self._unlocked()
        matches = []
        for process in psutil.process_iter(["pid", "exe"]):
            try:
                if process.info["exe"] and Path(process.info["exe"]).resolve() == self.executable.resolve():
                    matches.append(process.info["pid"])
            except (OSError, psutil.Error):
                continue
        if len(matches) != 1:
            raise CaptureBlocked("CLIENT_PROCESS_NOT_UNIQUE")
        self.pid = matches[0]
        windows = Desktop(backend="win32").windows(process=self.pid, visible_only=False)
        targets = [w for w in windows if w.window_text() == "网上股票交易系统5.0"]
        if len(targets) != 1:
            raise CaptureBlocked("MAIN_WINDOW_UNAVAILABLE")
        self.main = targets[0]
        # This single initial activation is what the one-shot request authorizes.
        self.main.set_focus()
        self.phase = "SESSION_FINGERPRINT"
        self.session_hash = self._session_hash()
        self._guard()

    def export(self, page: str, destination: Path, check):
        self.phase = "SELECT_QUERY_PAGE"
        if page not in {p for _, p in PAGES}:
            raise CaptureBlocked("PAGE_NOT_ALLOWED")
        check()
        self._guard()
        trees = [w for w in self.main.descendants(class_name="SysTreeView32") if w.control_id() == 129]
        if len(trees) != 1:
            raise CaptureBlocked("QUERY_TREE_UNAVAILABLE")
        self.expected_page = None
        item = trees[0].get_item(["查询[F4]", page], exact=True)
        item.select()
        time.sleep(0.4)
        check()
        self._guard()
        if not item.is_selected():
            raise CaptureBlocked("QUERY_PAGE_NOT_SELECTED")
        self.expected_page = page
        self.phase = "REFRESH_GRID"
        self.main.type_keys("{F5}", set_foreground=False)
        time.sleep(0.6)
        check()
        self._guard()
        # Use the actual Huatai UIA surface (Custom1 / 1047), not EasyTHS's
        # upstream CVirtualGridCtrl native-class assumption.
        ui = self.Desktop(backend="uia").window(handle=self.main.handle).wrapper_object()
        grids = [w for w in ui.descendants(control_type="Pane")
                 if w.element_info.automation_id == "1047" and w.window_text() == "Custom1"
                 and w.is_visible()]
        if len(grids) != 1:
            raise CaptureBlocked("GRID_UNAVAILABLE")
        self.phase = "FOCUS_GRID"
        grids[0].set_focus()
        self._guard()
        self.phase = "EXPORT_SHORTCUT"
        grids[0].type_keys("^s", set_foreground=False)
        self.phase = "WAIT_SAVE_DIALOG"
        deadline = time.monotonic() + 4
        dialog = None
        while time.monotonic() < deadline:
            check()
            self._unlocked()
            windows = self.Desktop(backend="win32").windows(process=self.pid, visible_only=True)
            dialogs = [w for w in windows if w.class_name() == "#32770" and w.window_text() == "另存为"]
            if len(dialogs) == 1:
                dialog = dialogs[0]
                break
            if any(w.handle != self.main.handle
                   and w.class_name() not in {"tooltips_class32", "IME", "MSCTFIME UI"}
                   and not inert_broker_helper(w) for w in windows):
                self.diagnostics["save_window_inventory"] = [
                    {"class": w.class_name() if re.fullmatch(r"[A-Za-z0-9_#:. -]{1,100}", w.class_name()) else "OTHER",
                     "is_main": w.handle == self.main.handle,
                     "is_save_title": w.window_text() == "另存为",
                     "empty_title": not w.window_text(),
                     "width": w.rectangle().width(), "height": w.rectangle().height()}
                    for w in windows]
                raise CaptureBlocked("UNEXPECTED_DIALOG")
            time.sleep(0.1)
        if dialog is None:
            raise CaptureBlocked("SAVE_DIALOG_UNAVAILABLE")
        self._guard(dialog.handle)
        ui = self.Desktop(backend="uia").window(handle=dialog.handle).wrapper_object()
        self.phase = "FILENAME_EDITOR"
        editors = [w for w in ui.descendants(control_type="Edit")
                   if w.element_info.automation_id == "1001" or w.window_text().startswith("文件名")]
        if len(editors) != 1:
            raise CaptureBlocked("FILENAME_EDITOR_UNVERIFIED")
        self._guard(dialog.handle)
        editors[0].set_focus()
        self._guard(dialog.handle)
        editors[0].set_edit_text(str(destination))
        if editors[0].get_value() != str(destination):
            raise CaptureBlocked("FILENAME_NOT_CONFIRMED")
        # IFileDialog may expose a changed Value while its internal filename
        # remains the old default until an edit/change and focus-loss event.
        # Only fixed non-submitting keys in the verified filename editor.
        self.phase = "COMMIT_FILENAME"
        self._guard(dialog.handle)
        editors[0].type_keys("{END}{SPACE}{BACKSPACE}{TAB}", set_foreground=False)
        self._guard(dialog.handle)
        if editors[0].get_value() != str(destination):
            raise CaptureBlocked("FILENAME_NOT_COMMITTED")
        self.phase = "SAVE_BUTTON"
        buttons = [w for w in ui.descendants(control_type="Button")
                   if w.element_info.automation_id == "1" and w.window_text().startswith("保存")]
        if len(buttons) != 1:
            raise CaptureBlocked("SAVE_BUTTON_UNVERIFIED")
        check()
        self.phase = "BEFORE_SAVE_INVOKE"
        self._guard(dialog.handle)
        buttons[0].invoke()
        self.phase = "WAIT_SAVE_CLOSE"
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and dialog.is_visible():
            check()
            self._unlocked()
            time.sleep(0.1)
        if dialog.is_visible():
            raise CaptureBlocked("SAVE_DIALOG_DID_NOT_CLOSE")
        self._guard()

    def finish(self, check):
        self.phase = "FINISH"
        check()
        self._guard()
        # Do not switch apps or restore Codex. Only minimize our own client.
        self.main.minimize()


def main():
    root = Path(__file__).resolve().parents[2]
    job_id = sys.argv[1] if len(sys.argv) == 2 else ""
    mutex = None
    owns_mutex = False
    adapter = None
    try:
        if sys.platform != "win32" or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise CaptureBlocked("INVALID_PLATFORM_OR_JOB")
        from .desktop_capture import owner_alive

        grant = json.loads((root / f"state/private/desktop-capture/job-{job_id}.json").read_text(encoding="utf-8"))

        def check_lease():
            if (grant.get("job_id") != job_id or grant.get("authorization") != "ONE_SHOT_FOREGROUND_READ_ONLY"
                    or datetime.fromisoformat(grant["expires_at"]) <= datetime.now(UTC)
                    or not owner_alive(grant["owner_pid"], grant["owner_started"])):
                raise CaptureBlocked("LEASE_EXPIRED_OR_OWNER_EXITED")

        check_lease()
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        mutex = kernel32.CreateMutexW(None, False, "Local\\MoneyMoreHuataiReadOnlyCapture")
        owns_mutex = bool(mutex) and kernel32.WaitForSingleObject(mutex, 0) in (0, 0x80)
        if not owns_mutex:
            raise CaptureBlocked("CAPTURE_ALREADY_RUNNING")
        profile = json.loads((root / "state/private/desktop-client.json").read_text(encoding="utf-8"))
        executable = Path(profile.get("executable", ""))
        if not executable.is_absolute() or not executable.is_file() or executable.name.lower() != "xiadan.exe":
            raise CaptureBlocked("CLIENT_NOT_CONFIGURED")
        adapter = HuataiForegroundAdapter(executable)
        # Let the initiating browser click/release complete before the single
        # foreground activation. Otherwise the browser can take focus back as
        # the POST response updates its page, falsely interrupting our capture.
        adapter.phase = "ACTIVATION_GRACE"
        for _ in range(20):
            check_lease()
            if (root / f"state/private/desktop-capture/stop-{job_id}").exists():
                raise CaptureBlocked("STOP_REQUESTED")
            time.sleep(0.1)
        result = capture(root, job_id, adapter, lease_check=check_lease)
    except CaptureBlocked as error:
        result = {"status": "BLOCKED", "reason_code": str(error), "submitted": False}
    except Exception as error:  # noqa: BLE001 - never expose arbitrary COM error text/account values.
        result = {"status": "BLOCKED", "reason_code": "CAPTURE_ADAPTER_ERROR", "submitted": False}
        if adapter is not None and re.fullmatch(r"[A-Za-z_]{1,80}", type(error).__name__):
            adapter.diagnostics["error_type"] = type(error).__name__
    finally:
        if mutex:
            if owns_mutex:
                kernel32.ReleaseMutex(mutex)
            kernel32.CloseHandle(mutex)
    if adapter is not None:
        result["phase"] = adapter.phase
        if adapter.diagnostics:
            result["diagnostics"] = adapter.diagnostics
    if re.fullmatch(r"[a-f0-9]{32}", job_id):
        # Sanitized, private audit result only; never persist UI text/account values.
        audit_dir = root / "state/private/desktop-capture"
        audit_dir.mkdir(parents=True, exist_ok=True)
        temporary = audit_dir / f"result-{job_id}.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(audit_dir / f"result-{job_id}.json")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
