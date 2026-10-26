"""One-shot, explicit foreground lease. No schedules or trading endpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from .desktop_exports import DesktopExportStore

ACTIVE = ("QUEUED", "RUNNING", "STOP_REQUESTED")


def owner_alive(pid, started):
    try:
        import psutil
    except ImportError:
        return False
    try:
        return psutil.Process(pid).create_time() == started
    except (psutil.Error, ValueError, OSError):
        return False


class DesktopCaptureService:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.database = self.root / "state/desktop-execution.sqlite3"
        self.private = self.root / "state/private/desktop-capture"
        self.private.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS desktop_capture_jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL, owner_pid INTEGER NOT NULL, owner_started REAL NOT NULL,
                result TEXT)""")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS desktop_capture_single_active
                ON desktop_capture_jobs((1)) WHERE status IN ('QUEUED','RUNNING','STOP_REQUESTED')""")
        self._recover()

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def _recover(self):
        with self._connect() as db:
            for job in db.execute("SELECT * FROM desktop_capture_jobs WHERE status IN ('QUEUED','RUNNING','STOP_REQUESTED')"):
                expired = datetime.fromisoformat(job["expires_at"]) < datetime.now(UTC)
                if expired or not owner_alive(job["owner_pid"], job["owner_started"]):
                    (self.private / f"stop-{job['id']}").touch(exist_ok=True)
                    db.execute("UPDATE desktop_capture_jobs SET status='INTERRUPTED',result=? WHERE id=?",
                               (json.dumps({"reason_code": "LEASE_EXPIRED_OR_OWNER_EXITED", "submitted": False}), job["id"]))

    def start(self):
        if not (self.root / "state/private/desktop-client.json").is_file():
            raise ValueError("先配置已登录华泰客户端的exe路径；不会自动启动或登录")
        self._recover()
        try:
            import psutil
        except ImportError:
            raise ValueError("未安装desktop可选依赖，采集未启动") from None
        job_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=100)
        started = psutil.Process(os.getpid()).create_time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            recent = db.execute("SELECT created_at FROM desktop_capture_jobs ORDER BY created_at DESC LIMIT 1").fetchone()
            if recent and (now - datetime.fromisoformat(recent["created_at"])).total_seconds() < 15:
                raise ValueError("采集请求过于频繁，请稍后再试")
            try:
                db.execute("INSERT INTO desktop_capture_jobs VALUES (?,?,?,?,?,?,NULL)", (
                    job_id, "QUEUED", now.isoformat(), expires.isoformat(), os.getpid(), started))
            except sqlite3.IntegrityError:
                raise ValueError("已有前台采集任务，请等待完成或停止") from None
        grant = {"job_id": job_id, "authorization": "ONE_SHOT_FOREGROUND_READ_ONLY",
                 "owner_pid": os.getpid(), "owner_started": started, "expires_at": expires.isoformat()}
        try:
            (self.private / f"job-{job_id}.json").write_text(json.dumps(grant), encoding="utf-8")
            threading.Thread(target=self._run, args=(job_id,), daemon=True,
                             name=f"huatai-readonly-{job_id[:8]}").start()
        except (OSError, RuntimeError):
            self._finish(job_id, {"status": "BLOCKED", "reason_code": "WORKER_START_FAILED", "submitted": False})
            raise ValueError("采集代理启动失败，未执行前台回退") from None
        return {"job_id": job_id, "status": "QUEUED", "foreground_authorized_once": True, "submitted": False}

    def stop(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("作业编号无效")
        with self._connect() as db:
            job = db.execute("SELECT status FROM desktop_capture_jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise ValueError("采集作业不存在")
            if job["status"] not in ACTIVE:
                return {"status": job["status"], "submitted": False}
            (self.private / f"stop-{job_id}").touch(exist_ok=True)
            db.execute("UPDATE desktop_capture_jobs SET status='STOP_REQUESTED' WHERE id=?", (job_id,))
        return {"status": "STOP_REQUESTED", "submitted": False}

    def _finish(self, job_id, result):
        with self._connect() as db:
            # Recovery/stop must not later be overwritten as a successful job.
            current = db.execute("SELECT status FROM desktop_capture_jobs WHERE id=?", (job_id,)).fetchone()
            if current is None or current["status"] == "INTERRUPTED":
                return
            if current["status"] == "STOP_REQUESTED":
                result = {"status": "STOPPED", "reason_code": "STOP_REQUESTED", "submitted": False,
                          "imports": result.get("imports", [])}
            db.execute("UPDATE desktop_capture_jobs SET status=?,result=? WHERE id=?",
                       (result["status"], json.dumps(result, ensure_ascii=False), job_id))

    def _run(self, job_id):
        result = {"status": "BLOCKED", "reason_code": "CAPTURE_WORKER_ERROR", "submitted": False}
        imports = []
        try:
            with self._connect() as db:
                updated = db.execute("UPDATE desktop_capture_jobs SET status='RUNNING' WHERE id=? AND status='QUEUED'", (job_id,)).rowcount
            if not updated:
                self._finish(job_id, {"status": "STOPPED", "submitted": False})
                return
            process = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "moneymore.desktop_capture_worker", job_id],
                cwd=self.root, capture_output=True, timeout=85, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            output = json.loads(process.stdout) if process.returncode == 0 else None
            if not isinstance(output, dict):
                raise TypeError("invalid output")
            if output.get("status") != "CAPTURED_UNVERIFIED":
                reason = output.get("reason_code", "INVALID_WORKER_RESULT")
                result["reason_code"] = reason if re.fullmatch(r"[A-Z_]{1,80}", str(reason)) else "INVALID_WORKER_RESULT"
                phase = output.get("phase")
                if isinstance(phase, str) and re.fullmatch(r"[A-Z_]{1,80}", phase):
                    result["phase"] = phase
            else:
                date = datetime.now(UTC).astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d")
                if output.get("job_id") != job_id or output.get("trade_date") != date:
                    raise ValueError("wrong job/date")
                files = output.get("exports", [])
                if len(files) != 3 or {f["kind"] for f in files} != {"positions", "orders", "trades"}:
                    raise ValueError("incomplete export")
                folder = self.root / "state/private/huatai-acceptance"
                for file in files:
                    if file["filename"] != f"{file['kind']}-{date}-{job_id}.xls":
                        raise ValueError("wrong file")
                    with (folder / file["filename"]).open("rb") as source:
                        data = source.read(5_000_001)
                    if len(data) > 5_000_000 or hashlib.sha256(data).hexdigest() != file["sha256"]:
                        raise ValueError("file changed")
                if (self.private / f"stop-{job_id}").exists():
                    self._finish(job_id, {"status": "STOPPED", "submitted": False})
                    return
                store = DesktopExportStore(self.root)
                for file in files:
                    if (self.private / f"stop-{job_id}").exists():
                        raise ValueError("stopped")
                    imports.append(store.ingest(filename=file["filename"], kind=file["kind"],
                                                account_ref="huatai_local", trade_date=date,
                                                collection_job_id=job_id))
                result = {"status": "COMPLETED_UNVERIFIED", "imports": imports,
                          "account_verified": False, "broker_receipts_verified": False,
                          "foreground_used": True, "submitted": False}
        except subprocess.TimeoutExpired:
            (self.private / f"stop-{job_id}").touch(exist_ok=True)
            result["reason_code"] = "WORKER_TIMEOUT"
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
            # Never expose stderr, window contents, filenames supplied by worker or credentials.
            result["reason_code"] = "WORKER_OR_IMPORT_FAILED"
        if imports and result["status"] == "BLOCKED":
            result.update(status="PARTIAL_IMPORT", imports=imports)
        self._finish(job_id, result)

    def snapshot(self):
        self._recover()
        with self._connect() as db:
            jobs = [dict(row) for row in db.execute("SELECT * FROM desktop_capture_jobs ORDER BY created_at DESC LIMIT 10")]
        for job in jobs:
            job.pop("owner_pid")
            job.pop("owner_started")
            job["result"] = json.loads(job["result"]) if job["result"] else None
        return {"mode": "ONE_SHOT_FOREGROUND_READ_ONLY", "jobs": jobs,
                "scheduled": False, "background_invisible": False,
                "notice": "仅显式授权本次采集；会临时占用前台。切换窗口、锁屏、异常弹窗或账户选择变化将阻断。"}
