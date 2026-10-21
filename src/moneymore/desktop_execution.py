"""Local desktop preparation ledger, isolated from paper and real broker ledgers.

There is deliberately no submit/cancel endpoint or foreground input primitive.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


class DesktopExecutionService:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.database = self.root / "state/desktop-execution.sqlite3"
        self._probe_lock = threading.Lock()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS desktop_jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, payload TEXT NOT NULL, result TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS desktop_intents (
                id TEXT PRIMARY KEY, source_account TEXT NOT NULL,
                source_order_key TEXT NOT NULL, source_fingerprint TEXT NOT NULL,
                strategy_id TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
                quantity INTEGER NOT NULL, signal_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PREPARED_ONLY', created_at TEXT NOT NULL,
                UNIQUE(source_account, source_order_key, source_fingerprint))""")

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def _source_orders(self):
        path = self.root / "state/paper_orders.sqlite3"
        if not path.exists():
            return []
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute(
                    "SELECT account_id,idempotency_key,strategy_id,symbol,side,quantity,signal_date "
                    "FROM orders WHERE status='PENDING' ORDER BY signal_date DESC LIMIT 500"
                )
            ]

    def preview(self):
        return self._source_orders()

    def configure_client(self, executable: str):
        path = Path(executable)
        if not path.is_absolute() or path.suffix.lower() != ".exe" or not path.is_file():
            raise ValueError("请选择已安装交易客户端的绝对exe路径；不会启动或安装软件")
        target = self.root / "state/private/desktop-client.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                {"executable": str(path.resolve()), "background_only": True, "mode": "READ_ONLY"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(target)
        return {"status": "CONFIGURED_READ_ONLY", "launched": False}

    def prepare(self, source_account: str):
        orders = [o for o in self._source_orders() if o["account_id"] == source_account]
        if not orders:
            raise ValueError("该模拟策略没有待处理委托；未生成桌面计划")
        # Never combine independent 100%-capital paper accounts into one real account.
        ids = []
        with self._connect() as db:
            for order in orders:
                fingerprint = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
                intent_id = hashlib.sha256(
                    f"{source_account}:{order['idempotency_key']}:{fingerprint}".encode()
                ).hexdigest()[:32]
                db.execute(
                    """INSERT OR IGNORE INTO desktop_intents
                    (id,source_account,source_order_key,source_fingerprint,strategy_id,symbol,side,
                     quantity,signal_date,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        intent_id,
                        source_account,
                        order["idempotency_key"],
                        fingerprint,
                        order["strategy_id"],
                        order["symbol"],
                        order["side"],
                        order["quantity"],
                        order["signal_date"],
                        datetime.now(UTC).isoformat(),
                    ),
                )
                ids.append(intent_id)
        return {"status": "PREPARED_ONLY", "intent_ids": ids, "submitted": False}

    def probe(self):
        if not self._probe_lock.acquire(blocking=False):
            raise ValueError("只读探测正在进行，请勿重复启动")
        job_id = uuid.uuid4().hex
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO desktop_jobs VALUES (?,?,?,?,?,NULL)",
                    (
                        job_id,
                        "READ_ONLY_CAPABILITY_PROBE",
                        "RUNNING",
                        datetime.now(UTC).isoformat(),
                        '{"background_only":true}',
                    ),
                )
            try:
                process = subprocess.run(
                    [sys.executable, "-X", "utf8", "-m", "moneymore.desktop_readonly_worker"],
                    cwd=self.root,
                    capture_output=True,
                    timeout=25,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if process.returncode:
                    # stderr can contain account information. Do not echo it to the browser.
                    result = {"status": "BLOCKED", "reason": "只读代理启动失败或依赖未安装"}
                else:
                    result = json.loads(process.stdout)
                    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                        raise ValueError("Invalid local worker result")
            except subprocess.TimeoutExpired:
                result = {"status": "BLOCKED", "reason": "控件读取超时；未抢焦点或重复尝试"}
            except (OSError, ValueError, TypeError, UnicodeError):
                result = {"status": "BLOCKED", "reason": "本地代理不可用或返回格式异常"}
            with self._connect() as db:
                db.execute(
                    "UPDATE desktop_jobs SET status=?,result=? WHERE id=?",
                    (
                        result.get("status", "BLOCKED"),
                        json.dumps(result, ensure_ascii=False),
                        job_id,
                    ),
                )
            return {"job_id": job_id, **result}
        finally:
            self._probe_lock.release()

    def snapshot(self):
        source_orders = self.preview()
        fingerprints = {
            hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()
            for o in source_orders
        }
        with self._connect() as db:
            jobs = [
                dict(r)
                for r in db.execute("SELECT * FROM desktop_jobs ORDER BY created_at DESC LIMIT 10")
            ]
            intents = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM desktop_intents ORDER BY created_at DESC LIMIT 200"
                )
            ]
        for job in jobs:
            job["result"] = json.loads(job["result"]) if job["result"] else None
        for intent in intents:
            if intent["source_fingerprint"] not in fingerprints:
                intent["status"] = "SOURCE_STALE"
            intent["broker_order_id"] = None
            intent["confirmed_filled_quantity"] = None
            intent["receipt_status"] = "未取得券商回执，不能认定已委托或成交"
        return {
            "mode": "READ_ONLY_AND_PREPARATION",
            "foreground_input_allowed": False,
            "live_submission_supported": False,
            "broker_receipts_supported": False,
            "jobs": jobs,
            "intents": intents,
            "source_orders": source_orders,
            "configured_client": json.loads(
                (self.root / "state/private/desktop-client.json").read_text(encoding="utf-8")
            )
            if (self.root / "state/private/desktop-client.json").exists()
            else None,
            "limitations": [
                "不激活窗口，不移动鼠标，不读取或填写密码，不提交或撤销委托",
                "准备计划不是下单；模拟成交不是券商成交",
                "后台读取须按实际客户端验收；不支持时阻断，不抢占你的桌面",
                "锁屏、UAC安全桌面、跨用户会话不视为可用环境",
            ],
        }
