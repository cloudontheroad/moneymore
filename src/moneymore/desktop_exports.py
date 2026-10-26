"""Private local export snapshots; no GUI, broker or paper-ledger writes."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from .desktop_tables import PROFILES, TableValidationError, parse_huatai_export

FILE_NAME = re.compile(r"(?:positions|orders|trades)[A-Za-z0-9_.-]*\.(?:xls|tsv|csv)")


class DesktopExportStore:
    def __init__(self, root: Path):
        self.folder = Path(root) / "state/private/huatai-acceptance"
        self.database = Path(root) / "state/desktop-execution.sqlite3"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS desktop_export_snapshots (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, account_ref TEXT NOT NULL,
                data_date TEXT NOT NULL, source_name TEXT NOT NULL,
                source_sha256 TEXT NOT NULL, source_mtime TEXT NOT NULL,
                imported_at TEXT NOT NULL, payload TEXT NOT NULL)""")

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def _file(self, filename: str) -> Path:
        if not FILE_NAME.fullmatch(filename):
            raise ValueError("只能读取私密导出目录内的持仓、委托或成交文件名")
        path = self.folder / filename
        if path.resolve().parent != self.folder.resolve() or path.is_symlink():
            raise ValueError("导出文件不得指向目录外")
        return path

    def candidates(self):
        if not self.folder.is_dir():
            return []
        result = []
        # Bound both enumeration and UI payload. Never read contents on GET.
        for index, path in enumerate(self.folder.iterdir()):
            if index >= 500:
                break
            if not FILE_NAME.fullmatch(path.name):
                continue
            try:
                checked = self._file(path.name)
                stat = checked.stat()
                if not checked.is_file() or stat.st_size > 5_000_000:
                    continue
                result.append({"filename": path.name, "bytes": stat.st_size,
                               "source_mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()})
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda r: r["source_mtime"], reverse=True)[:100]

    def ingest(self, *, filename: str, kind: str, account_ref: str, trade_date: str,
               collection_job_id: str | None = None):
        path = self._file(filename)
        if kind not in PROFILES or not filename.startswith(kind):
            raise ValueError("文件名类型与导入表类型不一致")
        try:
            before = path.stat()
            with path.open("rb") as source:
                data = source.read(5_000_001)
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise ValueError("导出文件正在变化，请保存完成后重试")
            parsed = parse_huatai_export(data, kind=kind, account_ref=account_ref,
                                         trade_date=trade_date)
        except OSError:
            raise ValueError("本地导出文件不存在或不可读取") from None
        except TableValidationError as error:
            raise ValueError(str(error)) from None
        now = datetime.now(UTC)
        source_time = datetime.fromtimestamp(after.st_mtime, UTC)
        if source_time > now:
            raise ValueError("导出文件时间在未来，请核查系统时间")
        identity_parts = [kind, account_ref, trade_date, parsed["source_sha256"], parsed["parser_profile"]]
        if collection_job_id is not None:
            if not re.fullmatch(r"[a-f0-9]{32}", collection_job_id):
                raise ValueError("采集作业编号无效")
            identity_parts.append(collection_job_id)
            parsed["collection_job_id"] = collection_job_id
        identity = hashlib.sha256(json.dumps(identity_parts).encode()).hexdigest()
        with self._connect() as db:
            db.execute("""INSERT OR IGNORE INTO desktop_export_snapshots
                VALUES (?,?,?,?,?,?,?,?,?)""", (
                identity, kind, account_ref, trade_date, filename, parsed["source_sha256"],
                source_time.isoformat(), now.isoformat(), json.dumps(parsed, ensure_ascii=False)))
        return {"status": "IMPORTED_UNVERIFIED", "snapshot_id": identity,
                "row_count": len(parsed["rows"]), "ledger_updated": False,
                "submitted": False}

    def snapshot(self):
        now = datetime.now(UTC)
        with self._connect() as db:
            records = db.execute("""SELECT * FROM desktop_export_snapshots
                ORDER BY imported_at DESC LIMIT 30""").fetchall()
        items = []
        for record in records:
            item = dict(record)
            parsed = json.loads(item.pop("payload"))
            item["row_count"] = len(parsed["rows"])
            item["rows"] = parsed["rows"][:100]
            item["rows_truncated"] = len(parsed["rows"]) > 100
            item["age_seconds"] = max(0, int((now - datetime.fromisoformat(item["source_mtime"])).total_seconds()))
            current_date = now.astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d")
            item["freshness"] = "RECENT_UNVERIFIED" if (
                item["age_seconds"] <= 300 and item["data_date"] == current_date
            ) else "STALE"
            item["account_verified"] = False
            item["date_verified"] = False
            item["broker_receipts_verified"] = False
            item["status"] = "PARSED_UNVERIFIED"
            items.append(item)
        return {"mode": "LOCAL_EXPORT_SNAPSHOTS", "candidates": self.candidates(),
                "snapshots": items, "live": False, "automated_gui_collection": False,
                "submission_supported": False,
                "notice": "仅解析已保存导出，不会刷新券商客户端；导入时间不等于行情或券商数据时间。"}
