"""Explicit observation links and conservative reconciliation, never execution.

An account alias or manually supplied association is not broker authentication.
Even internally consistent receipts remain unverified and cannot update fills.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path


def reconcile(link: dict, snapshots: list[dict]) -> dict:
    issues, fills, cumulative = set(), {}, []
    latest_orders = latest_trades = None
    observed_times = {}
    for snapshot in sorted(snapshots, key=lambda s: (s["source_mtime"], s.get("imported_at", ""), s["id"])):
        if (snapshot.get("account_ref", link["account_ref"]) != link["account_ref"]
                or snapshot.get("data_date", link["trade_date"]) != link["trade_date"]):
            issues.add("SNAPSHOT_SCOPE_MISMATCH")
            continue
        matching = []
        for row in snapshot["rows"]:
            if row.get("order_id") != link["broker_order_id"]:
                continue
            if not row.get("market"):
                issues.add("MARKET_UNKNOWN")
                continue
            if row["market"] != link["market"]:
                continue
            if (row.get("code"), row.get("side")) != (link["code"], link["side"]):
                issues.add("SECURITY_OR_SIDE_CHANGED")
                continue
            if not row.get("date_observed") or row.get("trade_date") != link["trade_date"]:
                issues.add("DATE_UNVERIFIED")
                continue
            matching.append(row)
        if snapshot["kind"] == "orders":
            latest_orders = matching
            if len(matching) > 1:
                issues.add("AMBIGUOUS_ORDER")
            for row in matching:
                if row["quantity"] != link["quantity"] or Decimal(row["price"]) != Decimal(link["price"]):
                    issues.add("ORDER_TERMS_CHANGED")
                timestamp = snapshot["source_mtime"]
                if timestamp in observed_times and observed_times[timestamp] != row["filled_quantity"]:
                    issues.add("SOURCE_TIME_COLLISION")
                observed_times[timestamp] = row["filled_quantity"]
                if cumulative and row["filled_quantity"] < cumulative[-1]:
                    issues.add("CUMULATIVE_FILL_REGRESSION")
                cumulative.append(row["filled_quantity"])
        elif snapshot["kind"] == "trades":
            latest_trades = {r["trade_id"] for r in matching}
            for row in matching:
                fingerprint = (row["code"], row["side"], row["quantity"], Decimal(row["price"]), row["time"])
                previous = fills.get(row["trade_id"])
                if previous is not None and previous != fingerprint:
                    issues.add("TRADE_ID_CONFLICT")
                else:
                    fills[row["trade_id"]] = fingerprint
    quantity = sum(f[2] for f in fills.values())
    if quantity > link["quantity"]:
        issues.add("OVERFILL")
    if latest_orders == []:
        issues.add("ORDER_MISSING_IN_LATEST")
    if latest_trades is not None and set(fills) - latest_trades:
        issues.add("TRADE_MISSING_IN_LATEST")
    if cumulative and latest_trades is not None and cumulative[-1] != quantity:
        issues.add("SUMMARY_DETAIL_MISMATCH")
    status = "REVIEW_REQUIRED" if issues else (
        "WAITING_ORDER" if not cumulative else "WAITING_TRADES" if latest_trades is None else
        "OBSERVED_FULL" if quantity == link["quantity"] else
        "OBSERVED_PARTIAL" if quantity else "OBSERVED_ORDER_ONLY")
    return {"status": status, "issues": sorted(issues), "trade_count": len(fills),
            "latest_source_mtime": max((s["source_mtime"] for s in snapshots), default=None),
            "observed_filled_quantity": quantity if latest_trades is not None else None,
            "observed_cumulative_quantity": cumulative[-1] if cumulative else None,
            "confirmed_filled_quantity": None, "broker_receipts_verified": False,
            "account_verified": False, "ledger_updated": False, "allow_resubmit": False}


class DesktopReceiptStore:
    def __init__(self, root: Path):
        self.database = Path(root) / "state/desktop-execution.sqlite3"
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS desktop_receipt_links (
                id TEXT PRIMARY KEY, intent_id TEXT NOT NULL UNIQUE,
                account_ref TEXT NOT NULL, trade_date TEXT NOT NULL, market TEXT NOT NULL,
                broker_order_id TEXT NOT NULL, order_snapshot_id TEXT NOT NULL,
                created_at TEXT NOT NULL, note TEXT NOT NULL, expected TEXT NOT NULL,
                UNIQUE(account_ref,trade_date,market,broker_order_id))""")

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def bind(self, *, intent_id: str, snapshot_id: str, broker_order_id: str, note: str):
        if not note.strip() or len(note) > 200:
            raise ValueError("请填写不超过200字的观察关联说明")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            intent = db.execute("SELECT * FROM desktop_intents WHERE id=?", (intent_id,)).fetchone()
            snapshot = db.execute("SELECT * FROM desktop_export_snapshots WHERE id=?", (snapshot_id,)).fetchone()
            if intent is None or snapshot is None or snapshot["kind"] != "orders":
                raise ValueError("必须选择已有桌面计划及委托快照")
            rows = [r for r in json.loads(snapshot["payload"])["rows"]
                    if r.get("order_id") == broker_order_id]
            if len(rows) != 1:
                raise ValueError("该券商委托编号不存在或不唯一")
            order = rows[0]
            if order["quantity"] <= 0:
                raise ValueError("零数量委托不能关联为买卖计划")
            if not order.get("market") or not order.get("date_observed"):
                raise ValueError("委托导出缺少已识别市场或行内日期；不能只靠别名关联")
            if (intent["symbol"], intent["side"], intent["quantity"]) != (
                f"{order['code']}.{order['market']}", order["side"], order["quantity"]):
                raise ValueError("计划与券商委托的证券、市场、方向或数量不一致")
            if snapshot["data_date"] < intent["signal_date"]:
                raise ValueError("委托日期早于计划信号日")
            existing = db.execute("""SELECT * FROM desktop_receipt_links WHERE intent_id=? OR
                (account_ref=? AND trade_date=? AND market=? AND broker_order_id=?)""", (
                    intent_id, snapshot["account_ref"], snapshot["data_date"],
                    order["market"], broker_order_id)).fetchall()
            if existing:
                if len(existing) == 1 and all(existing[0][k] == v for k, v in {
                    "intent_id": intent_id, "account_ref": snapshot["account_ref"],
                    "trade_date": snapshot["data_date"], "market": order["market"],
                    "broker_order_id": broker_order_id}.items()):
                    return {"status": "LINKED_OBSERVATION_ONLY", "link_id": existing[0]["id"], "submitted": False}
                raise ValueError("计划或券商委托已有关联，禁止重复归属或静默改绑")
            link_id = uuid.uuid4().hex
            db.execute("INSERT INTO desktop_receipt_links VALUES (?,?,?,?,?,?,?,?,?,?)", (
                link_id, intent_id, snapshot["account_ref"], snapshot["data_date"], order["market"],
                broker_order_id, snapshot_id, datetime.now(UTC).isoformat(), note.strip(),
                json.dumps({k: order[k] for k in ("code", "side", "quantity", "price")}, sort_keys=True)))
        return {"status": "LINKED_OBSERVATION_ONLY", "link_id": link_id, "submitted": False}

    def snapshot(self):
        with self._connect() as db:
            links = db.execute("SELECT * FROM desktop_receipt_links ORDER BY created_at DESC LIMIT 200").fetchall()
            result = []
            for record in links:
                link = dict(record)
                link.update(json.loads(link.pop("expected")))
                metadata = db.execute("""SELECT id,length(payload) AS bytes FROM desktop_export_snapshots
                    WHERE account_ref=? AND data_date=? AND kind IN ('orders','trades') LIMIT 101""",
                    (link["account_ref"], link["trade_date"])).fetchall()
                if len(metadata) > 100 or sum(r["bytes"] for r in metadata) > 10_000_000:
                    report = {"status": "REVIEW_REQUIRED", "issues": ["HISTORY_LIMIT_EXCEEDED"],
                              "observed_filled_quantity": None, "confirmed_filled_quantity": None,
                              "broker_receipts_verified": False, "allow_resubmit": False}
                else:
                    snapshots = []
                    for row in metadata:
                        source = db.execute("SELECT * FROM desktop_export_snapshots WHERE id=?", (row["id"],)).fetchone()
                        snapshots.append({**dict(source), "rows": json.loads(source["payload"])["rows"]})
                    report = reconcile(link, snapshots)
                result.append({**link, "reconciliation": report})
        return {"links": result, "mode": "EXPLICIT_OBSERVATION_ONLY",
                "live_submission_supported": False,
                "notice": "关联是人工观察标记，不证明券商账户身份，不修改模拟成交；任何状态均不自动补单。"}
