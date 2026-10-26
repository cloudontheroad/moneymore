import copy
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from moneymore import desktop_routes
from moneymore.desktop_execution import DesktopExecutionService
from moneymore.desktop_receipts import reconcile

LINK = {"broker_order_id": "000023", "account_ref": "test", "trade_date": "20261022",
        "market": "SH", "code": "600036", "side": "BUY", "quantity": 100, "price": "10.50"}
ORDER = {"order_id": "000023", "market": "SH", "code": "600036", "side": "BUY",
         "quantity": 100, "price": "10.50", "filled_quantity": 0,
         "trade_date": "20261022", "date_observed": True}
TRADE = {**ORDER, "quantity": 50, "trade_id": "000001", "time": "09:31:01"}


def snap(kind, rows, sequence=1, **extra):
    return {"id": str(sequence), "source_mtime": f"2026-10-22T01:31:{sequence:02}+00:00",
            "kind": kind, "rows": rows, **extra}


def test_missing_detail_is_unknown_not_zero_and_no_automatic_retry():
    report = reconcile(LINK, [snap("orders", [ORDER])])
    assert report["status"] == "WAITING_TRADES"
    assert report["observed_filled_quantity"] is None
    assert report["confirmed_filled_quantity"] is None
    assert not report["allow_resubmit"] and not report["ledger_updated"]


def test_observed_zero_order_partial_and_full_are_not_confirmed_fills():
    assert reconcile(LINK, [snap("orders", [ORDER]), snap("trades", [], 2)])["status"] == "OBSERVED_ORDER_ONLY"
    partial = reconcile(LINK, [snap("orders", [{**ORDER, "filled_quantity": 50}]), snap("trades", [TRADE], 2)])
    assert partial["status"] == "OBSERVED_PARTIAL" and partial["observed_filled_quantity"] == 50
    full = reconcile(LINK, [snap("orders", [{**ORDER, "filled_quantity": 100}]),
                            snap("trades", [{**TRADE, "quantity": 100}], 2)])
    assert full["status"] == "OBSERVED_FULL" and not full["broker_receipts_verified"]


def test_repeated_fill_snapshots_are_deduplicated_with_numeric_prices():
    report = reconcile(LINK, [snap("orders", [{**ORDER, "filled_quantity": 50, "price": "10.5"}]),
                             snap("trades", [TRADE], 2),
                             snap("trades", [{**TRADE, "price": "10.500"}], 3)])
    assert report["observed_filled_quantity"] == 50 and report["trade_count"] == 1
    assert report["status"] == "OBSERVED_PARTIAL"


@pytest.mark.parametrize("change,issue", [
    ({"code": "600000"}, "SECURITY_OR_SIDE_CHANGED"),
    ({"side": "SELL"}, "SECURITY_OR_SIDE_CHANGED"),
    ({"quantity": 200}, "ORDER_TERMS_CHANGED"),
    ({"price": "20"}, "ORDER_TERMS_CHANGED"),
    ({"market": None}, "MARKET_UNKNOWN"),
    ({"date_observed": False}, "DATE_UNVERIFIED"),
])
def test_changed_terms_require_review(change, issue):
    report = reconcile(LINK, [snap("orders", [{**ORDER, **change}]), snap("trades", [], 2)])
    assert issue in report["issues"] and report["status"] == "REVIEW_REQUIRED"


def test_regression_and_summary_detail_disagreement():
    report = reconcile(LINK, [snap("orders", [{**ORDER, "filled_quantity": 50}]),
                             snap("orders", [ORDER], 2), snap("trades", [TRADE], 3)])
    assert {"CUMULATIVE_FILL_REGRESSION", "SUMMARY_DETAIL_MISMATCH"} <= set(report["issues"])


def test_trade_identity_conflict_and_disappearance_do_not_erase_fills():
    report = reconcile(LINK, [snap("orders", [{**ORDER, "filled_quantity": 50}]),
                             snap("trades", [TRADE], 2),
                             snap("trades", [{**TRADE, "price": "11"}], 3),
                             snap("trades", [], 4)])
    assert {"TRADE_ID_CONFLICT", "TRADE_MISSING_IN_LATEST"} <= set(report["issues"])
    assert report["observed_filled_quantity"] == 50


def test_overfill_and_missing_order_are_review_not_complete():
    report = reconcile(LINK, [snap("orders", [ORDER]), snap("orders", [], 2),
                             snap("trades", [{**TRADE, "quantity": 200}], 3)])
    assert {"OVERFILL", "ORDER_MISSING_IN_LATEST"} <= set(report["issues"])


def test_cross_account_or_date_and_source_time_collisions_are_rejected():
    assert "SNAPSHOT_SCOPE_MISMATCH" in reconcile(LINK, [snap("orders", [ORDER], account_ref="other")])["issues"]
    same_time = snap("orders", [{**ORDER, "filled_quantity": 50}])
    assert "SOURCE_TIME_COLLISION" in reconcile(LINK, [snap("orders", [ORDER]), same_time])["issues"]


def setup_service(tmp_path):
    source = tmp_path / "state/paper_orders.sqlite3"
    source.parent.mkdir(parents=True)
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE orders (account_id TEXT,idempotency_key TEXT,strategy_id TEXT,"
                   "symbol TEXT,side TEXT,quantity INTEGER,signal_date TEXT,status TEXT)")
        db.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", [
            ("base", "one", "factor", "600036.SH", "BUY", 100, "20261021", "PENDING"),
            ("other", "two", "gru", "600036.SH", "BUY", 100, "20261021", "PENDING")])
    service = DesktopExecutionService(tmp_path)
    intents = [service.prepare(a)["intent_ids"][0] for a in ("base", "other")]
    folder = tmp_path / "state/private/huatai-acceptance"
    folder.mkdir(parents=True)
    text = ("委托日期\t证券代码\t证券名称\t委托编号\t操作\t委托数量\t成交数量\t委托价格\t委托时间\t市场类别\n"
            "20261022\t600036\t虚构证券\t000023\t买入\t100\t0\t10.50\t09:31:00\t上海Ａ\n")
    (folder / "orders-test.xls").write_bytes(text.encode("gbk"))
    imported = service.import_exports(filename="orders-test.xls", kind="orders", account_ref="test", trade_date="20261022")
    return service, intents, imported["snapshot_id"], source


def test_explicit_link_is_unique_idempotent_persistent_and_never_mutates_source(tmp_path):
    service, intents, snapshot_id, source = setup_service(tmp_path)
    before = source.read_bytes()
    kwargs = {"intent_id": intents[0], "snapshot_id": snapshot_id, "broker_order_id": "000023", "note": "只做观察关联"}
    first = service.bind_receipt(**kwargs)
    assert service.bind_receipt(**kwargs) == first and not first["submitted"]
    with pytest.raises(ValueError, match="重复归属"):
        service.bind_receipt(**{**kwargs, "intent_id": intents[1]})
    result = DesktopExecutionService(tmp_path).snapshot()
    assert len(result["receipt_links"]["links"]) == 1
    assert result["receipt_links"]["links"][0]["reconciliation"]["status"] == "WAITING_TRADES"
    assert all(i["confirmed_filled_quantity"] is None for i in result["intents"])
    assert source.read_bytes() == before


def test_link_rejects_missing_or_mismatched_order(tmp_path):
    service, intents, snapshot_id, _ = setup_service(tmp_path)
    with pytest.raises(ValueError, match="不存在"):
        service.bind_receipt(intent_id=intents[0], snapshot_id=snapshot_id, broker_order_id="absent", note="观察")
    with service._connect() as db:
        db.execute("UPDATE desktop_intents SET side='SELL' WHERE id=?", (intents[0],))
    with pytest.raises(ValueError, match="不一致"):
        service.bind_receipt(intent_id=intents[0], snapshot_id=snapshot_id, broker_order_id="000023", note="观察")


def test_link_route_requires_local_action_and_observation_acknowledgement(tmp_path, monkeypatch):
    service, intents, snapshot_id, _ = setup_service(tmp_path)
    monkeypatch.setattr(desktop_routes, "service", lambda: service)
    app = FastAPI()
    app.include_router(desktop_routes.router)
    client = TestClient(app)
    body = {"intent_id": intents[0], "snapshot_id": snapshot_id, "broker_order_id": "000023",
            "note": "观察测试", "acknowledge_observation_only": True}
    headers = {"Origin": "http://127.0.0.1:3000", "X-MoneyMore-Desktop": "prepare-v1"}
    assert client.post("/api/desktop-execution/receipts/link", json=body).status_code == 403
    invalid = copy.copy(body)
    invalid["acknowledge_observation_only"] = False
    assert client.post("/api/desktop-execution/receipts/link", json=invalid, headers=headers).status_code == 422
    assert client.post("/api/desktop-execution/receipts/link", json=body, headers=headers).json()["submitted"] is False


@pytest.mark.parametrize("missing", ["market", "date_observed"])
def test_legacy_snapshots_without_market_or_date_cannot_be_bound(tmp_path, missing):
    service, intents, snapshot_id, _ = setup_service(tmp_path)
    import json
    with service._connect() as db:
        payload = json.loads(db.execute("SELECT payload FROM desktop_export_snapshots WHERE id=?", (snapshot_id,)).fetchone()[0])
        payload["rows"][0].pop(missing)
        db.execute("UPDATE desktop_export_snapshots SET payload=? WHERE id=?", (json.dumps(payload), snapshot_id))
    with pytest.raises(ValueError, match="市场或行内日期"):
        service.bind_receipt(intent_id=intents[0], snapshot_id=snapshot_id, broker_order_id="000023", note="观察")


def test_link_cannot_be_silently_replaced_and_stale_source_stays_stale(tmp_path):
    service, intents, snapshot_id, source = setup_service(tmp_path)
    service.bind_receipt(intent_id=intents[0], snapshot_id=snapshot_id, broker_order_id="000023", note="观察")
    with sqlite3.connect(source) as db:
        db.execute("UPDATE orders SET status='FILLED'")
    result = service.snapshot()
    assert all(i["status"] == "SOURCE_STALE" for i in result["intents"])
    assert not result["receipt_links"]["links"][0]["reconciliation"]["allow_resubmit"]


def test_history_limit_is_a_review_state_not_empty_fill(tmp_path):
    service, intents, snapshot_id, _ = setup_service(tmp_path)
    service.bind_receipt(intent_id=intents[0], snapshot_id=snapshot_id, broker_order_id="000023", note="观察")
    with service._connect() as db:
        row = list(db.execute("SELECT * FROM desktop_export_snapshots WHERE id=?", (snapshot_id,)).fetchone())
        for i in range(100):
            duplicate = row.copy()
            duplicate[0] = f"test-extra-{i}"
            db.execute("INSERT INTO desktop_export_snapshots VALUES (?,?,?,?,?,?,?,?,?)", duplicate)
    report = service.receipts.snapshot()["links"][0]["reconciliation"]
    assert report["status"] == "REVIEW_REQUIRED" and report["observed_filled_quantity"] is None
    assert report["issues"] == ["HISTORY_LIMIT_EXCEEDED"]


def test_nonempty_exports_reconcile_through_persistent_store_without_paper_writes(tmp_path):
    service, intents, snapshot_id, source = setup_service(tmp_path)
    before = source.read_bytes()
    service.bind_receipt(intent_id=intents[0], snapshot_id=snapshot_id, broker_order_id="000023", note="观察")
    folder = tmp_path / "state/private/huatai-acceptance"
    updated = (folder / "orders-test.xls").read_bytes().decode("gbk").replace("\t100\t0\t", "\t100\t100\t")
    (folder / "orders-filled.xls").write_bytes(updated.encode("gbk"))
    service.import_exports(filename="orders-filled.xls", kind="orders", account_ref="test", trade_date="20261022")
    trades = ("成交日期\t成交时间\t证券代码\t证券名称\t操作\t成交数量\t成交均价\t市场类别\t委托编号\t成交编号\n"
              "20261022\t09:32:00\t600036\t虚构证券\t买入\t100\t10.49\t上海Ａ\t000023\t000001\n")
    (folder / "trades-test.xls").write_bytes(trades.encode("gbk"))
    service.import_exports(filename="trades-test.xls", kind="trades", account_ref="test", trade_date="20261022")
    result = DesktopExecutionService(tmp_path).snapshot()
    report = result["receipt_links"]["links"][0]["reconciliation"]
    assert report["status"] == "OBSERVED_FULL" and report["observed_filled_quantity"] == 100
    assert report["confirmed_filled_quantity"] is None and not report["allow_resubmit"]
    assert source.read_bytes() == before
