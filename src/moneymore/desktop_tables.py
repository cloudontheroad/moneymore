"""Offline TSV/CSV validation boundary, not a broker receipt verifier.

Accepts explicitly supplied exports only. No clipboard, GUI, network or ledger
access. Column profiles require validation against a real Huatai export.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import date, time
from decimal import Decimal

PROFILES = {
    "positions": {
        "code": ("证券代码",), "name": ("证券名称",),
        "quantity": ("持仓数量", "股票余额"),
        "available_quantity": ("可用数量", "可用余额"),
    },
    "orders": {
        "code": ("证券代码",), "name": ("证券名称",),
        "order_id": ("合同编号", "委托编号"), "side": ("操作", "买卖标志"),
        "quantity": ("委托数量",), "filled_quantity": ("成交数量",),
        "price": ("委托价格",), "time": ("委托时间",),
    },
    "trades": {
        "code": ("证券代码",), "name": ("证券名称",),
        "order_id": ("合同编号", "委托编号"), "trade_id": ("成交编号",),
        "side": ("操作", "买卖标志"), "quantity": ("成交数量",),
        "price": ("成交价格", "成交均价"), "time": ("成交时间",),
    },
}


class TableValidationError(ValueError):
    """Messages deliberately omit account details and cell contents."""


def parse_export(text: str, *, kind: str, account_ref: str, trade_date: str) -> dict:
    """Return unverified rows, preserving identifiers and exact decimal strings.

    Caller-supplied account/date are context, NOT independently verified facts.
    Even a valid empty table is only an unverified empty export.
    """
    if kind not in PROFILES or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", account_ref):
        raise TableValidationError("不支持的表类型或本地账户别名")
    try:
        if not re.fullmatch(r"\d{8}", trade_date):
            raise ValueError()
        date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:8]))
    except ValueError:
        raise TableValidationError("交易日无效") from None
    if not text.strip() or len(text) > 5_000_000:
        raise TableValidationError("表格为空或超过大小上限")
    source = text.lstrip("\ufeff")
    delimiter = "\t" if "\t" in source.splitlines()[0] else ","
    try:
        reader = csv.reader(io.StringIO(source, newline=None), delimiter=delimiter, strict=True)
        headers = [v.strip() for v in next(reader)]
        if len(headers) != len(set(headers)):
            raise TableValidationError("重复列名")
        columns = {}
        for field, aliases in PROFILES[kind].items():
            found = [headers.index(a) for a in aliases if a in headers]
            if len(found) != 1:
                raise TableValidationError(f"必需字段缺失或歧义：{field}")
            columns[field] = found[0]
        rows, identities = [], set()
        for line_number, values in enumerate(reader, start=2):
            if line_number > 10002:
                raise TableValidationError("超过行数上限")
            if not values or all(not v.strip() for v in values):
                continue
            if len(values) != len(headers):
                raise TableValidationError(f"第{line_number}行列数不符")
            row = {field: values[index].strip() for field, index in columns.items()}
            market_columns = [headers.index(h) for h in ("市场类别", "交易市场") if h in headers]
            if len(market_columns) > 1:
                raise TableValidationError("市场字段歧义")
            market = values[market_columns[0]].strip().replace("Ａ", "A") if market_columns else ""
            row["market"] = {"上海A": "SH", "深圳A": "SZ"}.get(market)
            row["date_observed"] = False
            date_column = {"orders": "委托日期", "trades": "成交日期"}.get(kind)
            if date_column and date_column in headers:
                exported_date = values[headers.index(date_column)].strip().replace("-", "").replace("/", "")
                if exported_date != trade_date:
                    raise TableValidationError(f"第{line_number}行交易日不匹配")
                row["date_observed"] = True
            if not re.fullmatch(r"\d{6}", row["code"]):
                raise TableValidationError(f"第{line_number}行证券代码无效")
            if not row["name"]:
                raise TableValidationError(f"第{line_number}行名称为空")
            for field in ("quantity", "available_quantity", "filled_quantity"):
                if field not in row:
                    continue
                value = row[field].replace(",", "")
                if not re.fullmatch(r"\d+", value):
                    raise TableValidationError(f"第{line_number}行数量无效")
                row[field] = int(value)
            if any(row.get(f, 0) > row["quantity"] for f in ("available_quantity", "filled_quantity")):
                raise TableValidationError(f"第{line_number}行数量关系无效")
            if kind != "positions":
                if row["side"] not in {"买入", "卖出", "证券买入", "证券卖出"}:
                    raise TableValidationError(f"第{line_number}行方向未支持")
                row["side"] = "BUY" if row["side"].endswith("买入") else "SELL"
                if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", row["time"]):
                    raise TableValidationError(f"第{line_number}行时间无效")
                try:
                    time.fromisoformat(row["time"])
                except ValueError:
                    raise TableValidationError(f"第{line_number}行时间无效") from None
                price = row["price"].replace(",", "")
                if not re.fullmatch(r"\d+(?:\.\d+)?", price):
                    raise TableValidationError(f"第{line_number}行价格无效")
                number = Decimal(price)
                if kind == "trades" and (number <= 0 or row["quantity"] <= 0):
                    raise TableValidationError(f"第{line_number}行成交数量或价格无效")
                row["price"] = str(number)
                for field in ("order_id", "trade_id"):
                    if field in row and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row[field]):
                        raise TableValidationError(f"第{line_number}行回执编号无效")
            identity = row.get("trade_id") or row.get("order_id") or row["code"]
            if identity in identities:
                raise TableValidationError(f"第{line_number}行标识重复，须核对市场或分笔维度")
            identities.add(identity)
            # Never infer strategy association or ledger updates from stock alone.
            row.update(account_ref=account_ref, trade_date=trade_date, strategy_id=None)
            rows.append(row)
    except (csv.Error, StopIteration):
        raise TableValidationError("表格格式不完整") from None
    return {
        "parser_profile": "huatai-text-v2-market-date",
        "status": "PARSED_UNVERIFIED", "kind": kind, "rows": rows,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "account_verified": False, "date_verified": False,
        "broker_receipts_verified": False, "ledger_updated": False,
        "live_submission_supported": False,
    }


def parse_huatai_export(data: bytes, *, kind: str, account_ref: str, trade_date: str) -> dict:
    """Read observed Huatai .xls exports, actually GBK TSV rather than Excel.

    True binary workbooks and undecodable exports are rejected. No guess-based
    lossy decoding: malformed bytes must not turn into altered receipt IDs.
    """
    if not data or len(data) > 5_000_000:
        raise TableValidationError("导出为空或超过大小上限")
    if data.startswith((b"\xd0\xcf\x11\xe0", b"PK\x03\x04")) or b"\x00" in data:
        raise TableValidationError("不支持二进制工作簿，仅接受已验收的文本导出")
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "gbk"
    try:
        text = data.decode(encoding, errors="strict")
    except UnicodeError:
        raise TableValidationError("导出编码异常") from None
    report = parse_export(text, kind=kind, account_ref=account_ref, trade_date=trade_date)
    report["source_format"] = "HUATAI_TEXT_EXPORT"
    report["encoding"] = encoding
    report["source_sha256"] = hashlib.sha256(data).hexdigest()
    return report
