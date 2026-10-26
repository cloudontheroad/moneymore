"""Conservative Huatai UI inventory normalizer; no desktop or trading imports.

Field IDs are a provisional profile, not proof of business semantics. Values
remain local to the caller and must not be included in diagnostic logs.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

FIELDS = {
    "balance": "1012", "frozen_amount": "1013", "market_value": "1014",
    "total_assets": "1015", "available_amount": "1016",
    "withdrawable_amount": "1017", "holding_profit": "1027",
    "daily_profit": "1026", "daily_profit_ratio": "1029",
}


def normalize_funds(controls: list[dict]) -> dict:
    """Consume raw local Text inventory, never masked diagnostic text.

    Duplicate selectors fail closed. Percentage values keep percentage units
    (1.25 means 1.25 percent). Monetary values are decimal strings in CNY.
    Missing data is not zero; no accounting identity is used to invent balance.
    """
    fields = {}
    for name, aid in FIELDS.items():
        matches = [c for c in controls if c.get("type") == "Text"
                   and c.get("class_name") == "Static"
                   and str(c.get("automation_id", "")) == aid]
        value = None
        status = "MISSING" if not matches else "AMBIGUOUS" if len(matches) > 1 else "INVALID"
        if len(matches) == 1:
            raw = str(matches[0].get("name", "")).strip().replace(",", "")
            pattern = r"[+-]?\d+(?:\.\d+)?%?" if name == "daily_profit_ratio" else r"[+-]?\d+(?:\.\d+)?"
            if re.fullmatch(pattern, raw):
                try:
                    number = Decimal(raw.rstrip("%"))
                    if number.is_finite():
                        value, status = str(number), "OBSERVED_UNVERIFIED"
                except InvalidOperation:
                    pass
        fields[name] = {"value": value, "status": status, "automation_id": aid,
                        "unit": "percent" if name == "daily_profit_ratio" else "CNY"}
    clearing = any("正在清算" in str(c.get("name", "")) for c in controls)
    return {
        "profile": "huatai-static-provisional-v1", "fields": fields,
        "status": "PARTIAL_UNVERIFIED" if any(f["value"] is not None for f in fields.values()) else "UNREADABLE",
        "clearing_notice": clearing, "account_verified": False,
        "semantics_verified": False, "trading_ready": False,
        "tables": {kind: {"status": "UNREADABLE", "rows": None}
                   for kind in ("positions", "orders", "trades")},
    }
