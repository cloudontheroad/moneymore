"""Pure selector audit for EasyTHS 2.0.7. Never imports or runs its operations."""
from __future__ import annotations


# These are selectors from the pinned upstream, not asserted Huatai mappings.
EXPECTED = {
    "frame": ("AfxMDIFrame140s", "59648"),
    "funds_balance": ("Static", "1012"),
    "funds_market_value": ("Static", "1014"),
    "funds_total_assets": ("Static", "1015"),
    "funds_available": ("Static", "1016"),
    "funds_withdrawable": ("Static", "1017"),
    "grid_panel": (None, "1047"),
    "order_filter": ("Edit", "3348"),
    "order_query": ("Button", "3349"),
    "security_editor": ("Edit", "1032"),
    "price_editor": ("Edit", "1033"),
    "quantity_editor": ("Edit", "1034"),
}


def audit_inventory(controls: list[dict]) -> dict:
    """Flat inventory proves existence only, not hierarchy, visibility or semantics."""
    selectors = []
    for role, (class_name, automation_id) in EXPECTED.items():
        matches = sum(
            (class_name is None or item.get("class_name") == class_name)
            and str(item.get("automation_id", "")) == automation_id
            for item in controls
        )
        selectors.append({"role": role, "count": matches,
                          "result": "MISSING" if not matches else "PRESENT" if matches == 1 else "AMBIGUOUS"})
    return {
        "profile": "easyths-2.0.7-c7ea854",
        "status": "SELECTOR_INVENTORY_ONLY",
        "selectors": selectors,
        "grid_class_present": any(c.get("class_name") == "CVirtualGridCtrl" for c in controls),
        "hierarchy_verified": False,
        "account_verified": False,
        "broker_receipts_verified": False,
        "compatible": None,
        "submission_supported": False,
        "limitations": ["只检查当前页面，缺失控件可能在其他页面", "控件编号存在不等于业务含义或回执正确"],
    }
