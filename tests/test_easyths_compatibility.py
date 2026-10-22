from moneymore.easyths_compatibility import EXPECTED, audit_inventory


def test_empty_inventory_does_not_claim_incompatible_client():
    report = audit_inventory([])
    assert all(s["result"] == "MISSING" for s in report["selectors"])
    assert report["compatible"] is None
    assert not report["submission_supported"]


def test_all_selectors_is_still_not_a_business_acceptance():
    controls = [{"class_name": cls or "Pane", "automation_id": aid}
                for cls, aid in EXPECTED.values()]
    controls.append({"class_name": "CVirtualGridCtrl"})
    result = audit_inventory(controls)
    assert all(s["result"] == "PRESENT" for s in result["selectors"])
    assert result["grid_class_present"]
    assert result["compatible"] is None
    assert not result["hierarchy_verified"]
    assert not result["account_verified"]
    assert not result["broker_receipts_verified"]


def test_same_id_with_wrong_class_does_not_match():
    result = audit_inventory([{"class_name": "Button", "automation_id": "1032"}])
    assert next(s for s in result["selectors"] if s["role"] == "security_editor")["result"] == "MISSING"


def test_duplicate_selectors_require_disambiguation():
    item = {"class_name": "Edit", "automation_id": "1032"}
    result = audit_inventory([item, item])
    assert next(s for s in result["selectors"] if s["role"] == "security_editor")["result"] == "AMBIGUOUS"
