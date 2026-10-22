from moneymore.huatai_readonly import normalize_funds


def control(aid, value, **extra):
    return {"type": "Text", "class_name": "Static", "automation_id": aid, "name": value, **extra}


def test_missing_balance_does_not_discard_other_fields_or_invent_zero():
    report = normalize_funds([control("1016", "1,234.56"), control("1356", "999")])
    assert report["fields"]["available_amount"]["value"] == "1234.56"
    assert report["fields"]["balance"]["value"] is None
    assert not report["trading_ready"]
    assert report["tables"]["trades"]["rows"] is None


def test_duplicate_and_wrong_type_fail_closed():
    report = normalize_funds([control("1016", "1"), control("1016", "2"),
                              control("1015", "3", type="Edit")])
    assert report["fields"]["available_amount"]["status"] == "AMBIGUOUS"
    assert report["fields"]["total_assets"]["value"] is None


def test_invalid_masked_values_and_nonfinite_numbers_are_not_money():
    for value in ("***", "NaN", "Infinity", "1元", "", "1e10"):
        assert normalize_funds([control("1016", value)])["fields"]["available_amount"]["value"] is None


def test_negative_profit_percent_and_clearing_notice():
    report = normalize_funds([control("1026", "-12.34"), control("1029", "-1.25%"),
                              {"name": "系统正在清算"}])
    assert report["fields"]["daily_profit"]["value"] == "-12.34"
    assert report["fields"]["daily_profit_ratio"]["value"] == "-1.25"
    assert report["clearing_notice"]
    assert not report["semantics_verified"]
