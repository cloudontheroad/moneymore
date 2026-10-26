import pytest

from moneymore.desktop_tables import TableValidationError, parse_export, parse_huatai_export


def parse(text, kind="orders", **kw):
    return parse_export(text, kind=kind, account_ref="local_test", trade_date=kw.get("date", "20261022"))


ORDER = "证券代码\t证券名称\t合同编号\t操作\t委托数量\t成交数量\t委托价格\t委托时间\n000001\t测试\t000023\t买入\t100\t0\t10.50\t09:31:00\n"
TRADE = "证券代码\t证券名称\t合同编号\t成交编号\t操作\t成交数量\t成交价格\t成交时间\n000001\t测试\t000023\t000001\t买入\t100\t10.49\t09:31:01\n"


def test_ids_preserved_and_no_inferred_fills_or_strategy():
    report = parse(ORDER)
    row = report["rows"][0]
    assert row["code"] == "000001" and row["order_id"] == "000023"
    assert row["filled_quantity"] == 0 and row["strategy_id"] is None
    assert not report["ledger_updated"] and not report["broker_receipts_verified"]


def test_trade_is_not_an_order_and_keeps_decimal_precision():
    report = parse(TRADE, "trades")
    assert report["rows"][0]["price"] == "10.49"
    assert report["rows"][0]["trade_id"] == "000001"
    with pytest.raises(TableValidationError):
        parse(ORDER, "trades")


def test_positions_zero_available_is_not_treated_as_missing():
    result = parse("证券代码,证券名称,股票余额,可用余额\n000001,测试,100,0\n", "positions")
    assert result["rows"][0]["available_quantity"] == 0


@pytest.mark.parametrize("text", ["", ORDER + ORDER.splitlines()[1],
    ORDER.replace("\t0\t", "\t101\t"), ORDER.replace("09:31:00", "25:31:00"),
    ORDER.replace("000001", "1", 1), ORDER.replace("10.50", "NaN"),
    ORDER.replace("买入", "融资买入"), ORDER.replace("\t100\t", "\t-1\t"),
    ORDER.replace("证券名称", "证券代码"), ORDER.replace("证券名称", "未知列"),
    ORDER.replace("测试\t", "测试\t多余\t")])
def test_malformed_exports_fail_closed_without_cell_contents(text):
    with pytest.raises(TableValidationError) as error:
        parse(text)
    assert "测试" not in str(error.value)


def test_date_required_and_header_only_is_not_verified_empty_account():
    with pytest.raises(TableValidationError):
        parse(ORDER, date="20260230")
    result = parse(ORDER.splitlines()[0])
    assert result["rows"] == []
    assert result["status"] == "PARSED_UNVERIFIED" and not result["account_verified"]


def test_alias_collision_is_not_silently_accepted():
    with pytest.raises(TableValidationError):
        parse("证券代码\t证券名称\t股票余额\t持仓数量\t可用余额\n", "positions")


def test_duplicate_trade_ids_rejected_but_split_fills_allowed():
    with pytest.raises(TableValidationError):
        parse(TRADE + TRADE.splitlines()[1], "trades")
    second = TRADE.splitlines()[1].replace("\t000001\t买入", "\t000002\t买入")
    assert len(parse(TRADE + second, "trades")["rows"]) == 2


def test_observed_huatai_gbk_tsv_xls_format_with_trailing_column_and_crcrlf():
    # Real column layout, fabricated row. No user holdings or shareholder data.
    text = ("证券代码\t证券名称\t股票余额\t可用余额\t卖出冻结数量\t成本价\t"
            "保本价\t市价\t盈亏比例(%)\t盈亏(约)\t市值(约)\t市场类别\t"
            "股东代码\t今日盈亏(约)\t今日盈亏比例(%)\t数量单位\t\r\r\n"
            "000001\t测试\t100\t0\t0\t10\t10\t10\t0\t0\t1000\t深圳\t"
            "FAKE\t0\t0\t股\t\r\r\n")
    result = parse_huatai_export(text.encode("gbk"), kind="positions",
                                 account_ref="test", trade_date="20261022")
    assert len(result["rows"]) == 1
    assert result["rows"][0]["code"] == "000001"
    assert "FAKE" not in str(result)
    assert result["encoding"] == "gbk"
    assert not result["account_verified"]


@pytest.mark.parametrize("data", [b"", b"\xd0\xcf\x11\xe0stuff", b"PK\x03\x04stuff", b"\x00", b"\x81"])
def test_binary_or_bad_encoding_is_not_silently_decoded(data):
    with pytest.raises(TableValidationError):
        parse_huatai_export(data, kind="positions", account_ref="test", trade_date="20261022")


def test_exported_trade_date_must_match_requested_date():
    text = "成交日期\t" + TRADE.replace("\n000001", "\n20261021\t000001")
    with pytest.raises(TableValidationError, match="交易日不匹配"):
        parse(text, "trades")


def test_non_trade_tax_event_never_becomes_buy_or_sell():
    text = TRADE.replace("买入", "股息红利税补缴").replace("\t100\t", "\t0\t")
    with pytest.raises(TableValidationError, match="方向未支持"):
        parse(text, "trades")


def test_market_is_explicit_not_inferred_from_stock_code_and_date_flag_is_preserved():
    assert parse(ORDER)["rows"][0]["market"] is None
    text = "委托日期\t" + ORDER.replace("委托时间", "委托时间\t市场类别").replace(
        "\n000001", "\n20261022\t000001").replace("09:31:00", "09:31:00\t深圳Ａ")
    row = parse(text)["rows"][0]
    assert row["market"] == "SZ" and row["date_observed"] is True
