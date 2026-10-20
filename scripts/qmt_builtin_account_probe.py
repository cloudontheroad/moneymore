# coding: utf-8
"""QMT-internal account visibility probe: queries only, never orders.

Generate the private ready copy with prepare_qmt_probe.py --kind account.
No file/HTTP bridge, no manual login or authentication change, no retry loops.
Run in simulation mode; stop the strategy after inspecting callbacks.
"""
ACCOUNT_ID = ""
ACCOUNT_TYPE = "STOCK"
QMT_USERDATA = u"D:\\迅投极速策略交易系统交易终端 华泰证券QMT实盘\\userdata"
_trader = None
_started = False


def init(C):
    global _started
    _started = False
    print("MONEYMORE_ACCOUNT_SDK START READ_ONLY")


def _asset_result(asset):
    print("ACCOUNT_SDK_ASSET present=" + str(asset is not None))


def _position_result(rows):
    print("ACCOUNT_SDK_POSITIONS returned=" + str(rows is not None) +
          " count=" + str(len(rows) if rows is not None else 0))


def _accounts_result(rows):
    try:
        rows = rows or []
        matches = [row for row in rows if str(getattr(row, "account_id", "")) == ACCOUNT_ID]
        print("ACCOUNT_SDK_LIST count=" + str(len(rows)) + " configured_match=" + str(bool(matches)))
        # An absent configured account is a blocker, not a reason to guess IDs.
        if not matches:
            print("ACCOUNT_SDK_BLOCK configured account not visible; no further queries")
            return
        from xtquant.xttype import StockAccount
        account = StockAccount(ACCOUNT_ID, ACCOUNT_TYPE)
        _trader.query_stock_asset_async(account, _asset_result)
        _trader.query_stock_positions_async(account, _position_result)
        print("ACCOUNT_SDK_ASYNC_QUERIES_SENT NO_ORDERS")
    except Exception as error:
        print("ACCOUNT_SDK_CALLBACK_FAIL " + type(error).__name__)


def handlebar(C):
    global _started, _trader
    if _started:
        return
    _started = True
    if not ACCOUNT_ID:
        print("ACCOUNT_SDK_BLOCK missing private account setting")
        return
    try:
        from xtquant.xttrader import XtQuantTrader
        # Unique local session; never change permissions or use another path.
        import time
        session = int(time.time())
        _trader = XtQuantTrader(QMT_USERDATA, session)
        _trader.start()
        result = _trader.connect()
        print("ACCOUNT_SDK_CONNECT result=" + str(result))
        if result != 0:
            print("ACCOUNT_SDK_BLOCK no connection; no retries")
            return
        _trader.query_account_infos_async(_accounts_result)
        print("ACCOUNT_SDK_LIST_REQUESTED WAIT_FOR_CALLBACK NO_ORDERS")
    except Exception as error:
        # SDK messages may contain account identifiers: type only.
        print("ACCOUNT_SDK_FAIL " + type(error).__name__)


def stop(C):
    global _trader
    if _trader is not None:
        try:
            _trader.stop()
        except Exception:
            pass
        _trader = None
    print("MONEYMORE_ACCOUNT_SDK STOPPED NO_ORDERS_SENT")
