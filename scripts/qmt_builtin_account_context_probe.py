# coding: utf-8
"""Query the QMT-hosted account cache at the latest bar only. No orders.

No XtQuantTrader instance, external session, file bridge or network connection.
Empty account results remain a failed verification, not successful integration.
"""
ACCOUNT_ID = ""
ACCOUNT_TYPE = "STOCK"
_done = False


def init(C):
    global _done
    _done = False
    print("MONEYMORE_CONTEXT_ACCOUNT START READ_ONLY")
    if ACCOUNT_ID:
        try:
            C.set_account(ACCOUNT_ID)
            print("CONTEXT_ACCOUNT_BIND REQUESTED")
        except Exception as error:
            print("CONTEXT_ACCOUNT_BIND FAIL " + type(error).__name__)


def handlebar(C):
    global _done
    if _done:
        return
    try:
        if not C.is_last_bar():
            return
    except Exception as error:
        _done = True
        print("CONTEXT_ACCOUNT_BLOCK cannot verify latest bar: " + type(error).__name__)
        return
    _done = True
    print("CONTEXT_ACCOUNT_AT_LATEST_BAR bar=" + str(C.barpos))
    if not ACCOUNT_ID:
        print("CONTEXT_ACCOUNT_BLOCK missing account")
        return
    for kind in ("ACCOUNT", "POSITION", "ORDER", "DEAL"):
        try:
            rows = get_trade_detail_data(ACCOUNT_ID, ACCOUNT_TYPE, kind)
            print("CONTEXT_ACCOUNT " + kind + " returned=" + str(rows is not None) +
                  " rows=" + str(len(rows) if rows is not None else 0))
        except Exception as error:
            print("CONTEXT_ACCOUNT " + kind + " FAIL " + type(error).__name__)
    print("MONEYMORE_CONTEXT_ACCOUNT END NO_ORDERS_SENT")
