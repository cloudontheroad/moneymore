# coding: utf-8
"""Run in QMT only. One official download request, one symbol, one hour.

No external connections, file manipulation, retry loop or trading operations.
The SDK manages its own authorized client cache; no manual cache edits.
"""
_done = False


def init(C):
    global _done
    _done = False
    print("MONEYMORE_GAP START ONE_REQUEST_ONLY")


def _check(xtdata, label):
    data = xtdata.get_market_data_ex(
        field_list=["close", "volume"], stock_list=["600036.SH"],
        period="1m", start_time="20261019135500", end_time="20261019150000",
        count=-1, dividend_type="none", fill_data=False)
    frame = (data or {}).get("600036.SH")
    n = 0 if frame is None else len(frame)
    print("GAP_" + label + " rows=" + str(n))
    if n:
        print("GAP_RANGE " + str(frame.index[0]) + " -> " + str(frame.index[-1]))


def handlebar(C):
    global _done
    if _done:
        return
    _done = True
    try:
        from xtquant import xtdata
        _check(xtdata, "BEFORE")
        print("GAP_DOWNLOAD_REQUEST 600036.SH 1m 20261019135500..20261019150000")
        xtdata.download_history_data("600036.SH", "1m", "20261019135500", "20261019150000")
        print("GAP_DOWNLOAD_RETURNED (not proof of completeness)")
        _check(xtdata, "AFTER")
    except Exception as error:
        print("GAP_FAIL " + type(error).__name__ + ": " + str(error)[:240])
    print("MONEYMORE_GAP END NO_ORDERS_SENT")
