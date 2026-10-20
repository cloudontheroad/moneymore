# coding: utf-8
"""QMT-editor-only SDK history test. No downloads, subscriptions or orders."""
_done = False


def init(C):
    global _done
    _done = False
    print("MONEYMORE_SDK_HISTORY START READ_ONLY")


def handlebar(C):
    global _done
    if _done:
        return
    _done = True
    try:
        from xtquant import xtdata
        for period, start, end in (
                ("1d", "20251001", "20261019"),
                ("1m", "20261019093000", "20261019150000")):
            try:
                data = xtdata.get_market_data_ex(
                    field_list=["open", "high", "low", "close", "volume"],
                    stock_list=["600036.SH"], period=period,
                    start_time=start, end_time=end, count=-1,
                    dividend_type="none", fill_data=False)
                frame = (data or {}).get("600036.SH")
                n = 0 if frame is None else len(frame)
                print("SDK_HISTORY " + period + " rows=" + str(n))
                if n:
                    print("SDK_RANGE " + period + " " + str(frame.index[0]) + " -> " + str(frame.index[-1]))
                    print("SDK_VALID " + period + " unique=" + str(frame.index.is_unique) +
                          " sorted=" + str(frame.index.is_monotonic_increasing) +
                          " nulls=" + str(int(frame.isnull().sum().sum())))
            except Exception as error:
                print("SDK_HISTORY " + period + " FAIL " + type(error).__name__ + ": " + str(error)[:160])
        tick = (xtdata.get_full_tick(["600036.SH"]) or {}).get("600036.SH") or {}
        print("SDK_TICK present=" + str(bool(tick)) + " last=" + str(tick.get("lastPrice")))
    except Exception as error:
        print("SDK_IMPORT_OR_TICK FAIL " + type(error).__name__)
    print("MONEYMORE_SDK_HISTORY END NO_ORDERS_SENT")
