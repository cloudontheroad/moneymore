# coding: utf-8
"""Run ONLY in updated QMT editor; bounded read-only API discovery.

No trading objects, account identifiers, authentication calls, files or sockets.
"""
_done = False


def init(C):
    global _done
    _done = False
    import sys
    print("MONEYMORE_NEW_API START python=" + sys.version.split()[0])
    for name in ("xtdata", "xttrade", "xttrader"):
        print("INJECTED " + name + "=" + str(name in globals()))


def handlebar(C):
    global _done
    if _done:
        return
    _done = True
    # Discover imports only: never create a trader or attempt authentication.
    for name in ("xtdata", "xttrade", "xtquant.xtdata", "xtquant.xttrader"):
        try:
            mod = __import__(name, fromlist=["*"])
            print("IMPORT " + name + " OK")
            for attr in ("get_full_tick", "get_market_data_ex", "XtQuantTrader"):
                print("API " + name + "." + attr + "=" + str(hasattr(mod, attr)))
            if name == "xtquant.xtdata":
                try:
                    tick = (mod.get_full_tick(["600036.SH"]) or {}).get("600036.SH") or {}
                    print("BUILTIN_SDK_TICK present=" + str(bool(tick)) +
                          " last=" + str(tick.get("lastPrice")))
                except Exception as error:
                    print("BUILTIN_SDK_TICK FAIL " + type(error).__name__ + ": " + str(error)[:200])
        except Exception as error:
            print("IMPORT " + name + " FAIL " + type(error).__name__)
    # Only a module explicitly injected by the QMT host is called here. Do not
    # auto-connect the old filesystem SDK, which the external test rejected.
    module = globals().get("xtdata")
    if module is not None and callable(getattr(module, "get_full_tick", None)):
        try:
            tick = (module.get_full_tick(["600036.SH"]) or {}).get("600036.SH") or {}
            print("INJECTED_TICK present=" + str(bool(tick)) + " last=" + str(tick.get("lastPrice")))
        except Exception as error:
            print("INJECTED_TICK FAIL " + type(error).__name__)
    else:
        print("INJECTED_TICK SKIPPED no host-injected xtdata")
    print("MONEYMORE_NEW_API END NO_ORDERS_SENT")
