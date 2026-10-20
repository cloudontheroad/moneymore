# coding: utf-8
"""Run inside QMT's strategy editor, NOT from an external Python process.

Read-only: no orders, files, network clients, account identifiers or credentials.
Prints a bounded diagnostic once per strategy start. Library import success does
not establish that model-file loading or external communication is permitted.
"""

def init(C):
    C.moneymore_probe_done = False
    print("MONEYMORE_PROBE START READ_ONLY")
    import sys
    print("PYTHON " + sys.version.replace("\n", " "))
    for name in ("numpy", "pandas", "sklearn", "torch"):
        try:
            module = __import__(name)
            print("LIB " + name + " OK " + str(getattr(module, "__version__", "unknown")))
        except Exception as error:
            print("LIB " + name + " FAIL " + type(error).__name__ + ": " + str(error)[:200])
    print("MONEYMORE_PROBE waiting for first handlebar")


def handlebar(C):
    if getattr(C, "moneymore_probe_done", False):
        return
    C.moneymore_probe_done = True
    try:
        data = C.get_market_data_ex(
            ["open", "high", "low", "close", "volume"],
            ["600036.SH"], period="1d", count=5,
        )
        frame = data.get("600036.SH") if data is not None else None
        if frame is None or len(frame) == 0:
            print("DAILY EMPTY: check QMT historical data availability")
        else:
            print("DAILY OK rows=" + str(len(frame)))
            print(frame.tail(2).to_string())
    except Exception as error:
        print("DAILY FAIL " + type(error).__name__ + ": " + str(error)[:300])
    try:
        ticks = C.get_full_tick(["600036.SH"]) or {}
        tick = ticks.get("600036.SH") or {}
        print("TICK " + ("OK" if tick else "EMPTY") +
              " time=" + str(tick.get("time")) +
              " lastPrice=" + str(tick.get("lastPrice")))
    except Exception as error:
        print("TICK FAIL " + type(error).__name__ + ": " + str(error)[:300])
    print("MONEYMORE_PROBE END NO_ORDERS_SENT")
