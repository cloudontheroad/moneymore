# coding: utf-8
"""Paste into QMT's built-in editor. Read-only; no orders or external I/O.

ACCOUNT_ID is optional: enter your own account ONLY in the QMT editor, not chat.
No account identifiers, balances or position symbols are printed.
History calls read existing client data only (subscribe=False).
"""
ACCOUNT_ID = ""
ACCOUNT_TYPE = "STOCK"  # Ordinary securities account; use CREDIT only for a credit account.
_probe_done = False


def init(C):
    global _probe_done
    _probe_done = False
    print("MONEYMORE_STAGE2 START READ_ONLY")
    if ACCOUNT_ID:
        try:
            C.set_account(ACCOUNT_ID)
            print("ACCOUNT_BIND REQUESTED (not proof of authorization)")
        except Exception as error:
            print("ACCOUNT_BIND FAIL " + type(error).__name__)


def _history(C, period, start, end, expected_min):
    try:
        result = C.get_market_data_ex(
            ["open", "high", "low", "close", "volume"],
            ["600036.SH"], period=period, start_time=start,
            end_time=end, count=-1, dividend_type="none",
            fill_data=False, subscribe=False,
        )
        frame = result.get("600036.SH")
        n = 0 if frame is None else len(frame)
        print("HISTORY " + period + " rows=" + str(n) +
              " status=" + ("ENOUGH_FOR_PROBE" if n >= expected_min else "INCOMPLETE"))
        if n:
            print("RANGE " + period + " " + str(frame.index[0]) + " -> " + str(frame.index[-1]))
            print("VALID " + period + " unique=" + str(frame.index.is_unique) +
                  " sorted=" + str(frame.index.is_monotonic_increasing) +
                  " null_cells=" + str(int(frame.isnull().sum().sum())))
    except Exception as error:
        print("HISTORY " + period + " FAIL " + type(error).__name__ + ": " + str(error)[:200])


def _account_probe():
    if not ACCOUNT_ID:
        print("ACCOUNT SKIPPED: ACCOUNT_ID is blank; configure locally in QMT if desired")
        return
    query = globals().get("get_trade_detail_data")
    if not callable(query):
        print("ACCOUNT FAIL: built-in get_trade_detail_data unavailable")
        return
    for kind in ("ACCOUNT", "POSITION", "ORDER", "DEAL"):
        try:
            rows = query(ACCOUNT_ID, ACCOUNT_TYPE, kind)
            print("ACCOUNT " + kind + " RETURNED rows=" + str(0 if rows is None else len(rows)))
        except Exception as error:
            # Error messages may contain account identifiers: emit only type.
            print("ACCOUNT " + kind + " FAIL " + type(error).__name__)
    print("ACCOUNT NOTE: empty results do not establish authorization; no balances printed")


def _gru_probe():
    try:
        import torch
        # Synthetic in-memory CPU inference only. Does NOT load our saved model.
        torch.manual_seed(7)
        net = torch.nn.GRU(input_size=6, hidden_size=8, num_layers=2,
                           batch_first=True)
        net.eval()
        with torch.no_grad():
            output, hidden = net(torch.zeros(2, 60, 6))
        print("GRU_SYNTHETIC OK output=" + str(tuple(output.size())) +
              " hidden=" + str(tuple(hidden.size())))
        print("GRU_NOTE: saved-model compatibility, feature parity and loading permission NOT VERIFIED")
    except Exception as error:
        print("GRU_SYNTHETIC FAIL " + type(error).__name__ + ": " + str(error)[:200])


def handlebar(C):
    global _probe_done
    if _probe_done:
        return
    _probe_done = True
    # Fixed closed historical windows make the diagnostics reproducible.
    _history(C, "1d", "20251001000000", "20261019235959", 200)
    _history(C, "1m", "20261019000000", "20261019235959", 200)
    # Control query without date boundaries distinguishes range parsing from
    # an empty local cache. Reads existing data; does not subscribe/download.
    for period in ("1d", "1m"):
        try:
            result = C.get_market_data_ex(
                ["close"], ["600036.SH"], period=period, count=5,
                dividend_type="none", fill_data=False, subscribe=False)
            frame = result.get("600036.SH")
            n = 0 if frame is None else len(frame)
            print("CACHE_CONTROL " + period + " rows=" + str(n))
            if n:
                print("CACHE_RANGE " + period + " " + str(frame.index[0]) + " -> " + str(frame.index[-1]))
        except Exception as error:
            print("CACHE_CONTROL " + period + " FAIL " + type(error).__name__)
    _account_probe()
    _gru_probe()
    print("MONEYMORE_STAGE2 END NO_ORDERS_SENT")
