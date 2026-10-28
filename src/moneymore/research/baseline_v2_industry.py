"""Provider effective-date industry membership; no backwards current-label filling."""

import pandas as pd


def industry_as_of(frame, as_of):
    date = pd.Timestamp(as_of)
    start = pd.to_datetime(frame.in_date, format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(frame.out_date, format="%Y%m%d", errors="coerce")
    empty_end = frame.out_date.isna() | frame.out_date.astype(str).str.strip().isin(
        ["", "None", "nan"]
    )
    # Conservatively exclude an exit boundary rather than assume it remains valid.
    valid = start.notna() & start.le(date) & (empty_end | end.gt(date))
    selected = frame.loc[valid].copy()
    conflicts = selected.groupby("ts_code").l1_code.nunique()
    selected = selected.loc[~selected.ts_code.isin(conflicts.loc[conflicts.gt(1)].index)]
    return (
        selected.sort_values("in_date").drop_duplicates("ts_code", keep="last").set_index("ts_code")
    )


def download_industry(root, store, max_requests=20):
    import hashlib
    import time

    from dotenv import load_dotenv

    from ..challenger_diagnostics import write_diagnostic
    from ..data.tushare_provider import TushareProvider

    load_dotenv(root / ".env", override=False)
    provider = TushareProvider()
    requests = 0
    signatures = set()
    complete = True
    for mode in ("N", "Y"):
        for offset in range(0, 40000, 2000):
            key = mode + "_" + str(offset).zfill(4)
            if store.has_snapshot("baseline_v2_industry_membership", "tushare", key):
                frame = store.read_snapshot("baseline_v2_industry_membership", "tushare", key)
            else:
                if requests >= max_requests:
                    complete = False
                    break
                time.sleep(1.5)
                frame = provider._client.index_member_all(is_new=mode, limit=2000, offset=offset)
                requests += 1
                if not {
                    "ts_code",
                    "l1_code",
                    "l1_name",
                    "in_date",
                    "out_date",
                    "is_new",
                    "l3_code",
                }.issubset(frame):
                    raise ValueError("INVALID_INDUSTRY_SCHEMA")
                digest = hashlib.sha256(frame.to_json().encode()).hexdigest()
                if digest in signatures and len(frame) > 0:
                    raise ValueError("INDUSTRY_PAGINATION_NOT_ADVANCING")
                store.save_snapshot(
                    "baseline_v2_industry_membership",
                    frame,
                    "tushare",
                    ["ts_code", "l3_code", "in_date", "is_new"],
                    key,
                )
            signatures.add(hashlib.sha256(frame.to_json().encode()).hexdigest())
            print({"industry_mode": mode, "offset": offset, "rows": len(frame)}, flush=True)
            if len(frame) < 2000:
                break
        else:
            complete = False
    result = {
        "complete": complete,
        "requests": requests,
        "source": "TUSHARE_INDEX_MEMBER_ALL_N_AND_Y_EFFECTIVE_INTERVALS",
    }
    write_diagnostic(root / "state/baseline-v2-validation/industry-download.json", result)
    return result
