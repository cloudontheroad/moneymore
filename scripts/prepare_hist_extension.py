"""Historical inputs for frozen HIST extension; bounded isolated Tushare cache."""
import argparse
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from prepare_ashare_three_year import Fetcher
from prepare_hist_reproduction import ROOT, digest, write_json

CONFIG = json.loads((ROOT / "configs/hist_extension_20261114.json").read_text(encoding="utf-8"))
OUT = ROOT / CONFIG["output"]
SOURCE = ROOT / CONFIG["source_run"]


class ExtensionFetcher(Fetcher):
    def __init__(self):
        from moneymore.data.tushare_provider import TushareProvider
        self.config = CONFIG
        self.output = OUT / "data"
        self.output.mkdir(parents=True, exist_ok=True)
        load_dotenv(ROOT / ".env", override=False)
        self.client = TushareProvider()._client
        self.client._DataApi__timeout = 25
        self.calls = self.consecutive_failures = 0
        self.journal = self.output / "requests.jsonl"
        self.previous_calls = len(self.journal.read_text().splitlines()) if self.journal.exists() else 0


def membership_from_snapshots(weights, days):
    snapshots = {str(d): sorted(g.con_code.unique()) for d, g in weights.groupby("trade_date")}
    if not snapshots or any(len(v) != 300 for v in snapshots.values()):
        raise ValueError("Historical CSI300 snapshot does not have 300 unique constituents")
    available = sorted(snapshots)
    rows = []
    for day in days:
        pos = np.searchsorted(available, day, side="right")-1
        if pos < 0:
            raise ValueError("No past membership snapshot: " + day)
        dated = available[pos]
        if (pd.Timestamp(day)-pd.Timestamp(dated)).days > 65:
            raise ValueError("Stale membership snapshot: " + day)
        rows.extend({"trade_date": day, "ts_code": code, "snapshot_date": dated} for code in snapshots[dated])
    return pd.DataFrame(rows)


def fetch_inputs(probe=False):
    f = ExtensionFetcher()
    data = f.output
    calendar = f.get("trade_cal", "20200901-20261113", nonempty=True, exchange="SSE",
                     start_date=CONFIG["raw_start"], end_date=CONFIG["requested_end"])
    days = sorted(calendar.loc[calendar.is_open.eq(1), "cal_date"].astype(str))
    assert days[-1] == CONFIG["requested_end"], "Requested cutoff is not a session"
    # Establish actual latest daily data before claiming extension through today.
    latest = f.get("daily", days[-1], nonempty=True, trade_date=days[-1])
    if len(latest) < 4000 or not latest.trade_date.eq(days[-1]).all():
        raise ValueError("LATEST_OFFICIAL_CLOSE_INCOMPLETE")
    months = pd.period_range("2020-12", pd.Timestamp(days[-1]), freq="M")
    pieces = []
    for i, month in enumerate(months):
        lo, hi = month.start_time.strftime("%Y%m%d"), min(days[-1], month.end_time.strftime("%Y%m%d"))
        frame = f.get("index_weight", str(month), nonempty=False, index_code=CONFIG["index_code"],
                      start_date=lo, end_date=hi)
        if not frame.empty:
            if len(frame) >= 4000:
                raise ValueError("INDEX_WEIGHT_POSSIBLY_TRUNCATED")
            pieces.append(frame)
        f.progress("MEMBERSHIP", completed=i+1, total=len(months), month=str(month))
        if probe:
            print("PROBE_OK", "latest", days[-1], len(latest), "membership", len(frame), flush=True)
            return
    weights = pd.concat(pieces, ignore_index=True).drop_duplicates(["trade_date", "con_code"])
    weights.trade_date = weights.trade_date.astype(str)
    weights.to_parquet(data / "index_weights.parquet", index=False)
    evaluation = [d for d in days if d >= CONFIG["extension_start"]]
    members = membership_from_snapshots(weights, evaluation)
    members.to_parquet(data / "membership.parquet", index=False)
    # Retain old symbols too: positions held at the splice must be able to sell.
    old_index = pd.read_parquet(SOURCE / "test/index.parquet")
    prior = {code[2:] + "." + code[:2] for code in old_index.instrument.unique()}
    symbols = sorted(set(members.ts_code) | prior)
    filters = [("trade_date", ">=", CONFIG["raw_start"]), ("trade_date", "<=", days[-1]), ("ts_code", "in", symbols)]
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]
    bars = pd.read_parquet(ROOT / "data/processed/daily.parquet", columns=columns, filters=filters)
    adj = pd.read_parquet(ROOT / "data/processed/adj_factor.parquet", filters=filters)
    bar_days, adj_days = set(bars.trade_date), set(adj.trade_date)
    extra_b, extra_a = [], []
    for day in days:
        if day not in bar_days or day == days[-1]:
            b = f.get("daily", day, nonempty=True, trade_date=day)
            extra_b.append(b.loc[b.ts_code.isin(symbols), columns])
        if day not in adj_days or day == days[-1]:
            a = f.get("adj_factor", day, nonempty=True, trade_date=day)
            extra_a.append(a.loc[a.ts_code.isin(symbols), ["ts_code", "trade_date", "adj_factor"]])
    bars = pd.concat([bars, *extra_b], ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
    adj = pd.concat([adj, *extra_a], ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
    bars = bars.merge(adj, on=["ts_code", "trade_date"], validate="one_to_one", how="left")
    missing = bars.loc[bars.close.gt(0) & bars.adj_factor.isna(), "ts_code"].unique()
    for code in missing:
        patch = f.get("adj_factor", code+"-full", nonempty=True, ts_code=code,
                      start_date=CONFIG["raw_start"], end_date=days[-1])
        lookup = patch.set_index("trade_date").adj_factor
        mask = bars.ts_code.eq(code) & bars.adj_factor.isna()
        bars.loc[mask, "adj_factor"] = bars.loc[mask, "trade_date"].map(lookup)
    assert bars.loc[bars.close.gt(0), "adj_factor"].gt(0).all()
    bars.sort_values(["ts_code", "trade_date"]).to_parquet(data / "bars.parquet", index=False)
    caps = []
    cache = ROOT / "state/ashare-three-year-20261113/data-v1/official"
    for i, day in enumerate(evaluation):
        path = cache / f"daily_basic-{day}.parquet"
        if path.exists():
            b = pd.read_parquet(path, columns=["ts_code", "trade_date", "total_mv"])
        else:
            b = f.get("daily_basic", day, nonempty=True, trade_date=day,
                      fields="ts_code,trade_date,total_mv")
        if b.ts_code.duplicated().any() or not b.trade_date.astype(str).eq(day).all():
            raise ValueError("Invalid market value cross section " + day)
        caps.append(b.loc[b.ts_code.isin(symbols), ["ts_code", "trade_date", "total_mv"]])
        if i % 20 == 0:
            f.progress("MARKET_VALUE", completed=i+1, total=len(evaluation), day=day)
    market = pd.concat(caps, ignore_index=True)
    market.to_parquet(data / "market_value.parquet", index=False)
    names = {"000300.SH": "CSI300_price", "H00300.CSI": "CSI300_total_return",
             "399006.SZ": "ChiNext_price", "399606.SZ": "ChiNext_total_return"}
    (OUT / "benchmarks").mkdir(exist_ok=True)
    for code, name in names.items():
        b = f.get("index_daily", code+"-2016-current", nonempty=True, ts_code=code,
                  start_date="20161230", end_date=days[-1])
        if b.trade_date.min() != "20161230" or b.trade_date.max() != days[-1] or b.trade_date.duplicated().any():
            raise ValueError("Benchmark window incomplete " + code)
        b.sort_values("trade_date").to_parquet(OUT / "benchmarks" / f"{name}.parquet", index=False)
    write_json(data / "calendar.json", {"days": days, "evaluation": evaluation})
    write_json(data / "complete.json", {"end": days[-1], "evaluation_days": len(evaluation),
        "symbols": len(symbols), "membership_snapshots": weights.trade_date.nunique(),
        "hashes": {name: digest(data/name) for name in
                   ("bars.parquet", "market_value.parquet", "membership.parquet", "calendar.json")}})
    f.progress("INPUTS_COMPLETE", end=days[-1], symbols=len(symbols))


def audit_missing_quotes():
    f = ExtensionFetcher()
    members = pd.read_parquet(f.output/"membership.parquet")
    bars = pd.read_parquet(f.output/"bars.parquet")
    coverage = members.merge(bars[["ts_code", "trade_date", "close", "vol"]], how="left", on=["ts_code", "trade_date"])
    missing = coverage.loc[~(coverage.close.gt(0) & coverage.vol.gt(0))].copy()
    proof = set()
    for i, (symbol, rows) in enumerate(missing.groupby("ts_code")):
        halt = f.get("suspend_d", symbol+"-2021-current", ts_code=symbol, start_date=CONFIG["extension_start"],
                     end_date=CONFIG["requested_end"], suspend_type="S")
        if not halt.empty:
            full = halt.loc[halt.suspend_type.eq("S") & (halt.suspend_timing.isna() | halt.suspend_timing.eq(""))]
            proof.update(zip(full.ts_code, full.trade_date))
        if i % 10 == 0:
            f.progress("HALT_AUDIT", checked_symbols=i+1, total_symbols=missing.ts_code.nunique())
    missing["official_full_day_halt"] = [(s, d) in proof for s, d in zip(missing.ts_code, missing.trade_date)]
    retired = f.get("stock_basic", "delisted-20261113", list_status="D",
                    fields="ts_code,name,list_date,delist_date,list_status")
    delist = retired.set_index("ts_code").delist_date.to_dict()
    missing["official_delisted"] = [bool(delist.get(s)) and d >= str(delist[s])
                                     for s, d in zip(missing.ts_code, missing.trade_date)]
    # Distinguish local-cache gaps from actual exchange suspensions/delistings.
    unknown = missing.loc[~(missing.official_full_day_halt | missing.official_delisted)]
    recovered = []
    for symbol, rows in unknown.groupby("ts_code"):
        actual = f.get("daily", symbol+"-missing-quotes", ts_code=symbol,
                       start_date=rows.trade_date.min(), end_date=rows.trade_date.max())
        actual = actual.loc[actual.trade_date.isin(rows.trade_date)]
        if actual.empty:
            continue
        factors = f.get("adj_factor", symbol+"-missing-quotes", ts_code=symbol,
                        start_date=rows.trade_date.min(), end_date=rows.trade_date.max())
        actual = actual.merge(factors[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], validate="one_to_one")
        assert actual.adj_factor.gt(0).all()
        recovered.append(actual.loc[:, bars.columns])
    if recovered:
        patches = pd.concat(recovered, ignore_index=True)
        bars = pd.concat([bars, patches], ignore_index=True).drop_duplicates(["ts_code", "trade_date"], keep="last")
        bars.sort_values(["ts_code", "trade_date"]).to_parquet(f.output/"bars.parquet", index=False)
        record = json.loads((f.output/"complete.json").read_text())
        record["hashes"]["bars.parquet"] = digest(f.output/"bars.parquet")
        record["recovered_official_quotes"] = len(patches)
        write_json(f.output/"complete.json", record)
        recovered_keys = set(zip(patches.ts_code, patches.trade_date))
    else:
        recovered_keys = set()
    missing["recovered_official_quote"] = [(s, d) in recovered_keys for s, d in zip(missing.ts_code, missing.trade_date)]
    missing.to_parquet(f.output/"missing-quotes-audit.parquet", index=False)
    result = {"missing_member_quotes": len(missing), "official_full_day_halts": int(missing.official_full_day_halt.sum()),
              "after_official_delisting": int(missing.official_delisted.sum()), "recovered_quotes": len(recovered_keys),
              "unexplained": int((~(missing.official_full_day_halt | missing.official_delisted | missing.recovered_official_quote)).sum())}
    write_json(f.output/"quote-audit.json", result)
    print("QUOTE_AUDIT", result, flush=True)
    if result["unexplained"]:
        raise ValueError("Unexplained missing membership quotes; do not certify results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit_missing_quotes()
    else:
        fetch_inputs(args.probe)
