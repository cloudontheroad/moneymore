"""Bounded monthly snapshot ingestion and isolated Qlib factor-weight validation."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_model import refresh_official_basic


def monthly_schedule(store, policy):
    dates = [d for d in trading_dates(store, policy["as_of"]) if d <= policy["as_of"]]
    frame = pd.DataFrame({"signal_date": dates})
    frame["month"] = pd.to_datetime(frame.signal_date, format="%Y%m%d").dt.to_period("M")
    selected = frame.groupby("month").tail(1)
    selected = selected.loc[selected.signal_date.between(policy["start"], policy["end"])]
    # The snapshot is available before the signal close; valuation and cap use the same lag as live.
    return [(d, dates[dates.index(d) - 1]) for d in selected.signal_date]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build-panel", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--historical-financials", action="store_true")
    parser.add_argument("--industry", action="store_true")
    parser.add_argument("--audit-samples", action="store_true")
    parser.add_argument("--execution-preflight", action="store_true")
    parser.add_argument("--classify-price-gaps", action="store_true")
    parser.add_argument("--max-requests", type=int, default=25)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    policy = yaml.safe_load(
        (root / "configs/baseline_v2_validation.yaml").read_text(encoding="utf-8")
    )
    schedule = monthly_schedule(store, policy)
    if args.classify_price_gaps:
        from moneymore.research.baseline_v2_sample_audit import refine_cached_price_audit

        print(json.dumps(refine_cached_price_audit(root, store, policy), default=str))
    if args.execution_preflight:
        from moneymore.research.baseline_v2_execution_preflight import execution_preflight

        print(json.dumps(execution_preflight(root, store), default=str))
    if args.audit_samples:
        from moneymore.research.baseline_v2_sample_audit import audit_samples

        print(json.dumps(audit_samples(root, store, policy, schedule), default=str))
    if args.industry:
        from moneymore.research.baseline_v2_industry import download_industry

        print(json.dumps(download_industry(root, store, args.max_requests)))
    if args.historical_financials:
        from dotenv import load_dotenv

        from moneymore.data.fundamental_research import sync_research_financials
        from moneymore.data.tushare_provider import TushareProvider

        load_dotenv(root / ".env", override=False)
        basic = store.read("baseline_v2_daily_basic")
        required = set()
        for _, date in schedule:
            f = basic.loc[basic.trade_date.eq(date)]
            codes = f.ts_code.astype(str)
            f = f.loc[
                (
                    (codes.str.endswith(".SH") & codes.str.startswith(("60", "68")))
                    | (codes.str.endswith(".SZ") & codes.str.startswith(("00", "30")))
                )
            ]
            required.update(f.sort_values("total_mv", ascending=False).head(1000).ts_code)
        known = set(store.read("fina_indicator_research_v2", columns=["ts_code"]).ts_code)
        pending = sorted(required - known)
        print(
            json.dumps(
                {"historical_symbols": len(required), "missing_financial_symbols": len(pending)}
            ),
            flush=True,
        )
        result = sync_research_financials(
            root,
            store,
            TushareProvider(),
            pending,
            policy["as_of"],
            max_requests=args.max_requests,
            interval_seconds=1.1,
            checkpoint_name="historical-weight-checkpoint.json",
            flush_every=50,
        )
        print(
            json.dumps({k: v for k, v in result.items() if not k.endswith("_symbols")}), flush=True
        )
    if args.download:
        results = []
        requests = 0
        for signal, date in schedule:
            cached = store.read("baseline_v2_daily_basic", filters=[("trade_date", "=", date)])
            if not cached.empty:
                continue
            if requests >= min(100, args.max_requests):
                break
            if requests:
                time.sleep(1.3)
            try:
                result = refresh_official_basic(root, store, date)
                requests += int(result["status"] == "FETCHED")
                results.append({**result, "signal_date": signal})
                print(json.dumps(results[-1]), flush=True)
            except Exception as error:  # noqa: BLE001 - stop on provider error, never expose secret payload
                # No provider message/credential in artifacts; no blind retries.
                print(
                    json.dumps(
                        {
                            "status": "STOPPED_ERROR",
                            "error_type": type(error).__name__,
                            "date": date,
                        }
                    ),
                    flush=True,
                )
                break
        write_diagnostic(
            root / "state/baseline-v2-validation/download.json",
            {"requests": requests, "results": results},
        )
    if args.build_panel:
        from moneymore.research.baseline_v2_weights import build_monthly_validation_panel

        print(
            json.dumps(build_monthly_validation_panel(root, store, policy, schedule), default=str)
        )
    if args.fit:
        from moneymore.research.baseline_v2_weights import validate_weights

        print(json.dumps(validate_weights(root, policy), default=str))
