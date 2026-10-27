"""Prepare bounded, isolated official execution evidence before style replay."""

import argparse
import hashlib
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv

from moneymore.challenger_diagnostics import prediction_reports, trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore


def execution_coverage(daily, limits, calendar, symbols, start, end):
    expected = pd.MultiIndex.from_product(
        [[d for d in calendar if start <= d <= end], symbols], names=["trade_date", "ts_code"])
    bars = daily.drop_duplicates(["trade_date", "ts_code"]).set_index(["trade_date", "ts_code"])
    controls = limits.drop_duplicates(["trade_date", "ts_code"]).set_index(["trade_date", "ts_code"])
    active = bars.reindex(expected)
    quoted = active.open.gt(0) & active.close.gt(0)
    limit = controls.reindex(expected)
    valid = limit.up_limit.gt(0) & limit.down_limit.gt(0) & limit.up_limit.ge(limit.down_limit)
    missing = expected[quoted & ~valid]
    return {"market_rows": int(quoted.sum()), "missing_limit_rows": len(missing),
            "missing_limit_examples": [dict(zip(expected.names, key)) for key in list(missing)[:30]],
            "missing_market_rows": int((~quoted).sum()),
            "missing_market_keys": [dict(zip(expected.names, key)) for key in expected[~quoted]],
            "status": "LIMITS_COMPLETE_REQUIRES_SUSPENSION_AND_CA_AUDIT" if len(missing) == 0 else "BLOCKED_MISSING_OFFICIAL_LIMITS"}


def run(root, start, end, fetch=False, reuse=None):
    store = ParquetStore(root / "data")
    calendar = trading_dates(store, end)
    sessions = [d for d in calendar if start <= d <= end]
    if not sessions or len(sessions) > 20:
        raise ValueError("RESEARCH_WINDOW_MUST_BE_1_TO_20_SESSIONS")
    sources = {"main_gru": root / "state/qlib-challenger-shadow"}
    sources.update({"candidate_" + p.name: p for p in (root / "state/qlib-candidate-shadow").iterdir() if p.is_dir()})
    saved = {name: [r for r in prediction_reports(path, end) if start <= r["trade_date"] <= end]
             for name, path in sources.items()}
    symbols = sorted({str(s["instrument"]) for reports in saved.values() for r in reports for s in r["scores"]}
                     | {str(p["symbol"]) for reports in saved.values() for r in reports
                        for p in r.get("portfolio", {}).get("positions", []) if p["quantity"] > 0})
    daily = store.read("daily", filters=[("trade_date", ">=", start), ("trade_date", "<=", end)])
    production_limits = store.read("stock_limits")
    output = dict(as_of_date=end, start_date=start, orders_enabled=False, automatic_promotion=False,
                  production_before=execution_coverage(daily, production_limits, calendar, symbols, start, end),
                  accounts={name: {"prediction_dates": [r["trade_date"] for r in rows],
                                   "missing_prediction_dates": sorted(set(sessions) - {r["trade_date"] for r in rows})}
                            for name, rows in saved.items()})
    if fetch or reuse:
        from moneymore.data.tushare_provider import TushareProvider
        if not reuse:
            load_dotenv(root / ".env")
        client = TushareProvider()._client if not reuse else None
        evidence = Path(reuse) if reuse else root / "state/enhanced-indexing-research/replay-inputs" / uuid4().hex
        evidence.mkdir(parents=True, exist_ok=bool(reuse))
        frames, suspensions, requests = [], [], []
        for date in sessions:
            for endpoint, target in (("stk_limit", frames), ("suspend_d", suspensions)):
                try:
                    path = evidence / f"{endpoint}-{date}.parquet"
                    data = pd.read_parquet(path) if reuse else getattr(client, endpoint)(trade_date=date)
                    if data is None:
                        raise ValueError("NULL_RESPONSE")
                    if not reuse:
                        data.to_parquet(path, index=False)
                    target.append(data)
                    requests.append(dict(endpoint=endpoint, date=date, rows=len(data),
                                         file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                except Exception as error:
                    # Do not serialize provider error bodies: they may contain secrets.
                    requests.append(dict(endpoint=endpoint, date=date, error_type=type(error).__name__))
                if not reuse:
                    time.sleep(1.1)
        output["requests"] = requests
        output["evidence_directory"] = str(evidence)
        limits = pd.concat(frames, ignore_index=True) if frames else production_limits.iloc[:0]
        output["research_after"] = execution_coverage(daily, limits, calendar, symbols, start, end)
        suspension_frame = pd.concat(suspensions, ignore_index=True) if suspensions else pd.DataFrame()
        confirmed = set()
        if not suspension_frame.empty and "suspend_timing" in suspension_frame:
            full = suspension_frame.loc[suspension_frame.suspend_type.eq("S") &
                                        (suspension_frame.suspend_timing.isna() | suspension_frame.suspend_timing.eq(""))]
            confirmed = set(zip(full.trade_date.astype(str), full.ts_code))
        gaps = output["research_after"]["missing_market_keys"]
        output["market_gap_audit"] = [{**gap, "official_full_session_suspension":
                                      (gap["trade_date"], gap["ts_code"]) in confirmed} for gap in gaps]
        output["suspension_response_dates"] = [r["date"] for r in requests if r["endpoint"] == "suspend_d" and "rows" in r]
        output["remaining_checks"] = ["Every missing market bar needs official full-session suspension or listing-status proof",
                                      "Corporate action entitlement and payment coverage",
                                      "Chronological predictions and rolling risk inputs",
                                      "Independent persistent holdings and next-open broker replay"]
    write_diagnostic(root / "state/enhanced-indexing-research/replay-preflight.json", output)
    print(output.get("research_after", output["production_before"]))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--reuse-evidence")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.start, args.end, args.fetch, args.reuse_evidence)
