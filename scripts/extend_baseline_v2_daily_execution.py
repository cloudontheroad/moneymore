"""Bounded source evidence for symbols stopped by the latest actual daily replay."""

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.data.tushare_provider import TushareProvider
from moneymore.research.baseline_v2_execution_data import (
    cache_entry,
    read_verified_cache,
    validate_response,
)
from moneymore.research.baseline_v2_gap_evidence import full_session_dates
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.reviewed_action_classes import select_reviewed_retail_actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closure", action="store_true", help="Acquire conservative support, never trading targets"
    )
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--score-panel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    if args.symbols:
        if args.closure or not args.score_panel or not args.score_panel.is_file():
            raise ValueError("EXPLICIT_CORRECTED_SYMBOLS_REQUIRE_SAVED_SCORE_PANEL")
        panel = args.score_panel.resolve()
        if not panel.is_relative_to(root / "state/baseline-v2-release"):
            raise ValueError("CORRECTED_PANEL_OUTSIDE_ISOLATED_RESEARCH")
        symbols = sorted(set(args.symbols))
        replay = {"start": "20240131", "end": "20251231"}
        scores = {"panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest()}
        base = root / "state/baseline-v2-release/corrected-execution-data"
    else:
        replay = json.loads((base / "daily-replay.json").read_text(encoding="utf-8"))
        symbols = sorted(
            {s for run in replay["runs"] for s in run.get("missing_execution_symbols", [])}
        )
        scores = json.loads((base / "daily-scores.json").read_text(encoding="utf-8"))
    if not args.closure and (not symbols or len(symbols) > 8):
        raise ValueError("ONE_TO_EIGHT_EXPLICIT_MISSING_SYMBOLS_REQUIRED")
    start, end = replay["start"], replay["end"]
    remaining_closure = None
    if args.closure:
        closure = json.loads((base / "execution-closure.json").read_text(encoding="utf-8"))
        support_path = Path(closure["output_directory"]) / "candidate-support.parquet"
        if (
            closure["score_panel_sha256"] != scores["panel_sha256"]
            or hashlib.sha256(support_path.read_bytes()).hexdigest() != closure["support_sha256"]
        ):
            raise ValueError("FROZEN_EXECUTION_CLOSURE_MISMATCH")
        start, end = closure["start"], closure["end"]
        original = json.loads(
            (root / "state/baseline-v2-validation/execution-preflight.json").read_text(
                encoding="utf-8"
            )
        )
        known = set(original["required_data_symbols"])
        for marker in (base / "execution-extensions").glob("*/report.json"):
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if (
                existing["score_panel_sha256"] == scores["panel_sha256"]
                and existing["start"] <= start
                and existing["end"] >= end
                and existing["status"] == "READY_FOR_BOUNDED_DAILY_REFERENCE_EXECUTION"
            ):
                for name, checksum in existing["artifact_hashes"].items():
                    if (
                        hashlib.sha256(
                            (Path(existing["output_directory"]) / name).read_bytes()
                        ).hexdigest()
                        != checksum
                    ):
                        raise ValueError("CLOSURE_EXISTING_EVIDENCE_HASH_MISMATCH")
                known.update(existing["symbols"])
        pending = sorted(set(closure["required_symbols"]) - known)
        if not pending:
            progress = {
                "status": "CLOSURE_EXECUTION_EVIDENCE_COMPLETE",
                "requests": 0,
                "required_symbols": len(closure["required_symbols"]),
                "remaining_symbols": 0,
            }
            write_diagnostic(base / "closure-acquisition.json", progress)
            print(json.dumps(progress))
            raise SystemExit(0)
        symbols = pending[:8]
        remaining_closure = len(pending) - len(symbols)
    scope = {
        "start": start,
        "end": end,
        "symbols": symbols,
        "score_panel_sha256": scores["panel_sha256"],
    }
    digest = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    directory = base / "execution-extensions" / digest[:20]
    directory.mkdir(parents=True, exist_ok=True)
    load_dotenv(root / ".env", override=False)
    provider, requests = TushareProvider(), 0
    frames = {kind: [] for kind in ["limits", "dividends", "suspensions", "adjustment_histories"]}
    with sqlite3.connect(directory / "sync-lock.sqlite3", timeout=0) as lock:
        lock.execute("BEGIN IMMEDIATE")
        cooldown = directory / "cooldown.json"
        if cooldown.exists() and json.loads(cooldown.read_text())["retry_after"] > time.time():
            raise ValueError("EXTENSION_PROVIDER_COOLDOWN")
        for symbol in symbols:
            for kind, captures in frames.items():
                request = {"kind": kind, "symbol": symbol, "start": start, "end": end}
                path = cache_entry(directory, request)
                frame = read_verified_cache(path, request)
                if frame is None:
                    for prior in sorted(
                        (base / "execution-extensions").glob("*/" + kind + "/" + path.name)
                    ):
                        frame = read_verified_cache(prior, request)
                        if frame is not None:
                            break
                if frame is None:
                    if requests >= 32:
                        raise ValueError("EXTENSION_BUDGET_EXHAUSTED")
                    time.sleep(1.3)
                    requests += 1
                    try:
                        if kind == "limits":
                            frame = provider.stock_limits(symbol, start, end)
                        elif kind == "dividends":
                            frame = provider.dividends(symbol)
                        elif kind == "suspensions":
                            frame = provider.suspensions(symbol, start, end)
                        else:
                            frame = provider.historical_adjustments(symbol, start, end)
                        frame = validate_response(frame, kind, symbol, start, end)
                    except Exception as exc:  # noqa: BLE001 - no credential-bearing errors or retries
                        write_diagnostic(
                            cooldown,
                            {"retry_after": time.time() + 1800, "error_type": type(exc).__name__},
                        )
                        raise RuntimeError("EXTENSION_PROVIDER_STOPPED") from None
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                    write_diagnostic(
                        path.with_suffix(".json"),
                        {
                            "request": request,
                            "rows": len(frame),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "captured_at": datetime.now(UTC).isoformat(),
                        },
                    )
                captures.append(frame)
        official = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
        store = ParquetStore(root / "data")
        calendar = [d for d in trading_dates(store, end) if start <= d <= end]
        required = pd.MultiIndex.from_product(
            [calendar, symbols], names=["trade_date", "ts_code"]
        ).to_frame(index=False)
        prices = store.read(
            "daily",
            columns=["ts_code", "trade_date", "open", "close"],
            filters=[
                ("ts_code", "in", symbols),
                ("trade_date", ">=", start),
                ("trade_date", "<=", end),
            ],
        )
        checked = required.merge(
            prices, on=["trade_date", "ts_code"], how="left", validate="one_to_one"
        )
        checked = checked.merge(
            official["limits"], on=["trade_date", "ts_code"], how="left", validate="one_to_one"
        )
        checked["price_ok"] = (
            np.isfinite(checked.open)
            & checked.open.gt(0)
            & np.isfinite(checked.close)
            & checked.close.gt(0)
        )
        checked["limits_ok"] = (
            np.isfinite(checked.up_limit)
            & np.isfinite(checked.down_limit)
            & checked.down_limit.gt(0)
            & checked.up_limit.gt(checked.down_limit)
        )
        suspend = official["suspensions"]
        proofs = {s: full_session_dates(f) for s, f in suspend.groupby("ts_code")}
        checked["verified_full_session_suspension"] = [
            d in proofs.get(s, set())
            for s, d in zip(checked.ts_code, checked.trade_date, strict=True)
        ]
        listing = store.read("instruments").set_index("ts_code").list_date
        listing_dates = pd.to_datetime(
            checked.ts_code.map(listing), format="%Y%m%d", errors="coerce"
        )
        if listing_dates.isna().any():
            raise ValueError("EXTENSION_LISTING_SCOPE_REQUIRES_REVIEW")
        checked["not_yet_listed"] = pd.to_datetime(checked.trade_date, format="%Y%m%d").lt(
            listing_dates
        )
        lifecycle = yaml.safe_load(
            (root / "configs/baseline_v2_market_evidence.yaml").read_text(encoding="utf-8")
        ).get("delistings", [])
        checked["after_verified_delisting"] = False
        for rule in lifecycle:
            if rule["symbol"] not in symbols:
                continue
            instrument = store.read("instruments").set_index("ts_code").loc[rule["symbol"]]
            if str(instrument.delist_date) != rule["effective"]:
                raise ValueError("DELISTING_SOURCE_METADATA_CONFLICT")
            checked.loc[
                checked.ts_code.eq(rule["symbol"]) & checked.trade_date.ge(rule["effective"]),
                "after_verified_delisting",
            ] = True
        invalid_market = (
            (~checked.price_ok | ~checked.limits_ok)
            & ~checked.verified_full_session_suspension
            & ~checked.not_yet_listed
            & ~checked.after_verified_delisting
        )
        if invalid_market.any():
            checked.loc[invalid_market].to_parquet(
                directory / "blocked-market-gaps.parquet", index=False
            )
            raise ValueError("EXTENSION_UNVERIFIED_PRICE_OR_LIMIT_GAP")
        raw_dividends = official["dividends"]
        # Initial accounts start in cash: only records inside the replay can create rights.
        relevant = raw_dividends.loc[
            pd.to_datetime(raw_dividends.record_date, errors="coerce").between(
                pd.Timestamp(start), pd.Timestamp(end)
            )
        ]
        class_config = root / "configs/baseline_v2_action_classes.yaml"
        class_rules = yaml.safe_load(class_config.read_text(encoding="utf-8"))["classes"]
        relevant, class_audit = select_reviewed_retail_actions(relevant, class_rules)
        for rule in class_audit:
            document = root / rule["local_document"]
            if hashlib.sha256(document.read_bytes()).hexdigest() != rule["sha256"]:
                raise ValueError("REVIEWED_ACTION_DOCUMENT_HASH_MISMATCH")
        actions = official_replay_actions(relevant)
        checked.to_parquet(directory / "market-data-readiness-extra.parquet", index=False)
        actions.to_parquet(directory / "replay-actions-extra.parquet", index=False)
        for kind, frame in official.items():
            frame.to_parquet(directory / ("official-" + kind + ".parquet"), index=False)
        result = {
            "status": "READY_FOR_BOUNDED_DAILY_REFERENCE_EXECUTION",
            **scope,
            "output_directory": str(directory),
            "requests": requests,
            "remaining_closure_symbols": remaining_closure,
            "quote_rows": len(checked),
            "unexplained_market_gaps": 0,
            "implemented_record_events": len(actions),
            "reviewed_retail_action_classes": class_audit,
            "verified_delisting_rules": [r for r in lifecycle if r["symbol"] in symbols],
            "action_class_config_sha256": hashlib.sha256(class_config.read_bytes()).hexdigest(),
            "artifact_hashes": {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ["market-data-readiness-extra.parquet", "replay-actions-extra.parquet"]
            },
            "limitations": [
                "NOT_FULL_HISTORY_ADJUSTMENT_ECONOMIC_VALIDATION",
                "REFERENCE_QUOTES_NOT_ACTUAL_PRINTS",
            ],
            "production_changed": False,
            "real_orders_enabled": False,
        }
        write_diagnostic(directory / "report.json", result)
        write_diagnostic(base / "execution-extension.json", result)
        if args.closure:
            write_diagnostic(
                base / "closure-acquisition.json",
                {
                    "status": "CLOSURE_EXECUTION_EVIDENCE_COMPLETE"
                    if remaining_closure == 0
                    else "CLOSURE_ACQUISITION_IN_PROGRESS",
                    "required_symbols": len(closure["required_symbols"]),
                    "remaining_symbols": remaining_closure,
                    "requests_this_batch": requests,
                    "last_symbols": symbols,
                    "score_panel_sha256": scores["panel_sha256"],
                    "last_extension": str(directory),
                },
            )
        print(json.dumps(result))
