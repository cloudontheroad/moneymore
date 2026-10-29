"""Bounded cross-provider evidence for whole-union unexplained factor steps.

No normalization or release authorization. Nearby event windows for the same
security/year share a cached query. Each event is still verified individually.
"""

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..challenger_diagnostics import write_diagnostic
from .baseline_v2_adjustment_verification import (
    PRICE_FIELDS,
    _read_cache,
    _save,
    verify_no_economic_adjustment,
)
from .baseline_v2_release import digest, read_json, verify_manifest
from .security_identity import SecurityIdentity


class RequestBudgetReached(Exception):
    """Local batch boundary, not a provider failure or retry trigger."""


def verify_union_batch(root, store, client, manifest, *, max_pairs=8):
    verify_manifest(manifest)
    directory = Path(manifest["directory"])
    source = directory / "full-union-native-prices.parquet"
    native = pd.read_parquet(source).sort_values(["ts_code", "trade_date"])
    native["previous_date"] = native.groupby("ts_code").trade_date.shift()
    issues = pd.read_parquet(directory / "full-union-action-alignment.parquet")
    issues = issues.loc[
        issues.action_alignment.eq("NO_IMPLEMENTED_EVENT_IN_CACHE_UNRESOLVED")
        & np.isclose(issues.pre_close, issues.previous_close, rtol=0, atol=1e-8)
    ].merge(
        native[["ts_code", "trade_date", "previous_date"]],
        on=["ts_code", "trade_date"],
        validate="one_to_one",
    )
    source_digest = digest(source)
    scope = hashlib.sha256(
        (
            source_digest
            + digest(directory / "full-union-action-alignment.parquet")
            + issues[["ts_code", "trade_date", "previous_date", "adj_factor"]].to_json()
        ).encode()
    ).hexdigest()
    cache = root / "state/baseline-v2-release/union-independent" / scope[:20]
    cache.mkdir(parents=True, exist_ok=True)
    cooldown = cache / "cooldown.json"
    if cooldown.exists() and read_json(cooldown)["retry_after"] > time.time():
        return {"status": "PROVIDER_COOLDOWN", "requests": 0}
    frozen = read_json(Path(manifest["replay_directory"]) / "frozen-protocol.json")
    identity = SecurityIdentity(frozen["scoring"]["coverage_audit"]["identity_policy"])
    requests, new_pairs = 0, 0
    budget = max(0, min(64, int(max_pairs)))
    # HTTP requests remain capped at 16 even when cached/merged frames allow
    # more than eight event pairs to be verified in a single batch.
    windows = {}
    for (stable, year), group in issues.groupby(["ts_code", issues.trade_date.str[:4]]):
        candidates = group.loc[
            group.apply(
                lambda r, stable=stable: (
                    identity.tradable(stable, r.previous_date)
                    == identity.tradable(stable, r.trade_date)
                ),
                axis=1,
            )
        ]
        if not candidates.empty:
            windows[stable, year] = (
                str(candidates.previous_date.min()),
                str(candidates.trade_date.max()),
            )
    with sqlite3.connect(cache / "lock.sqlite3", timeout=0) as lock:
        lock.execute("BEGIN IMMEDIATE")
        for row in issues.sort_values(["ts_code", "trade_date"]).itertuples():
            key = f"{row.ts_code}-{row.trade_date}"
            result_path = cache / (key + ".json")
            if result_path.exists():
                cached = read_json(result_path)
                if cached["native_source_sha256"] != source_digest or any(
                    digest(p) != checksum for p, checksum in cached["artifact_hashes"].items()
                ):
                    raise ValueError("INDEPENDENT_PROOF_SOURCE_OR_ARTIFACT_CHANGED")
                continue
            if new_pairs >= budget:
                break
            if requests >= 16:
                break
            symbol = identity.tradable(row.ts_code, row.trade_date)
            if identity.tradable(row.ts_code, row.previous_date) != symbol:
                continue  # Identity transition requires its own event proof.
            official = store.read(
                "daily",
                columns=["ts_code", "trade_date", "pre_close", *PRICE_FIELDS],
                filters=[
                    ("ts_code", "==", symbol),
                    ("trade_date", "in", [row.previous_date, row.trade_date]),
                ],
            )
            if len(official) != 2 or official.trade_date.duplicated().any():
                raise ValueError("OFFICIAL_TWO_SESSION_OHLC_NOT_COMPLETE")
            official = official.set_index("trade_date")
            responses, evidence = {}, []
            try:
                for adjustment in ("none", "backward"):
                    # Retain and reuse original two-session proof caches first.
                    path = cache / adjustment / (key + ".parquet")
                    frame = _read_cache(path)
                    if frame is None:
                        start, end = windows[row.ts_code, row.trade_date[:4]]
                        if identity.tradable(row.ts_code, start) != symbol:
                            start, end = row.previous_date, row.trade_date
                        path = cache / "merged" / adjustment / f"{symbol}-{start}-{end}.parquet"
                        frame = _read_cache(path)
                    if frame is None:
                        if requests >= 16:
                            raise RequestBudgetReached
                        time.sleep(1.3)
                        requests += 1
                        start_ms = int(pd.Timestamp(start, tz="Asia/Shanghai").timestamp() * 1000)
                        end_ms = (
                            int(
                                (
                                    pd.Timestamp(end, tz="Asia/Shanghai") + pd.Timedelta(days=1)
                                ).timestamp()
                                * 1000
                            )
                            - 1
                        )
                        frame = client.klines.get(
                            symbol,
                            period="1d",
                            adjust=adjustment,
                            count=400,
                            start_time=start_ms,
                            end_time=end_ms,
                            to_dataframe=True,
                        )
                        if (
                            frame is None
                            or not {"symbol", "trade_date", *PRICE_FIELDS}.issubset(frame.columns)
                            or frame.empty
                            or len(frame) >= 400
                        ):
                            raise ValueError("INDEPENDENT_TWO_SESSION_SCHEMA_OR_COVERAGE")
                        frame = frame.copy()
                        frame["trade_date"] = pd.to_datetime(frame.trade_date).dt.strftime("%Y%m%d")
                        if (
                            not frame.symbol.eq(symbol).all()
                            or frame.trade_date.duplicated().any()
                            or not frame.trade_date.between(start, end).all()
                            or not {row.previous_date, row.trade_date}.issubset(
                                set(frame.trade_date)
                            )
                        ):
                            raise ValueError("INDEPENDENT_SYMBOL_OR_DATE_FILTER")
                        _save(
                            path,
                            frame,
                            {
                                "source": "alphafeed",
                                "symbol": symbol,
                                "adjust": adjustment,
                                "start": start,
                                "end": end,
                            },
                        )
                    responses[adjustment] = frame.set_index("trade_date")
                    evidence += [path, path.with_suffix(".json")]
            except RequestBudgetReached:
                break
            except Exception as error:  # noqa: BLE001 - never log provider text or credentials
                result = {
                    "status": "STOPPED_PROVIDER_ERROR",
                    "error_type": type(error).__name__,
                    "requests": requests,
                    "retry_after": time.time() + 1800,
                }
                write_diagnostic(cooldown, result)
                write_diagnostic(cache / "progress.json", result)
                return result
            verified, reason = verify_no_economic_adjustment(
                official.loc[row.previous_date],
                official.loc[row.trade_date],
                responses["none"].loc[row.previous_date],
                responses["none"].loc[row.trade_date],
                responses["backward"].loc[row.previous_date],
                responses["backward"].loc[row.trade_date],
                has_action=False,
            )
            write_diagnostic(
                result_path,
                {
                    "symbol": row.ts_code,
                    "trade_date": row.trade_date,
                    "verified": bool(verified),
                    "reason": reason,
                    "native_source_sha256": source_digest,
                    "official_ohlc": official.reset_index().to_dict("records"),
                    "artifact_hashes": {str(p): digest(p) for p in evidence},
                    "source_normalized": False,
                    "release_authorized": False,
                },
            )
            new_pairs += 1
    details = [
        read_json(p)
        for p in cache.glob("*.json")
        if p.name not in {"progress.json", "cooldown.json"}
    ]
    result = {
        "status": "BOUNDED_INDEPENDENT_BATCH_COMPLETE",
        "requests": requests,
        "eligible_non_event_pairs": len(issues),
        "reviewed_pairs": len(details),
        "new_pairs": new_pairs,
        "max_http_requests_per_batch": 16,
        "max_pairs_per_batch": budget,
        "verified_non_economic_pairs": sum(r["verified"] for r in details),
        "reviewed_but_not_verified_pairs": sum(not r["verified"] for r in details),
        "unresolved_pairs": len(issues) - sum(r["verified"] for r in details),
        "remaining_pairs": len(issues) - len(details),
        "directory": str(cache),
        "source_normalized": False,
        "real_orders_enabled": False,
    }
    write_diagnostic(cache / "progress.json", result)
    write_diagnostic(root / "state/baseline-v2-release/union-independent-progress.json", result)
    return result
