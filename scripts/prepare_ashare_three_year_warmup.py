"""One bounded, resumable repair of pre-2022 financial warmup, before model fitting.

The main study configuration and frozen core panel remain unchanged. This supplemental
collection has its own 3,800-request ceiling, journal, and provenance. No VIP retry.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime

import pandas as pd
from prepare_ashare_three_year import BASE, ROOT, Fetcher
from prepare_four_strategy_2026 import save

from moneymore.research.ashare_mainline import sha256


def main():
    out = BASE / "warmup-v1"
    out.mkdir(exist_ok=True)
    if (out / "results.json").exists():
        print("WARMUP_ALREADY_COMPLETE", flush=True)
        return
    if list((BASE / "training-v1").glob("*/financial_rank_cap/results.json")):
        raise ValueError("CANNOT_REPLACE_FINANCIAL_INPUT_AFTER_TRAINING")
    core_hash = sha256(BASE / "panel-v1/core-panel.parquet")
    backup = out / "original"
    backup.mkdir(exist_ok=True)
    fields = {
        "cashflow": "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,n_cashflow_act",
        "income": "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,operate_profit,n_income_attr_p",
        "fina_indicator": "ts_code,ann_date,end_date,update_flag,roe,q_sales_yoy,q_roe,netdebt,fcff",
    }
    for name in [*(e + ".parquet" for e in fields), "financial-results.json"]:
        source = BASE / "data-v1" / name
        if not (backup / name).exists():
            shutil.copy2(source, backup / name)
    # Invalidate only untrained augmentation artifacts, retaining them recoverably.
    for name in [
        "results.json",
        "panel.parquet",
        "financial-events.parquet",
        "feature-coverage.parquet",
    ]:
        source = BASE / "panel-v1" / name
        target = backup / ("panel-" + name)
        if source.exists() and not target.exists():
            source.replace(target)
    f = Fetcher()
    f.config = {
        **f.config,
        "fetch": {**f.config["fetch"], "max_requests": 3800, "interval_seconds": 0.8},
    }
    f.journal = out / "requests.jsonl"
    f.previous_calls = len(f.journal.read_text().splitlines()) if f.journal.exists() else 0
    members = pd.read_parquet(BASE / "data-v1/membership.parquet")
    symbols = sorted(members.symbol.unique())
    old = ROOT / "state/ashare-simple-rules-20261113/data-v1"
    jobs = [(s, e) for s in symbols for e in fields if (old / f"{e}-{s}.parquet").exists()]
    save(
        out / "protocol.json",
        {
            "purpose": "Repair 2020-2021 warmup missing in reused caches before financial training",
            "jobs": len(jobs),
            "max_requests": 3800,
            "max_attempts": 2,
            "interval_seconds": 0.8,
            "permission_retry": False,
            "frozen_config_modified": False,
            "core_panel_sha256": core_hash,
            "original_hashes": {p.name: sha256(p) for p in backup.iterdir()},
            "asof_policy": "unchanged actual announcement plus one day; not original-vintage certified",
        },
    )
    pieces = {e: [] for e in fields}
    for i, (symbol, endpoint) in enumerate(jobs):
        frame = f.get(
            endpoint,
            symbol + "-warmup-2020-2021",
            ts_code=symbol,
            start_date="20200101",
            end_date="20211231",
            fields=fields[endpoint],
        )
        if not frame.empty and not frame.ts_code.eq(symbol).all():
            raise ValueError("WARMUP_SYMBOL_MISMATCH")
        if endpoint == "fina_indicator" and len(frame) >= 100:
            raise ValueError("WARMUP_INDICATOR_POSSIBLY_TRUNCATED")
        pieces[endpoint].append(frame)
        if i % 30 == 0 or i == len(jobs) - 1:
            status = {
                "status": "RUNNING",
                "completed": i + 1,
                "total": len(jobs),
                "new_requests": f.calls,
                "utc": datetime.now(UTC).isoformat(),
            }
            save(out / "progress.json", status)
            print(json.dumps(status), flush=True)
    if list((BASE / "training-v1").glob("*/financial_rank_cap/results.json")):
        raise ValueError("FINANCIAL_TRAINING_STARTED_DURING_REPAIR")
    hashes = {}
    counts = {}
    for endpoint in fields:
        original = pd.read_parquet(backup / (endpoint + ".parquet"))
        merged = pd.concat([original, *pieces[endpoint]], ignore_index=True).drop_duplicates()
        path = BASE / "data-v1" / (endpoint + ".parquet")
        temporary = path.with_suffix(".warmup.tmp")
        merged.to_parquet(temporary, index=False)
        temporary.replace(path)
        hashes[endpoint] = sha256(path)
        counts[endpoint] = {"before": len(original), "after": len(merged)}
    meta = json.loads((backup / "financial-results.json").read_bytes())
    meta["hashes"].update(hashes)
    meta["warmup_supplement"] = "warmup-v1/results.json"
    save(BASE / "data-v1/financial-results.json", meta)
    if sha256(BASE / "panel-v1/core-panel.parquet") != core_hash:
        raise ValueError("CORE_PANEL_CHANGED")
    result = {
        "status": "COMPLETE",
        "hashes": hashes,
        "rows": counts,
        "request_count": f.calls + f.previous_calls,
        "core_panel_sha256": core_hash,
        "production_modified": False,
    }
    save(out / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
