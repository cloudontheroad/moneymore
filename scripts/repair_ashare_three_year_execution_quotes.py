"""Fill execution-only historical price holes without revising frozen model inputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from prepare_ashare_three_year import BASE, Fetcher
from prepare_four_strategy_2026 import save

from moneymore.research.ashare_mainline import sha256


def main():
    data = BASE / "data-v1"
    out = BASE / "execution-repair-v1"
    out.mkdir(exist_ok=True)
    if (out / "results.json").exists():
        print("EXECUTION_REPAIR_ALREADY_COMPLETE", flush=True)
        return
    missing = pd.read_parquet(data / "missing-membership-quotes-audit.parquet")
    days = sorted(missing.date.unique())
    bars = pd.read_parquet(data / "bars.parquet")
    symbols = set(pd.read_parquet(data / "membership.parquet").symbol)
    known = pd.MultiIndex.from_frame(bars[["trade_date", "ts_code"]])
    f = Fetcher()
    f.config = {
        **f.config,
        "fetch": {**f.config["fetch"], "max_requests": 120, "interval_seconds": 0.8},
    }
    f.journal = out / "requests.jsonl"
    f.previous_calls = len(f.journal.read_text().splitlines()) if f.journal.exists() else 0
    pieces = []
    sources = []
    for day in days:
        raw = f.get("daily", "execution-repair-" + day, trade_date=day, nonempty=True)
        adjustment = f.get("adj_factor", "execution-repair-" + day, trade_date=day, nonempty=True)
        if (
            len(raw) < 3000
            or not raw.trade_date.eq(day).all()
            or not adjustment.trade_date.eq(day).all()
        ):
            raise ValueError("INCOMPLETE_OFFICIAL_EXECUTION_REPAIR_DAY")
        extra = raw.loc[raw.ts_code.isin(symbols)].merge(
            adjustment, on=["ts_code", "trade_date"], validate="one_to_one"
        )
        extra = extra.loc[
            ~pd.MultiIndex.from_frame(extra[["trade_date", "ts_code"]]).isin(known), bars.columns
        ]
        if (
            not np.isfinite(extra[["open", "high", "low", "close", "adj_factor"]]).all().all()
            or (extra[["open", "high", "low", "close", "adj_factor"]] <= 0).any().any()
        ):
            raise ValueError("INVALID_EXECUTION_REPAIR_PRICE")
        pieces.append(extra)
        for endpoint in ["daily", "adj_factor"]:
            source = data / "official" / (endpoint + "-execution-repair-" + day + ".parquet")
            sources.append({"file": source.name, "sha256": sha256(source)})
        print(
            json.dumps({"date": day, "extra_rows": len(extra), "new_requests": f.calls}), flush=True
        )
    extra = pd.concat(pieces, ignore_index=True) if pieces else bars.iloc[:0].copy()
    if extra.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("DUPLICATE_EXECUTION_REPAIR")
    proof = missing.merge(
        extra,
        left_on=["date", "symbol"],
        right_on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    if proof.close.isna().any():
        raise ValueError("UNRESOLVED_CANDIDATE_PRICE_HOLES")
    extra.to_parquet(out / "extra-bars.parquet", index=False)
    result = {
        "status": "COMPLETE_EXECUTION_ONLY_REPAIR",
        "original_candidate_missing_rows": len(missing),
        "additional_bars": len(extra),
        "repaired_dates": days,
        "source_files": sources,
        "extra_sha256": sha256(out / "extra-bars.parquet"),
        "base_sha256": sha256(data / "bars.parquet"),
        "frozen_training_and_selection_risk_inputs_unchanged": True,
        "model_feature_holes_retained_and_disclosed": True,
        "rationale": "Actual fills and held-stock marks require official prices; do not revise already fitted predictors or their frozen risk eligibility after seeing results.",
        "max_supplement_requests": 120,
        "actual_requests": f.previous_calls + f.calls,
        "production_modified": False,
    }
    save(out / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
