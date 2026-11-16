"""Read-only validation of the frozen research cache and disclosure of missing values."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import yaml
from prepare_ashare_three_year import BASE, CONFIG, ROOT
from prepare_four_strategy_2026 import save

from moneymore.research.ashare_mainline import FUNDAMENTALS, sha256, validate_financial_dates
from moneymore.research.replay_actions import official_replay_actions
from moneymore.research.three_year_actions import AppendOnlyActionCache, reviewed_actions


def audit_action_cache(raw):
    cache = AppendOnlyActionCache()
    cumulative = pd.DataFrame()
    start = time.monotonic()
    for day, rows in raw.groupby("record_date", sort=True):
        cumulative = pd.concat([cumulative, rows], ignore_index=True).drop_duplicates()
        actual = cache(cumulative)
        if len(actual) and actual.record_date.max() > pd.Timestamp(day):
            raise ValueError("FUTURE_CACHED_ACTION")
    cached_seconds = time.monotonic() - start
    expected = official_replay_actions(cumulative)
    # Raw row-number gaps are not event identity; action_key and every event value are.
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True
    )
    return {
        "prefix_dates": raw.record_date.nunique(),
        "actions": len(expected),
        "exact_uncached_equivalence": True,
        "no_future_events": True,
        "cached_all_prefix_seconds": cached_seconds,
    }


def main():
    data = BASE / "data-v1"
    cfg = json.loads(CONFIG.read_bytes())
    manifests = {
        name: json.loads((data / (name + "-results.json")).read_bytes())
        for name in ["market", "financial", "execution"]
    }
    for name, digest in manifests["market"]["hashes"].items():
        if sha256(data / name) != digest:
            raise ValueError("MARKET_HASH_MISMATCH:" + name)
    for endpoint, digest in manifests["financial"]["hashes"].items():
        if sha256(data / (endpoint + ".parquet")) != digest:
            raise ValueError("FINANCIAL_HASH_MISMATCH:" + endpoint)
    for record in manifests["execution"]["files"]:
        if sha256(data / "official" / record["file"]) != record["sha256"]:
            raise ValueError("EXECUTION_HASH_MISMATCH:" + record["file"])
    calendar = json.loads((data / "calendar.json").read_bytes())["days"]
    sessions = [d for d in calendar if cfg["evaluation_start"] <= d <= cfg["evaluation_end"]]
    members = pd.read_parquet(data / "membership.parquet")
    basic = pd.read_parquet(data / "daily_basic.parquet")
    bars = pd.read_parquet(data / "bars.parquet")
    panel = pd.read_parquet(BASE / "panel-v1/panel.parquet")
    meta = json.loads((BASE / "panel-v1/results.json").read_bytes())
    core = pd.read_parquet(BASE / "panel-v1/core-panel.parquet")
    pd.testing.assert_frame_equal(core, panel[core.columns], check_exact=True)
    if sha256(BASE / "panel-v1/panel.parquet") != meta["panel_sha256"]:
        raise ValueError("PANEL_HASH_MISMATCH")
    validate_financial_dates(panel)
    if (
        members.duplicated(["date", "symbol"]).any()
        or not members.groupby("date").size().eq(1000).all()
    ):
        raise ValueError("INVALID_DAILY_HISTORICAL_UNIVERSE")
    basic_groups = {d: g for d, g in basic.groupby("trade_date")}
    for day, group in members.groupby("date"):
        available = basic_groups[day]
        available = available.loc[
            available.ts_code.str.endswith((".SH", ".SZ")) & available.total_mv.gt(0)
        ]
        reference = available.sort_values(["total_mv", "ts_code"], ascending=[False, True]).head(
            1000
        )
        if set(reference.ts_code) != set(group.symbol):
            raise ValueError("HISTORICAL_TOP1000_NOT_REPRODUCIBLE:" + day)
    if bars.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("DUPLICATE_DAILY_BARS")
    q = bars.loc[bars.trade_date.isin(sessions)]
    invalid = q.loc[
        ~np.isfinite(q[["open", "high", "low", "close", "adj_factor"]]).all(axis=1)
        | q.close.le(0)
        | q.open.le(0)
        | q.adj_factor.le(0)
        | q.adj_factor.isna()
        | q.high.lt(q[["open", "close", "low"]].max(axis=1) - 1e-6)
        | q.low.gt(q[["open", "close", "high"]].min(axis=1) + 1e-6)
    ]
    if len(invalid):
        raise ValueError("INVALID_REPLAY_BAR_OR_ADJUSTMENT:" + str(len(invalid)))
    expected = members.loc[members.date.isin(sessions), ["date", "symbol"]]
    observed = q.rename(columns={"trade_date": "date", "ts_code": "symbol"})
    missing = expected.merge(
        observed[["date", "symbol"]], how="left", on=["date", "symbol"], indicator=True
    )
    missing = missing.loc[missing._merge.eq("left_only"), ["date", "symbol"]]
    halts = pd.concat(
        [pd.read_parquet(p) for p in (data / "official").glob("suspend_d-*.parquet")],
        ignore_index=True,
    )
    complete_halts = halts.loc[
        halts.suspend_type.eq("S") & (halts.suspend_timing.isna() | halts.suspend_timing.eq(""))
    ]
    proof = set(zip(complete_halts.trade_date, complete_halts.ts_code))
    missing["official_full_day_halt"] = [
        (d, s) in proof for d, s in zip(missing.date, missing.symbol)
    ]
    missing.to_parquet(data / "missing-membership-quotes-audit.parquet", index=False)
    # Missing candidate quotes remain ineligible, never fabricated. Held missing quotes halt replay.
    raw_actions = pd.concat(
        [pd.read_parquet(p) for p in (data / "official").glob("dividend-*.parquet")],
        ignore_index=True,
    ).drop_duplicates()
    raw_actions, action_evidence = reviewed_actions(
        raw_actions.loc[raw_actions.ts_code.isin(members.symbol.unique())],
        yaml.safe_load((ROOT / "configs/baseline_v2_action_classes.yaml").read_text())["classes"],
        json.loads((ROOT / "configs/ashare_three_year_action_evidence.json").read_bytes())[
            "cash_aggregate_overlap"
        ],
    )
    actions = official_replay_actions(raw_actions)
    cache_audit = audit_action_cache(raw_actions)
    actions.to_parquet(data / "audited-actions.parquet", index=False)
    coverage = panel.groupby(panel.date.str[:4])[FUNDAMENTALS + cfg["financial_features"]].agg(
        lambda x: float(x.notna().mean())
    )
    coverage.to_parquet(data / "financial-coverage-audit.parquet")
    known = panel.available.notna()
    if (panel.loc[known, "available"] > pd.to_datetime(panel.loc[known, "date"])).any():
        raise ValueError("FUTURE_FINANCIAL_FEATURE")
    instruments = pd.read_parquet(data / "instruments.parquet")
    retired = instruments.loc[
        instruments.ts_code.isin(members.symbol.unique())
        & instruments.delist_date.astype(str).between(sessions[0], sessions[-1]),
        ["ts_code", "delist_date"],
    ]
    unknown = panel.loc[panel.industry.isna()].groupby(panel.date.str[:4]).size().to_dict()
    record = {
        "status": "COMPLETE_WITH_DISCLOSED_LIMITATIONS",
        "window": [sessions[0], sessions[-1]],
        "sessions": len(sessions),
        "historical_union_symbols": members.symbol.nunique(),
        "historical_member_days": members.date.nunique(),
        "top1000_every_day_independently_reconstructed": True,
        "core_features_exact_after_augmentation": True,
        "missing_candidate_quotes": len(missing),
        "missing_with_official_halt": int(missing.official_full_day_halt.sum()),
        "missing_without_official_halt": int((~missing.official_full_day_halt).sum()),
        "unknown_industry_rows_by_year": unknown,
        "financial_coverage": coverage.to_dict(),
        "label_unavailable_rows_by_year": panel.loc[panel.label_20.isna()]
        .groupby(panel.date.str[:4])
        .size()
        .to_dict(),
        "actions": len(actions),
        "action_source_overlap_evidence": action_evidence,
        "action_cache_equivalence": cache_audit,
        "historical_delistings": retired.to_dict("records"),
        "missing_values_kept_missing": True,
        "unproven_held_quotes_block_replay": True,
        "original_financial_vintages_certified": False,
        "production_modified": False,
        "panel_sha256": meta["panel_sha256"],
        "core_panel_sha256": meta["core_panel_sha256"],
        "input_manifest_hashes": {
            name: sha256(data / (name + "-results.json")) for name in manifests
        },
    }
    save(BASE / "data-audit-v1.json", record)
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in ["financial_coverage", "historical_delistings"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
