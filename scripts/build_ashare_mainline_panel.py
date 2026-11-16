"""N1: immutable research panel, reuse audited inputs without touching production."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from moneymore.research.ashare_mainline import (
    AGGREGATES,
    FUNDAMENTALS,
    MOMENTUM,
    assert_keys,
    market_context,
    sha256,
    validate_financial_dates,
)

ROOT = Path(__file__).resolve().parents[1]


def main(output):
    if output.exists():
        raise FileExistsError("IMMUTABLE_N1_OUTPUT")
    output.mkdir(parents=True)
    native = ROOT / "state/mature-multifactor/p1-native-20261103-a"
    repair = ROOT / "state/ashare-pareto-rebuild-20261111/label-repair-v2/panel.parquet"
    config = ROOT / "configs/ashare_mainline_v1.json"
    manifest = json.loads((native / "results.json").read_bytes())
    paths = [
        native / "alpha158-panel.parquet",
        native / "bars-input.parquet",
        repair,
        config,
        native / "provider/calendars/day.txt",
    ]
    for path in paths[:2]:
        if sha256(path) != manifest["input_hashes"][path.name]:
            raise ValueError("NATIVE_INPUT_HASH_MISMATCH:" + path.name)
    base = pd.read_parquet(repair)
    assert_keys(base)
    validate_financial_dates(base)
    names = manifest["native_features"]
    features = (
        pd.read_parquet(paths[0], columns=names)
        .reset_index()
        .rename(columns={"instrument": "symbol"})
    )
    features["date"] = features.datetime.dt.strftime("%Y%m%d")
    features = features.drop(columns="datetime")
    assert_keys(features)
    panel = base.merge(features, on=["date", "symbol"], how="left", validate="one_to_one")
    panel["alpha_feature_fraction"] = panel[names].notna().mean(axis=1)
    bars = pd.read_parquet(paths[1], columns=["trade_date", "ts_code", "close", "adj_factor"])
    bars.trade_date = bars.trade_date.astype(str)
    calendar = pd.DatetimeIndex(pd.to_datetime(paths[4].read_text().splitlines())).strftime(
        "%Y%m%d"
    )
    context, coverage = market_context(bars, base[["date", "symbol"]], calendar=calendar)
    panel = panel.merge(context, on="date", how="left", validate="many_to_one")
    panel["log_size"] = np.log(panel.total_mv.where(panel.total_mv.gt(0)))
    # Observable board identity, not a target weight or future classification.
    panel["board_star"] = panel.symbol.str.startswith("68").astype(float)
    panel["board_chinext"] = panel.symbol.str.startswith("30").astype(float)
    extra = (
        FUNDAMENTALS
        + MOMENTUM
        + ["log_size", "board_star", "board_chinext"]
        + list(context.columns)
    )
    groups = {"aggregate": AGGREGATES, "alpha158": names, "full": names + extra}
    for column in {name for group in groups.values() for name in group}:
        panel[column] = (
            pd.to_numeric(panel[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel.to_parquet(output / "panel.parquet", index=False)
    coverage.to_parquet(output / "market-coverage.parquet")
    report = {
        "status": "N1_READY_FOR_ISOLATED_TRAINING",
        "rows": len(panel),
        "dates": panel.date.nunique(),
        "first": panel.date.min(),
        "last": panel.date.max(),
        "feature_groups": groups,
        "feature_counts_before_training_filters": {k: len(v) for k, v in groups.items()},
        "coverage": {c: float(panel[c].notna().mean()) for c in extra},
        "eligible_alpha_rows": int(panel.alpha_feature_fraction.ge(0.8).sum()),
        "missing_labels_retained": int(panel.label_20.isna().sum()),
        "input_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
        "panel_sha256": sha256(output / "panel.parquet"),
        "limitations": [
            "historical_membership_carry",
            "vendor_financial_revisions_not_original_vintage",
            "financial_fields_limited_cashflow_and_margin_not_integrated",
            "2024_start_insufficient_for_long_market_regime_claims",
            "historical_data_previously_examined_not_pristine_holdout",
        ],
        "source_code_sha256": sha256(Path(__file__)),
        "production_modified": False,
        "real_orders_enabled": False,
    }
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("feature_groups", "input_hashes")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "state/ashare-mainline-20261112/panel-v1"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.output)
