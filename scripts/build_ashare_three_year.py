"""Native Alpha158 and announcement-aware features for the frozen 36-month study."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_ashare_three_year import BASE, CONFIG
from prepare_four_strategy_2026 import save
from threadpoolctl import threadpool_limits

from moneymore.data.fundamental_research import financial_features_as_of
from moneymore.research.ashare_mainline import (
    FUNDAMENTALS,
    MOMENTUM,
    assert_keys,
    market_context,
    sha256,
    validate_financial_dates,
)
from moneymore.research.baseline_v2_financials import common_period_features, required_period
from moneymore.research.baseline_v2_industry import industry_as_of
from moneymore.research.mature_multifactor import native_bar_fields
from moneymore.research.three_year import attach_financial_events, financial_events


def native():
    data = BASE / "data-v1"
    out = BASE / "panel-v1"
    out.mkdir(exist_ok=True)
    if (out / "native-results.json").exists():
        print("NATIVE_ALREADY_COMPLETE", flush=True)
        return
    cfg = json.loads(CONFIG.read_bytes())
    manifest = json.loads((data / "market-results.json").read_bytes())
    for name, digest in manifest["hashes"].items():
        if sha256(data / name) != digest:
            raise ValueError("MARKET_INPUT_HASH_CHANGED")
    calendar = json.loads((data / "calendar.json").read_bytes())["days"]
    bars = pd.read_parquet(data / "bars.parquet")
    members = pd.read_parquet(data / "membership.parquet")
    provider = out / "provider"
    for name in ["calendars", "instruments", "features"]:
        (provider / name).mkdir(parents=True, exist_ok=True)
    (provider / "calendars/day.txt").write_text(
        "\n".join(str(pd.Timestamp(d).date()) for d in calendar) + "\n"
    )
    mapping = {}
    spans = []
    for symbol, g in bars.groupby("ts_code", sort=True):
        code = symbol.split(".")[1] + symbol[:6]
        mapping[code] = symbol
        values, _ = native_bar_fields(g)
        spans.append(
            f"{code}\t{pd.Timestamp(values.index.min()).date()}\t{pd.Timestamp(values.index.max()).date()}"
        )
        dest = provider / "features" / code.lower()
        dest.mkdir(exist_ok=True)
        for field in values:
            path = dest / f"{field}.day.bin"
            if not path.exists():
                np.concatenate(
                    [
                        np.array([0], dtype="<f4"),
                        values[field].reindex(calendar).to_numpy(dtype="<f4"),
                    ]
                ).tofile(path)
    (provider / "instruments/all.txt").write_text("\n".join(spans) + "\n")
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D

    qlib.init(
        provider_uri=str(provider.resolve()),
        region="cn",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
        logging_level=30,
    )
    expressions, names = Alpha158DL.get_feature_config()
    if len(names) != 158:
        raise ValueError("ALPHA158_SCHEMA_CHANGED")
    member_index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(members.date), members.symbol], names=["datetime", "instrument"]
    )
    codes = sorted(mapping)
    chunks = out / "alpha-chunks"
    chunks.mkdir(exist_ok=True)
    paths = []
    for start in range(0, len(codes), 80):
        path = chunks / f"{start:05d}.parquet"
        if not path.exists():
            f = D.features(
                codes[start : start + 80],
                expressions,
                start_time=cfg["training_start"],
                end_time=cfg["evaluation_end"],
            )
            f.columns = names
            f = f.reset_index()
            f["instrument"] = f.instrument.map(mapping)
            f = f.set_index(["datetime", "instrument"])
            f = (
                f.loc[f.index.isin(member_index)]
                .replace([np.inf, -np.inf], np.nan)
                .astype("float32")
            )
            f.to_parquet(path)
        paths.append(path)
        status = {
            "phase": "NATIVE_ALPHA158",
            "pid": os.getpid(),
            "completed_symbols": min(len(codes), start + 80),
            "total_symbols": len(codes),
        }
        save(out / "progress.json", status)
        print(json.dumps(status), flush=True)
    features = pd.concat([pd.read_parquet(p) for p in paths]).reindex(member_index).sort_index()
    features.to_parquet(out / "alpha158.parquet")
    save(
        out / "native-results.json",
        {
            "status": "COMPLETE",
            "features": names,
            "rows": len(features),
            "sha256": sha256(out / "alpha158.parquet"),
            "market_hashes": manifest["hashes"],
            "qlib_version": qlib.__version__,
            "native_source_sha256": sha256(Path(inspect.getsourcefile(Alpha158DL))),
            "config_sha256": sha256(CONFIG),
        },
    )


def build_core():
    data = BASE / "data-v1"
    out = BASE / "panel-v1"
    cfg = json.loads(CONFIG.read_bytes())
    if (out / "core-results.json").exists():
        print("CORE_PANEL_ALREADY_COMPLETE", flush=True)
        return
    meta = json.loads((out / "native-results.json").read_bytes())
    if meta["sha256"] != sha256(out / "alpha158.parquet") or meta["config_sha256"] != sha256(
        CONFIG
    ):
        raise ValueError("NATIVE_INPUT_CHANGED")
    calendar = json.loads((data / "calendar.json").read_bytes())["days"]
    bars = pd.read_parquet(data / "bars.parquet")
    members = pd.read_parquet(data / "membership.parquet")
    basic = pd.read_parquet(data / "daily_basic.parquet")
    basic = {d: g.set_index("ts_code") for d, g in basic.groupby("trade_date")}
    financial = pd.read_parquet(data / "core-financials.parquet")
    industries = pd.read_parquet(data / "baseline_v2_industry_membership.parquet")
    price = (
        bars.assign(adjusted=bars.close * bars.adj_factor)
        .pivot(index="trade_date", columns="ts_code", values="adjusted")
        .reindex(calendar)
    )
    opened = (
        bars.assign(adjusted=bars.open * bars.adj_factor)
        .pivot(index="trade_date", columns="ts_code", values="adjusted")
        .reindex_like(price)
    )
    mom6 = price.shift(20) / price.shift(126) - 1
    mom12 = price.shift(20) / price.shift(252) - 1
    labels = price.shift(-20) / opened.shift(-1) - 1
    label_ends = dict(zip(calendar[:-20], calendar[20:]))
    context, coverage = market_context(bars, members, calendar=calendar)
    context.to_parquet(out / "market-context.parquet")
    coverage.to_parquet(out / "market-coverage.parquet")
    pieces = []
    for i, (day, g) in enumerate(members.groupby("date", sort=True)):
        symbols = sorted(g.symbol)
        f = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
        b = basic[day].reindex(symbols)
        f["total_mv"] = b.total_mv
        f["industry"] = industry_as_of(industries, day).reindex(symbols).l1_code
        current = common_period_features(financial, day).set_index("ts_code").reindex(symbols)
        prior = str(int(required_period(day)) - 10000)
        previous = (
            financial_features_as_of(financial.loc[financial.end_date.astype(str).eq(prior)], day)
            .set_index("ts_code")
            .reindex(symbols)
        )
        for col in ["roe", "q_sales_yoy"]:
            if col not in previous:
                previous[col] = np.nan
            if col not in current:
                current[col] = np.nan
        f["ep"] = 1 / b.pe_ttm.where(b.pe_ttm.ne(0))
        f["bp"] = 1 / b.pb.where(b.pb.gt(0))
        f["roe"] = current.roe
        f["sales_yoy"] = current.q_sales_yoy
        f["roe_change"] = current.roe - previous.roe
        f["sales_change"] = current.q_sales_yoy - previous.q_sales_yoy
        for col in ["roe_ann_date", "q_sales_yoy_ann_date"]:
            f[col] = current.get(col, None)
            f["previous_" + col] = previous.get(col, None)
        f["mom6_1"] = mom6.loc[day].reindex(symbols)
        f["mom12_1"] = mom12.loc[day].reindex(symbols)
        f["label_20"] = labels.loc[day].reindex(symbols)
        f["label_end_20"] = label_ends.get(day)
        f["date"] = day
        f["log_size"] = np.log(f.total_mv.where(f.total_mv.gt(0)))
        f["board_star"] = f.index.str.startswith("68").astype(float)
        f["board_chinext"] = f.index.str.startswith("30").astype(float)
        pieces.append(f.reset_index())
        if i % 50 == 0:
            status = {
                "phase": "CORE_FEATURES",
                "completed": i + 1,
                "total": members.date.nunique(),
                "day": day,
                "pid": os.getpid(),
            }
            save(out / "progress.json", status)
            print(json.dumps(status), flush=True)
    panel = pd.concat(pieces, ignore_index=True).merge(
        context, on="date", how="left", validate="many_to_one"
    )
    validate_financial_dates(panel)
    alpha = (
        pd.read_parquet(out / "alpha158.parquet")
        .reset_index()
        .rename(columns={"instrument": "symbol"})
    )
    alpha["date"] = alpha.datetime.dt.strftime("%Y%m%d")
    panel = panel.merge(
        alpha.drop(columns="datetime"), on=["date", "symbol"], how="left", validate="one_to_one"
    )
    names = meta["features"]
    panel["alpha_feature_fraction"] = panel[names].notna().mean(axis=1)
    full = (
        names
        + FUNDAMENTALS
        + MOMENTUM
        + ["log_size", "board_star", "board_chinext"]
        + list(context.columns)
    )
    for col in full:
        panel[col] = (
            pd.to_numeric(panel[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
    assert_keys(panel)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel.to_parquet(out / "core-panel.parquet", index=False)
    save(
        out / "core-results.json",
        {
            "status": "READY_FOR_ROLLING_TRAINING",
            "rows": len(panel),
            "dates": panel.date.nunique(),
            "first": panel.date.min(),
            "last": panel.date.max(),
            "panel_sha256": sha256(out / "core-panel.parquet"),
            "config_sha256": sha256(CONFIG),
            "feature_groups": {"full": full, "financial": full + cfg["financial_features"]},
            "core_financial_sha256": sha256(data / "core-financials.parquet"),
            "production_modified": False,
        },
    )
    print("CORE_PANEL_COMPLETE", flush=True)


def build():
    data = BASE / "data-v1"
    out = BASE / "panel-v1"
    cfg = json.loads(CONFIG.read_bytes())
    if (out / "results.json").exists():
        print("PANEL_ALREADY_COMPLETE", flush=True)
        return
    meta = json.loads((out / "core-results.json").read_bytes())
    if meta["panel_sha256"] != sha256(out / "core-panel.parquet") or meta[
        "config_sha256"
    ] != sha256(CONFIG):
        raise ValueError("FROZEN_CORE_PANEL_CHANGED")
    panel = pd.read_parquet(out / "core-panel.parquet")
    full = meta["feature_groups"]["full"]
    save(out / "progress.json", {"phase": "FINANCIAL_PUBLICATION_EVENTS", "pid": os.getpid()})
    event_path = out / "financial-events.parquet"
    if not event_path.exists():
        events = financial_events(
            *(
                pd.read_parquet(data / f"{e}.parquet")
                for e in ["cashflow", "fina_indicator", "income"]
            )
        )
        events.to_parquet(event_path, index=False)
    else:
        events = pd.read_parquet(event_path)
    extra = attach_financial_events(panel, events)
    panel = panel.merge(extra, on=["date", "symbol"], how="left", validate="one_to_one")
    financial_names = full + cfg["financial_features"]
    for col in cfg["financial_features"]:
        panel[col] = (
            pd.to_numeric(panel[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
    assert_keys(panel)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    frozen = pd.read_parquet(out / "core-panel.parquet")
    pd.testing.assert_frame_equal(frozen, panel[frozen.columns], check_exact=True)
    panel.to_parquet(out / "panel.parquet", index=False)
    coverage = panel.groupby(panel.date.str[:4])[FUNDAMENTALS + cfg["financial_features"]].agg(
        lambda s: float(s.notna().mean())
    )
    coverage.to_parquet(out / "feature-coverage.parquet")
    result = {
        "status": "READY_FOR_ROLLING_TRAINING",
        "rows": len(panel),
        "dates": panel.date.nunique(),
        "first": panel.date.min(),
        "last": panel.date.max(),
        "panel_sha256": sha256(out / "panel.parquet"),
        "core_panel_sha256": meta["panel_sha256"],
        "config_sha256": sha256(CONFIG),
        "feature_groups": {"full": full, "financial": financial_names},
        "feature_coverage": coverage.to_dict(),
        "original_vintage_certified": False,
        "production_modified": False,
        "input_hashes": {p.name: sha256(p) for p in data.glob("*.parquet")},
    }
    save(out / "results.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["feature_groups", "input_hashes"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["native", "core", "panel"], required=True)
    a = p.parse_args()
    with threadpool_limits(limits=2):
        {"native": native, "core": build_core, "panel": build}[a.stage]()
