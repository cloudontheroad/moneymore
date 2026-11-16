"""Frozen training normalization and price-scale-safe extended Qlib provider."""
import argparse
import json

import numpy as np
import pandas as pd
from prepare_hist_extension import CONFIG, OUT, SOURCE
from prepare_hist_reproduction import CFG, ROOT, UPSTREAM, digest, write_json


def fit_original_normalizer():
    from prepare_hist_reproduction import init_qlib
    from qlib.contrib.data.handler import Alpha360
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from threadpoolctl import threadpool_limits
    destination = OUT / "normalizer.npz"
    if destination.exists() and (OUT / "normalizer-check.json").exists():
        return
    init_qlib()
    with threadpool_limits(limits=2):
        handler = Alpha360(start_time=CFG["train"][0], end_time=CFG["train"][1],
            fit_start_time=CFG["train"][0], fit_end_time=CFG["train"][1], instruments=CFG["universe"],
            infer_processors=[{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                              {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
            learn_processors=[{"class": "DropnaLabel"}, {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
            label=[CFG["label"]])
        data = DatasetH(handler, {"train": tuple(CFG["train"])})
        frame = data.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_I)
        index = pd.MultiIndex.from_frame(pd.read_parquet(SOURCE / "train/index.parquet"))
        actual = frame.reindex(index).to_numpy(dtype=np.float32)
        expected = np.load(SOURCE / "train/features.npy", mmap_mode="r")
        assert np.array_equal(actual, expected), "Original training normalization parity failed"
        normalizer = handler.infer_processors[0]
        np.savez(destination, mean=normalizer.mean_train, std=normalizer.std_train)
        write_json(OUT / "normalizer-check.json", {"exact_match": True, "rows": len(actual),
            "fit": CFG["train"], "sha256": digest(destination), "source_features_sha256": digest(SOURCE/"train/features.npy")})
        print("NORMALIZER_PARITY", len(actual), "exact", flush=True)


def original_array(provider, code, field, length):
    result = np.full(length, np.nan, dtype=np.float32)
    path = provider / "features" / code.lower() / f"{field}.day.bin"
    if path.exists():
        values = np.fromfile(path, dtype="<f4")
        first = int(values[0])
        n = min(len(values)-1, max(0, length-first))
        result[first:first+n] = values[1:1+n]
    return result


def execution_calendar(observed, scheduled):
    extra = scheduled[scheduled > observed[-1]]
    if extra.empty:
        raise ValueError("One scheduled session after cutoff is required for interval boundaries")
    # Calendar only: no future OHLC, features, predictions or returns.
    return observed.append(extra[:5])


def write_future_calendar():
    target = OUT/"provider/calendars"
    observed = pd.DatetimeIndex(pd.to_datetime((target/"day.txt").read_text().splitlines()))
    official = pd.read_parquet(ROOT/"data/processed/trade_calendar.parquet")
    scheduled = pd.DatetimeIndex(pd.to_datetime(sorted(official.loc[official.is_open.eq(1), "cal_date"].unique())))
    future = execution_calendar(observed, scheduled)
    (target/"day_future.txt").write_text("\n".join(future.strftime("%Y-%m-%d"))+"\n")
    assert future[:len(observed)].equals(observed)
    print("EXECUTION_CALENDAR", "observed_end", str(observed[-1].date()), "next_session", str(future[len(observed)].date()), flush=True)


def provider():
    data = OUT / "data"
    status = json.loads((data / "complete.json").read_text())
    for name, value in status["hashes"].items():
        assert digest(data/name) == value
    old = SOURCE / "provider"
    old_calendar = pd.to_datetime((old / "calendars/day.txt").read_text().splitlines())
    old_calendar = old_calendar[old_calendar <= "2020-12-31"]
    recent = json.loads((data / "calendar.json").read_text())
    calendar = old_calendar.append(pd.DatetimeIndex(pd.to_datetime(recent["evaluation"])))
    assert calendar.is_unique and calendar.is_monotonic_increasing
    nold, length = len(old_calendar), len(calendar)
    target = OUT / "provider"
    for name in ("calendars", "features", "instruments"):
        (target/name).mkdir(parents=True, exist_ok=True)
    (target/"calendars/day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d"))+"\n")
    bars = pd.read_parquet(data / "bars.parquet")
    members = pd.read_parquet(data / "membership.parquet")
    old_index = pd.read_parquet(SOURCE / "test/index.parquet")
    symbols = sorted(set(bars.ts_code) | {c[2:]+"."+c[:2] for c in old_index.instrument})
    fields = ("open", "high", "low", "close", "volume", "factor", "change")
    spans, anchors, disagreements = [], [], []
    for i, symbol in enumerate(symbols):
        code = symbol.split(".")[1]+symbol[:6]
        raw = bars.loc[bars.ts_code.eq(symbol)].copy().set_index("trade_date").sort_index()
        raw.index = pd.to_datetime(raw.index)
        values = {f: np.pad(original_array(old, code, f, nold), (0, length-nold), constant_values=np.nan) for f in fields}
        if not raw.empty:
            overlap = raw.index.intersection(old_calendar)
            candidates = [d for d in overlap if np.isfinite(values["factor"][calendar.get_loc(d)])
                          and raw.loc[d, "adj_factor"] > 0 and raw.loc[d, "close"] > 0]
            if candidates:
                anchor = candidates[-1]
                prior_factor = float(values["factor"][calendar.get_loc(anchor)])
                # Some public-package factors use split-adjusted "raw" units
                # (e.g. 002001's factor differs by 1.2). Price and volume are
                # consistent with actual unadjusted shares. Derive the new
                # share conversion from the observed price, not that factor.
                price_factor = float(values["close"][calendar.get_loc(anchor)]) / float(raw.loc[anchor, "close"])
                scale = price_factor / float(raw.loc[anchor, "adj_factor"])
                volume_ratio = float(values["volume"][calendar.get_loc(anchor)])*price_factor / (float(raw.loc[anchor, "vol"])*100)
                ratios = [float(values["close"][calendar.get_loc(d)]) /
                          (float(raw.loc[d, "close"]*raw.loc[d, "adj_factor"])*scale) for d in candidates]
                max_deviation = float(np.max(np.abs(np.array(ratios)-1.)))
                boundary_deviation = float(np.max(np.abs(np.array(ratios[-5:])-1.)))
                # Older public-package prices can differ from official prices;
                # retain that evidence without rewriting the old experiment.
                # The actual join requires agreement of its last five bars.
                if boundary_deviation > .005 or not np.isclose(volume_ratio, 1., rtol=.005):
                    disagreements.append({"symbol": symbol, "price": boundary_deviation, "volume": volume_ratio})
                anchors.append({"symbol": symbol, "date": str(anchor.date()), "kind": "original_overlap",
                    "source_factor_to_actual_price_factor": prior_factor/price_factor,
                    "volume_ratio": volume_ratio, "max_overlap_price_deviation": max_deviation,
                    "boundary_price_deviation": boundary_deviation})
            else:
                valid = raw.loc[raw.close.gt(0) & raw.adj_factor.gt(0)]
                if valid.empty:
                    continue
                anchor = valid.index[0]
                scale = 1. / float(valid.loc[anchor, "adj_factor"]*valid.loc[anchor, "close"])
                anchors.append({"symbol": symbol, "date": str(anchor.date()), "kind": "new_symbol"})
            derived = pd.DataFrame(index=raw.index)
            adjusted_factor = raw.adj_factor * scale
            for field in ("open", "high", "low", "close"):
                derived[field] = raw[field] * adjusted_factor
            derived["factor"] = adjusted_factor
            derived["volume"] = raw.vol * 100 / adjusted_factor
            derived["change"] = raw.pct_chg / 100.
            derived.loc[raw.vol.le(0) | raw.close.le(0), list(fields)] = np.nan
            # Original instruments are bit-preserved through 2020. New listings
            # need the 2020 warmup but never become historical 2017-2020 members.
            new_dates = derived.index[derived.index > "2020-12-31"] if candidates else derived.index
            new_dates = new_dates.intersection(calendar)
            locs = calendar.get_indexer(new_dates)
            for field in fields:
                values[field][locs] = derived.loc[new_dates, field].to_numpy(dtype=np.float32)
        if not np.isfinite(values["close"]).any():
            continue
        dest = target/"features"/code.lower()
        dest.mkdir(exist_ok=True)
        for field, array in values.items():
            np.r_[np.float32(0), array].astype("<f4").tofile(dest/f"{field}.day.bin")
        # Zero-channel compatibility: intentionally no vwap bin for frozen model.
        finite = np.flatnonzero(np.isfinite(values["close"]))
        spans.append(f"{code}\t{calendar[finite[0]].date()}\t{calendar[finite[-1]].date()}")
        if i % 100 == 0:
            print("PROVIDER", i+1, len(symbols), flush=True)
    # Qlib's reporting benchmark, scale-spliced identically to equities.
    b = pd.read_parquet(OUT/"benchmarks/CSI300_price.parquet").set_index("trade_date")
    b.index = pd.to_datetime(b.index)
    code = "SH000300"
    dest = target/"features"/code.lower()
    dest.mkdir(exist_ok=True)
    previous = original_array(old, code, "close", nold)
    scale = previous[-1] / b.loc[old_calendar[-1], "close"]
    for field in ("close", "open", "high", "low"):
        array = np.pad(original_array(old, code, field, nold), (0, length-nold), constant_values=np.nan)
        array[nold:] = b[field].reindex(calendar[nold:]).to_numpy()*scale
        assert np.isfinite(array[nold:]).all()
        np.r_[np.float32(0), array].astype("<f4").tofile(dest/f"{field}.day.bin")
    spans.append(f"SH000300\t{calendar[0].date()}\t{calendar[-1].date()}")
    (target/"instruments/all.txt").write_text("\n".join(spans)+"\n")
    pd.DataFrame(anchors).to_parquet(data/"splice-anchors.parquet", index=False)
    if disagreements:
        write_json(data/"splice-disagreements.json", disagreements)
        raise ValueError(f"Splice boundary disagreements: {disagreements[:8]}; total {len(disagreements)}")
    coverage = members.merge(bars[["trade_date", "ts_code", "close", "vol"]], how="left", on=["trade_date", "ts_code"], validate="one_to_one")
    coverage["valid_bar"] = coverage.close.gt(0) & coverage.vol.gt(0)
    coverage.groupby("trade_date").valid_bar.sum().to_frame().to_parquet(data/"coverage.parquet")
    write_json(OUT/"provider-complete.json", {"symbols": len(spans)-1, "first": str(calendar[0].date()),
        "last": str(calendar[-1].date()), "min_member_valid_bars": int(coverage.groupby("trade_date").valid_bar.sum().min()),
        "missing_member_bar_rows": int((~coverage.valid_bar).sum()), "original_prices_preserved_to": "20201231"})
    write_future_calendar()


def init_extended():
    import qlib
    qlib.init(provider_uri=str(OUT/"provider"), region="cn", kernels=2, joblib_backend="threading",
              expression_cache=None, dataset_cache=None, logging_level=30)


def features():
    from qlib.contrib.data.loader import Alpha360DL
    from qlib.data import D
    from threadpoolctl import threadpool_limits
    init_extended()
    members = pd.read_parquet(OUT/"data/membership.parquet")
    bars = pd.read_parquet(OUT/"data/bars.parquet")
    members = members.merge(bars[["trade_date", "ts_code", "close", "vol"]], on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    members = members.loc[members.close.gt(0) & members.vol.gt(0)].copy()
    members["instrument"] = members.ts_code.str[7:]+members.ts_code.str[:6]
    members["datetime"] = pd.to_datetime(members.trade_date)
    index = pd.MultiIndex.from_frame(members[["datetime", "instrument"]]).sort_values()
    expressions, names = Alpha360DL.get_feature_config()
    normal = np.load(OUT/"normalizer.npz")
    mapping = np.load(UPSTREAM/"data/csi300_stock_index.npy", allow_pickle=True).item()
    market = pd.read_parquet(OUT/"data/market_value.parquet")
    market["instrument"] = market.ts_code.str[7:]+market.ts_code.str[:6]
    market["datetime"] = pd.to_datetime(market.trade_date)
    market = market.set_index(["datetime", "instrument"]).total_mv * 1e4 / 1e9
    for year in range(2021, pd.Timestamp(CONFIG["requested_end"]).year+1):
        dest = OUT/"features"/str(year)
        if (dest/"complete.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        wanted = index[index.get_level_values("datetime").year == year]
        with threadpool_limits(limits=2):
            frame = D.features(sorted(wanted.get_level_values("instrument").unique()), expressions,
                               start_time=wanted.get_level_values("datetime").min(), end_time=wanted.get_level_values("datetime").max())
        frame = frame.swaplevel().sort_index().reindex(wanted)
        frame.columns = names
        x = frame.to_numpy(dtype=np.float32)
        x = np.clip((x-normal["mean"])/normal["std"], -3, 3)
        x[np.isnan(x)] = 0.
        assert np.isfinite(x).all() and np.all(x[:, 240:300] == 0)
        mv = market.reindex(wanted).to_numpy(dtype=np.float32)
        if not (np.isfinite(mv).all() and (mv > 0).all()):
            missing = wanted[~np.isfinite(mv) | (mv <= 0)].to_frame(index=False)
            missing.to_parquet(dest/"missing-market-value.parquet", index=False)
            raise ValueError("Missing actual market value for observed members")
        sid = pd.Series(wanted.get_level_values("instrument")).map(mapping).fillna(733).to_numpy(dtype=np.int64)
        rows = wanted.to_frame(index=False)
        for name, values in (("features", x), ("market_value", mv), ("stock_index", sid)):
            np.save(dest/(name+".npy"), values)
        rows.to_parquet(dest/"index.parquet", index=False)
        np.save(dest/"counts.npy", rows.groupby("datetime", sort=False).size().to_numpy(dtype=np.int64))
        detail = {"year": year, "rows": len(x), "days": rows.datetime.nunique(),
                  "unknown_concept_rows": int((sid == 733).sum()), "future_label_filtering": False,
                  "vwaps_all_zero_as_original_model": True}
        write_json(dest/"complete.json", detail)
        print("FEATURES", detail, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "provider", "features", "calendar"])
    {"fit": fit_original_normalizer, "provider": provider, "features": features,
     "calendar": write_future_calendar}[parser.parse_args().phase]()
