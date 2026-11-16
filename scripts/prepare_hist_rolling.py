"""Causal annual HIST folds. Old studies, source weights and services are read-only."""
import argparse
import json
import shutil
import warnings

import numpy as np
import pandas as pd
from build_hist_extension import init_extended, original_array
from prepare_hist_extension import OUT as EXT
from prepare_hist_reproduction import CFG as AUTHOR
from prepare_hist_reproduction import OUT as ORIGINAL
from prepare_hist_reproduction import ROOT, UPSTREAM, digest, init_qlib, write_json

CONFIG = json.loads((ROOT/"configs/hist_rolling_20261114.json").read_text())
OUT = ROOT/CONFIG["output"]


def frozen_inputs():
    paths = [ROOT/name for name in (
        "configs/hist_rolling_20261114.json", "configs/hist_reproduction_v1.json",
        "configs/hist_reproduction_execution_v1.json", "scripts/prepare_hist_rolling.py",
        "scripts/run_hist_rolling.py", "scripts/train_hist_reproduction.py",
        "scripts/replay_hist_reproduction.py", "scripts/hist_reproduction_exchange.py",
        "scripts/audit_hist_reproduction_replay.py", "scripts/finalize_hist_extension.py")]
    paths += [UPSTREAM/"model.py", UPSTREAM/"utils.py", ORIGINAL/"manifest.json", ORIGINAL/"concepts.npy",
              EXT/"data/complete.json", EXT/"provider-complete.json", EXT/"normalizer.npz"]
    paths += [UPSTREAM/"data"/name for name in ("csi300_market_value_07to20.pkl", "csi300_stock_index.npy")]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def fold_schedule(year, calendar):
    start, end = pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(CONFIG["cutoff"]))
    execution = calendar[(calendar >= start) & (calendar <= end)]
    if execution.empty:
        raise ValueError("No execution sessions")
    offsets = calendar.get_indexer(execution)-1
    if offsets.min() < 0:
        raise ValueError("Missing prior signal session")
    signal_start, signal_end = calendar[offsets[0]], calendar[offsets[-1]]
    valid_start = pd.Timestamp(year=year-1, month=10, day=1)
    train_end = calendar[calendar < valid_start][-1]
    train_start = valid_start-pd.DateOffset(years=CONFIG["training_years"])
    return {"year": year, "train": [str(train_start.date()), str(train_end.date())],
            "valid": [str(valid_start.date()), str(signal_start.date())],
            "test": [str(signal_start.date()), str(signal_end.date())],
            "execution": [str(execution[0].date()), str(execution[-1].date())],
            "train_label_cutoff": str(train_end.date()), "valid_label_cutoff": str(signal_start.date())}


def learning_mask(index, labels, label_dates, start, end):
    dates = index.get_level_values("datetime")
    return ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)) &
            (label_dates <= np.datetime64(end)) & np.isfinite(labels))


def fit_robust(x):
    from qlib.data.dataset.processor import EPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # intentionally absent VWAP
        mean = np.nanmedian(x, axis=0)
        std = (np.nanmedian(np.abs(x-mean), axis=0)+EPS)*1.4826
    return mean, std


def normalize(x, mean, std):
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.clip((x-mean)/std, -3., 3.)
    result[np.isnan(result)] = 0.
    if not np.isfinite(result).all():
        raise ValueError("Infinite features")
    return result.astype(np.float32)


def rank_labels(labels, index):
    return ((pd.Series(labels, index=index).groupby(level="datetime").rank(pct=True)-.5)*3.46).to_numpy(dtype=np.float32)


def raw_alpha360(provider, index):
    """Vectorized exact Alpha360DL expressions over unchanged Qlib float32 bins."""
    calendar = pd.DatetimeIndex(pd.to_datetime((provider/"calendars/day.txt").read_text().splitlines()))
    dates = index.get_level_values("datetime")
    offsets = calendar.get_indexer(dates)
    if (offsets < 59).any():
        raise ValueError("Missing 60-session feature warmup")
    groups = pd.Series(np.arange(len(index)), index=index.get_level_values("instrument")).groupby(level=0, sort=False)
    output = np.full((len(index), 360), np.nan, dtype=np.float32)
    for code, rows in groups:
        locs = rows.to_numpy()
        at = offsets[locs]
        close = original_array(provider, code, "close", len(calendar))
        for j, field in enumerate(("close", "open", "high", "low", "vwap", "volume")):
            values = close if field == "close" else original_array(provider, code, field, len(calendar))
            denominator = values[at]+np.float32(1e-12) if field == "volume" else close[at]
            with np.errstate(invalid="ignore", divide="ignore"):
                output[locs, j*60:(j+1)*60] = values[at[:, None]-np.arange(59, -1, -1)]/denominator[:, None]
    return output


def raw_year(year, fingerprint):
    from qlib.contrib.data.loader import Alpha360DL
    from qlib.data import D
    from threadpoolctl import threadpool_limits
    dest = OUT/"raw"/str(year)
    if (dest/"complete.json").exists():
        assert json.loads((dest/"complete.json").read_text())["fingerprint"] == fingerprint
        return
    dest.mkdir(parents=True, exist_ok=True)
    expressions, names = Alpha360DL.get_feature_config()
    start, end = f"{year}-01-01", min(f"{year}-12-31", CONFIG["cutoff"])
    if year <= 2020:
        init_qlib()
        instruments = D.instruments("csi300")
        with threadpool_limits(limits=2):
            frame = D.features(instruments, [AUTHOR["label"], "$close", "$volume"], start_time=start, end_time=end)
        frame = frame.swaplevel().sort_index()
        # Observable inference membership, never future-label DropnaLabel.
        frame = frame.loc[frame.iloc[:, -2].gt(0) & frame.iloc[:, -1].gt(0)]
        market = pd.read_pickle(UPSTREAM/"data/csi300_market_value_07to20.pkl").iloc[:, 0]/1e9
    else:
        init_extended()
        member = pd.read_parquet(EXT/"data/membership.parquet")
        dates = pd.to_datetime(member.trade_date)
        member = member.loc[dates.dt.year.eq(year)].copy()
        member["datetime"] = pd.to_datetime(member.trade_date)
        member["instrument"] = member.ts_code.str[7:]+member.ts_code.str[:6]
        wanted = pd.MultiIndex.from_frame(member[["datetime", "instrument"]]).sort_values()
        with threadpool_limits(limits=2):
            frame = D.features(sorted(member.instrument.unique()), [AUTHOR["label"], "$close", "$volume"],
                               start_time=start, end_time=end)
        frame = frame.swaplevel().sort_index().reindex(wanted)
        frame = frame.loc[frame.iloc[:, -2].gt(0) & frame.iloc[:, -1].gt(0)]
        market = pd.read_parquet(EXT/"data/market_value.parquet")
        market["datetime"] = pd.to_datetime(market.trade_date)
        market["instrument"] = market.ts_code.str[7:]+market.ts_code.str[:6]
        market = market.set_index(["datetime", "instrument"]).total_mv*1e4/1e9
    provider = (ORIGINAL if year <= 2020 else EXT)/"provider"
    x = raw_alpha360(provider, frame.index)
    # Validate all 360 expressions for two securities across 3 complete sessions.
    # This is an equality test, not a statistical approximation of feature values.
    sample_dates = frame.index.get_level_values("datetime").unique()[:3]
    sample_codes = frame.index.get_level_values("instrument").unique()[:2]
    sample_mask = frame.index.get_level_values("datetime").isin(sample_dates) & frame.index.get_level_values("instrument").isin(sample_codes)
    reference = D.features(list(sample_codes), expressions, start_time=sample_dates.min(), end_time=sample_dates.max()).swaplevel().reindex(frame.index[sample_mask])
    np.testing.assert_array_equal(x[sample_mask], reference.to_numpy(dtype=np.float32))
    label = frame.iloc[:, 0].to_numpy(dtype=np.float32)
    if year == 2020:
        # Public/official price splice uses official next close on the boundary.
        last = frame.index.get_level_values("datetime") == pd.Timestamp("2020-12-31")
        if last.any():
            init_extended()
            boundary = D.features(list(frame.index[last].get_level_values("instrument")), [AUTHOR["label"]],
                                  start_time="2020-12-31", end_time="2020-12-31").swaplevel()
            label[last] = boundary.reindex(frame.index[last]).iloc[:, 0].to_numpy(dtype=np.float32)
    mapping = np.load(UPSTREAM/"data/csi300_stock_index.npy", allow_pickle=True).item()
    sid = pd.Series(frame.index.get_level_values("instrument")).map(mapping).fillna(733).to_numpy(dtype=np.int64)
    mv = market.reindex(frame.index).to_numpy(dtype=np.float32)
    mv[mv <= 0] = np.nan
    if year > 2020 and not np.isfinite(mv).all():
        raise ValueError("Actual post-2020 market value missing")
    calendar = pd.DatetimeIndex(pd.to_datetime((EXT/"provider/calendars/day_future.txt").read_text().splitlines()))
    date_offsets = calendar.get_indexer(frame.index.get_level_values("datetime"))
    assert (date_offsets >= 0).all()
    label_dates = calendar[date_offsets+1].to_numpy(dtype="datetime64[ns]")
    assert np.isnan(x[:, 240:300]).all()
    for name, values in (("features", x), ("labels", label), ("label_dates", label_dates), ("market_value", mv), ("stock_index", sid)):
        np.save(dest/f"{name}.npy", values)
    frame.index.to_frame(index=False).to_parquet(dest/"index.parquet", index=False)
    write_json(dest/"complete.json", {"fingerprint": fingerprint, "year": year, "rows": len(frame),
        "days": frame.index.get_level_values("datetime").nunique(), "market_value_missing": int(np.isnan(mv).sum()),
        "unknown_concept_rows": int((sid == 733).sum()), "feature_names": names, "native_expression_exact_parity_rows": int(sample_mask.sum()),
        "hashes": {p.name: digest(p) for p in dest.glob("*.npy")}})
    print("RAW", year, len(frame), flush=True)


def prepare_fold(year, fingerprint):
    folder = OUT/"folds"/str(year)
    if (folder/"prepared.json").exists():
        assert json.loads((folder/"manifest.json").read_text())["input_fingerprint"] == fingerprint
        return
    calendar = pd.DatetimeIndex(pd.to_datetime((EXT/"provider/calendars/day.txt").read_text().splitlines()))
    plan = fold_schedule(year, calendar)
    years = range(pd.Timestamp(plan["train"][0]).year, year+1)
    indexes = [pd.MultiIndex.from_frame(pd.read_parquet(OUT/"raw"/str(y)/"index.parquet")) for y in years]
    index = indexes[0].append(indexes[1:])
    assert index.is_unique and index.is_monotonic_increasing
    names = ["features", "labels", "label_dates", "market_value", "stock_index"]
    arrays = {name: np.concatenate([np.load(OUT/"raw"/str(y)/f"{name}.npy") for y in years]) for name in names}
    dates = index.get_level_values("datetime")
    masks = {k: learning_mask(index, arrays["labels"], arrays["label_dates"], *plan[k]) for k in ("train", "valid")}
    # A validation label can mature on the first signal date, but that date's own label cannot.
    masks["test"] = (dates >= plan["test"][0]) & (dates <= plan["test"][1])
    assert not (masks["train"] & masks["valid"]).any() and not (masks["valid"] & masks["test"]).any()
    fitting = (dates >= plan["train"][0]) & (dates <= plan["train"][1])
    mean, std = fit_robust(arrays["features"][fitting])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder/"normalizer.npz", mean=mean, std=std)
    frozen = np.load(EXT/"normalizer.npz")
    mv_fill = float(np.nanmean(arrays["market_value"][masks["train"]]))
    assert np.isfinite(mv_fill) and mv_fill > 0
    manifest = {"config": {**AUTHOR, **{k: plan[k] for k in ("train", "valid", "test")}},
        "source_sha256": {name: digest(UPSTREAM/name) for name in ("model.py", "utils.py")},
        "input_fingerprint": fingerprint, "plan": plan, "normalizer_fit": plan["train"],
        "market_value_training_fill": mv_fill, "splits": {}, "normalizer_sha256": digest(folder/"normalizer.npz")}
    for split, mask in masks.items():
        dest = folder/split
        dest.mkdir(exist_ok=True)
        wanted = index[mask]
        x = normalize(arrays["features"][mask], mean, std)
        labels = rank_labels(arrays["labels"][mask], wanted)
        mv = arrays["market_value"][mask].copy()
        missing = ~np.isfinite(mv)
        mv[missing] = mv_fill
        values = {"features": x, "labels": labels, "market_value": mv,
                  "stock_index": arrays["stock_index"][mask], "label_dates": arrays["label_dates"][mask]}
        if split == "test":
            values["frozen_features"] = normalize(arrays["features"][mask], frozen["mean"], frozen["std"])
        else:
            assert np.isfinite(labels).all()
            assert pd.Timestamp(values["label_dates"].max()) <= pd.Timestamp(plan[split][1])
        for name, value in values.items():
            np.save(dest/f"{name}.npy", value)
        rows = wanted.to_frame(index=False)
        rows.to_parquet(dest/"index.parquet", index=False)
        np.save(dest/"counts.npy", rows.groupby("datetime", sort=False).size().to_numpy(dtype=np.int64))
        manifest["splits"][split] = {"rows": len(rows), "days": rows.datetime.nunique(),
            "first_signal": str(rows.datetime.min().date()), "last_signal": str(rows.datetime.max().date()),
            "latest_label_realization": str(pd.Timestamp(values["label_dates"].max()).date()),
            "missing_mv_filled": int(missing.sum()), "test_label_filtering": False if split == "test" else None,
            "hashes": {p.name: digest(p) for p in dest.glob("*.npy")}}
    shutil.copy2(ORIGINAL/"concepts.npy", folder/"concepts.npy")
    write_json(folder/"manifest.json", manifest)
    write_json(folder/"prepared.json", {"manifest_sha256": digest(folder/"manifest.json"), "status": "complete"})
    print("PREPARED", year, manifest["splits"], flush=True)


def main():
    from filelock import FileLock
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT/"prepare.lock"), timeout=0):
        fingerprint = frozen_inputs()
        if (OUT/"input-manifest.json").exists():
            assert json.loads((OUT/"input-manifest.json").read_text()) == fingerprint, "Study changed; refuse mixed artifacts"
        else:
            write_json(OUT/"input-manifest.json", fingerprint)
        for year in range(2016, 2027):
            write_json(OUT/"preparation-progress.json", {"status": "raw_features", "year": year})
            raw_year(year, fingerprint)
        for year in CONFIG["years"]:
            write_json(OUT/"preparation-progress.json", {"status": "rolling_folds", "year": year})
            prepare_fold(year, fingerprint)
        write_json(OUT/"preparation-progress.json", {"status": "complete", "folds": len(CONFIG["years"])})


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
