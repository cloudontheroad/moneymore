"""Prepare isolated HIST author inputs. Never imports MoneyMore services/credentials."""
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "state/hist-reproduction-20261113"
UPSTREAM = ROOT / "external_research/HIST-upstream"
CFG = json.loads((ROOT / "configs/hist_reproduction_v1.json").read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def init_qlib():
    import qlib
    qlib.init(provider_uri=str(OUT / "provider"), region="cn", kernels=2,
              joblib_backend="threading", expression_cache=None, dataset_cache=None,
              logging_level=30)


def prepare():
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert commit == CFG["source_commit"]
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--exit-code"], check=True)
    archive = OUT / "downloads" / CFG["dataset_archive"]
    if not archive.exists():
        part = archive.with_suffix(archive.suffix + ".part")
        assert part.stat().st_size == CFG["dataset_bytes"], "Incomplete download"
        with zipfile.ZipFile(part) as zipped:
            assert zipped.testzip() is None, "Corrupt archive"
        part.rename(archive)
    assert archive.stat().st_size == CFG["dataset_bytes"]
    dest = OUT / "provider"
    if not (dest / ".complete.json").exists():
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                path = (dest / entry.filename).resolve()
                if not path.is_relative_to(dest.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe archive member")
            zipped.extractall(dest)
        write_json(dest / ".complete.json", {"archive_sha256": digest(archive)})
    calendar = pd.to_datetime((dest / "calendars/day.txt").read_text().splitlines())
    assert calendar.min() <= pd.Timestamp("2006-09-01")
    assert calendar.max() > pd.Timestamp(CFG["test"][1]), "Full test window and next label required"
    source_hashes = {str(p.relative_to(UPSTREAM)): digest(p) for p in sorted(UPSTREAM.glob("*.py"))}
    source_hashes.update({str(p.relative_to(UPSTREAM)): digest(p) for p in sorted((UPSTREAM / "data").glob("csi300*"))})
    manifest = {"config": CFG, "commit": commit, "source_sha256": source_hashes,
                "archive_sha256": digest(archive), "calendar": [str(calendar.min()), str(calendar.max())],
                "paper_dataset_byte_identity": "not_certified", "splits": {}}
    write_json(OUT / "manifest.json", manifest)
    init_qlib()
    from qlib.contrib.data.handler import Alpha360
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    with threadpool_limits(limits=CFG["cpu_threads"]):
        print("PREPARING author Alpha360 2007-2020; fit only 2007-2014", flush=True)
        handler = Alpha360(start_time=CFG["train"][0], end_time=CFG["test"][1],
            fit_start_time=CFG["train"][0], fit_end_time=CFG["train"][1], instruments=CFG["universe"],
            infer_processors=[{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                              {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
            learn_processors=[{"class": "DropnaLabel"}, {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
            label=[CFG["label"]])
        dataset = DatasetH(handler, {key: tuple(CFG[key]) for key in ("train", "valid", "test")})
        # This is author-published data from the pinned source, not a downloaded model checkpoint.
        market = pd.read_pickle(UPSTREAM / "data/csi300_market_value_07to20.pkl").iloc[:, 0] / 1e9
        stock_index = np.load(UPSTREAM / "data/csi300_stock_index.npy", allow_pickle=True).item()
        concepts = np.load(UPSTREAM / "data/csi300_stock2concept.npy", allow_pickle=False)
        np.save(OUT / "concepts.npy", concepts.astype(np.float32))
        train_mv_mean = None
        for key in ("train", "valid", "test"):
            frame = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L).sort_index()
            assert frame.index.is_unique and frame.index.names == ["datetime", "instrument"]
            x = frame["feature"].to_numpy(dtype=np.float32)
            y = frame["label"].iloc[:, 0].to_numpy(dtype=np.float32)
            mv = market.reindex(frame.index)
            if key == "train":
                train_mv_mean = float(mv.mean())
            mv_missing = int(mv.isna().sum())
            mv = mv.fillna(train_mv_mean).to_numpy(dtype=np.float32)
            mapped = pd.Series(frame.index.get_level_values("instrument")).map(stock_index)
            indexes = mapped.fillna(733).to_numpy(dtype=np.int64)
            assert indexes.max() < len(concepts)
            assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(mv).all()
            split = OUT / key
            split.mkdir(exist_ok=True)
            for name, arr in (("features", x), ("labels", y), ("market_value", mv), ("stock_index", indexes)):
                np.save(split / (name + ".npy"), arr)
            index_frame = frame.index.to_frame(index=False)
            index_frame.to_parquet(split / "index.parquet", index=False)
            counts = index_frame.groupby("datetime", sort=False).size().to_numpy(dtype=np.int64)
            np.save(split / "counts.npy", counts)
            infer = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
            raw = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_R)
            raw_missing = raw["feature"].isna().mean()
            manifest["splits"][key] = {"rows": len(x), "days": len(counts), "features": x.shape[1],
                "first": str(index_frame.datetime.min()), "last": str(index_frame.datetime.max()),
                "stocks": int(index_frame.instrument.nunique()), "stock_index_missing_rows": int(mapped.isna().sum()),
                "market_value_missing_rows": mv_missing, "author_dkl_dropped_rows": len(infer) - len(frame),
                "daily_count_min": int(counts.min()), "daily_count_max": int(counts.max()),
                "raw_feature_missing_mean": float(raw_missing.mean()),
                "raw_vwap_missing_mean": float(raw_missing.loc[raw_missing.index.str.startswith("VWAP")].mean()),
                "entirely_missing_features": raw_missing[raw_missing.eq(1)].index.tolist()}
            # Public Qlib archive has no VWAP: retain author's Fillna behavior,
            # but never certify this as matching the paper's complete six channels.
            missing = manifest["splits"][key]["entirely_missing_features"]
            assert all(name.startswith("VWAP") for name in missing), "Unexpected absent feature channel"
            write_json(OUT / "manifest.json", manifest)
            print(key, manifest["splits"][key], flush=True)
        manifest["train_market_value_fill_mean"] = train_mv_mean
        manifest["preparation_complete"] = True
        write_json(OUT / "manifest.json", manifest)
        write_json(OUT / "prepared.json", {"manifest_sha256": digest(OUT / "manifest.json"), "status": "complete"})


if __name__ == "__main__":
    prepare()
