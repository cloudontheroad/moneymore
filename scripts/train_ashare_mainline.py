"""N2: fixed, purged chronological experiments using official Qlib DoubleEnsemble."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import pickle
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from moneymore.research import ashare_mainline as core

ROOT = Path(__file__).resolve().parents[1]


def main(inputs, output, requested_folds):
    config_path = ROOT / "configs/ashare_mainline_v1.json"
    config = json.loads(config_path.read_bytes())
    if os.environ.get("PYTHONHASHSEED") != str(config["seed"]):
        raise ValueError("START_WITH_PYTHONHASHSEED_42_FOR_NATIVE_SET_ORDER")
    if output.exists():
        raise FileExistsError("IMMUTABLE_N2_OUTPUT")
    manifest = json.loads((inputs / "results.json").read_bytes())
    if core.sha256(inputs / "panel.parquet") != manifest["panel_sha256"]:
        raise ValueError("N1_PANEL_HASH_CHANGED")
    input_hashes = {
        key.replace("\\", "/"): value for key, value in manifest["input_hashes"].items()
    }
    if core.sha256(config_path) != input_hashes["configs/ashare_mainline_v1.json"]:
        raise ValueError("CONFIG_CHANGED_AFTER_N1")
    folds = requested_folds or config["folds"]
    if len(set(folds)) != len(folds) or set(folds) - set(config["folds"]):
        raise ValueError("UNREGISTERED_OR_DUPLICATE_FOLD")
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(config_path.read_bytes())
    started = time.monotonic()
    completed = []

    def progress(phase, **extra):
        record = {
            "phase": phase,
            "process_id": os.getpid(),
            "completed_cases": len(completed),
            "total_cases": len(folds) * len(config["cases"]),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            **extra,
        }
        (output / "progress.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)

    try:
        progress("LOADING")
        panel = pd.read_parquet(inputs / "panel.parquet")
        core.assert_keys(panel)
        from qlib.contrib.model.double_ensemble import DEnsembleModel

        runtime = {
            name: importlib.metadata.version(name)
            for name in ("pyqlib", "lightgbm", "pandas", "scikit-learn", "numpy")
        }
        native_source = Path(inspect.getsourcefile(DEnsembleModel))
        source_snapshot = output / "source-snapshot"
        source_snapshot.mkdir()
        for source_path in (Path(__file__), Path(inspect.getsourcefile(core)), native_source):
            shutil.copy2(source_path, source_snapshot / source_path.name)
        (output / "runtime.json").write_text(
            json.dumps(
                {
                    "versions": runtime,
                    "native_class": "qlib.contrib.model.double_ensemble.DEnsembleModel",
                    "native_source_sha256": core.sha256(native_source),
                    "adapter_sha256": core.sha256(Path(inspect.getsourcefile(core))),
                    "runner_sha256": core.sha256(Path(__file__)),
                    "panel_sha256": manifest["panel_sha256"],
                    "pythonhashseed": os.environ["PYTHONHASHSEED"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for fold in folds:
            split = core.purged_split(
                panel, fold, config["validation_sessions"], config["minimum_train_sessions"]
            )
            alpha_ok = panel.alpha_feature_fraction.ge(config["minimum_alpha_fraction"])
            parts = {}
            for segment, mask in split.items():
                mask = mask & alpha_ok
                if segment != "test":
                    mask &= np.isfinite(panel.label_20)
                parts[segment] = panel.loc[mask].copy().reset_index(drop=True)
                if parts[segment].empty:
                    raise ValueError("EMPTY_PARTITION:" + segment)
            if parts["train"].date.nunique() < config["minimum_train_sessions"]:
                raise ValueError("INSUFFICIENT_TRAINING_AFTER_FEATURE_AND_LABEL_FILTERS")
            time_contract = {
                name: {
                    "rows": len(f),
                    "dates": f.date.nunique(),
                    "first": f.date.min(),
                    "last": f.date.max(),
                    "last_label_end": f.label_end_20.dropna().max(),
                }
                for name, f in parts.items()
            }
            time_contract["excluded_feature_rows"] = {
                name: int((mask & ~alpha_ok).sum()) for name, mask in split.items()
            }
            time_contract["excluded_missing_learning_labels"] = {
                name: int((split[name] & alpha_ok & ~np.isfinite(panel.label_20)).sum())
                for name in ("train", "valid")
            }
            assert parts["train"].label_end_20.max() < parts["valid"].date.min()
            assert parts["valid"].label_end_20.max() < parts["test"].date.min()
            target = {name: core.rank_target(parts[name]) for name in ("train", "valid")}
            fold_dir = output / fold
            fold_dir.mkdir()
            (fold_dir / "time-contract.json").write_text(
                json.dumps(time_contract, indent=2), encoding="utf-8"
            )
            for name, spec in config["cases"].items():
                progress("FITTING", fold=fold, case=name)
                case_start = time.monotonic()
                dest = fold_dir / name
                dest.mkdir()
                feature_names = manifest["feature_groups"][spec["features"]]
                processor = core.fit_processor(
                    parts["train"], feature_names, categorical=spec["features"] == "full"
                )
                xs = {segment: core.transform(processor, frame) for segment, frame in parts.items()}
                np.random.seed(config["seed"])
                if spec["model"] == "ridge":
                    model = Ridge(alpha=1.0)
                    # Correlated Alpha158 columns require double precision for the solve.
                    model.fit(xs["train"].astype("float64"), target["train"].astype("float64"))
                else:
                    model = DEnsembleModel(
                        **config["doubleensemble"],
                        num_threads=config["threads"],
                        seed=config["seed"],
                    )
                    dataset = core.NativeDataset(
                        {
                            segment: core.learning_frame(xs[segment], target[segment])
                            for segment in ("train", "valid")
                        }
                    )
                    model.fit(dataset)
                progress("PREDICTING_AND_RELOADING", fold=fold, case=name)
                score = core.predict(model, spec["model"], xs["test"])
                if not np.isfinite(score).all():
                    raise ValueError("NONFINITE_PREDICTIONS")
                artifact = {
                    "model": model,
                    "kind": spec["model"],
                    "processor": processor,
                    "feature_names": feature_names,
                    "target_unit": "cross_sectional_rank_not_return",
                    "panel_sha256": manifest["panel_sha256"],
                    "time_contract": time_contract,
                    "production_enabled": False,
                }
                with (dest / "model.pkl").open("wb") as stream:
                    pickle.dump(artifact, stream)
                with (dest / "model.pkl").open("rb") as stream:
                    reloaded = pickle.load(stream)
                second = core.predict(
                    reloaded["model"],
                    reloaded["kind"],
                    core.transform(reloaded["processor"], parts["test"]),
                )
                error = float(np.abs(score - second).max())
                if error > 1e-10:
                    raise ValueError("RELOAD_PREDICTION_CHANGED")
                predictions = parts["test"][
                    ["date", "symbol", "industry", "label_20", "label_end_20"]
                ].copy()
                predictions["score"] = score
                predictions.to_parquet(dest / "predictions.parquet", index=False)
                end = pd.Period(fold, freq="Q").end_time.strftime("%Y%m%d")
                daily = core.prediction_diagnostics(predictions, end)
                daily.to_parquet(dest / "daily-diagnostics.parquet", index=False)
                if spec["model"] == "ridge":
                    importance = pd.Series(
                        model.coef_, index=xs["train"].columns, name="coefficient"
                    )
                else:
                    importance = model.get_feature_importance(importance_type="gain").rename("gain")
                importance.rename_axis("feature").to_frame().to_parquet(
                    dest / "feature-importance.parquet"
                )
                minimum_unique = int(daily.distinct_scores.min())
                status = (
                    "DEGENERATE_RESEARCH_RESULT"
                    if minimum_unique < 10
                    else "PREDICTION_PIPELINE_VERIFIED_NOT_RETURN_ACCEPTANCE"
                )
                result = {
                    "case": name,
                    "fold": fold,
                    "status": status,
                    "training_rows": len(parts["train"]),
                    "validation_rows": len(parts["valid"]),
                    "predicted_rows": len(predictions),
                    "test_days": len(daily),
                    "mature_ic_days": int(daily.rank_ic.notna().sum()),
                    "mean_rank_ic": float(daily.rank_ic.mean())
                    if daily.rank_ic.notna().any()
                    else None,
                    "minimum_distinct_daily_scores": minimum_unique,
                    "mean_top50_chinext_star_fraction": float(
                        daily.top50_chinext_star_fraction.mean()
                    ),
                    "mean_top50_largest_industry_fraction": float(
                        daily.top50_largest_industry_fraction.mean()
                    ),
                    "raw_feature_count": len(feature_names),
                    "effective_numeric_features": len(processor["active"]),
                    "model_input_count_with_missing_and_industry": len(xs["train"].columns),
                    "dropped_training_features": processor["dropped_training_constant_or_missing"],
                    "submodel_trees": [m.num_trees() for m in model.ensemble]
                    if spec["model"] == "doubleensemble"
                    else None,
                    "reload_max_abs_error": error,
                    "elapsed_seconds": time.monotonic() - case_start,
                    "production_modified": False,
                    "net_performance_measured": False,
                    "hashes": {p.name: core.sha256(p) for p in dest.iterdir() if p.is_file()},
                }
                (dest / "results.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                completed.append(result)
                progress("CASE_COMPLETE", fold=fold, case=name, mean_rank_ic=result["mean_rank_ic"])
        report = {
            "status": "REQUESTED_FOLDS_COMPLETE_NOT_COSTED_REPLAY",
            "folds": folds,
            "full_registered_matrix_complete": set(folds) == set(config["folds"]),
            "results": completed,
            "historical_period_previously_examined": True,
            "production_modified": False,
            "real_orders_enabled": False,
        }
        (output / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        progress(
            "COMPLETE", full_registered_matrix_complete=report["full_registered_matrix_complete"]
        )
    except Exception as error:
        progress("FAILED", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "state/ashare-mainline-20261112/panel-v1"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", action="append", dest="folds")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.inputs, args.output, args.folds)
