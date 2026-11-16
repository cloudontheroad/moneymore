"""Frozen quarterly DoubleEnsemble, financial augmentation and official TRA ablation."""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import os
import pickle
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_ashare_three_year import BASE, CONFIG, ROOT
from prepare_four_strategy_2026 import save
from threadpoolctl import threadpool_limits

from moneymore.research import ashare_mainline as core
from moneymore.research.three_year import rolling_split, sequence_indices


@contextmanager
def bounded_lightgbm_prediction(threads):
    """Qlib permutation importance omits prediction thread count; bound this worker only."""
    import lightgbm as lgb

    original = lgb.Booster.predict

    def predict(self, *args, **kwargs):
        kwargs.setdefault("num_threads", threads)
        return original(self, *args, **kwargs)

    lgb.Booster.predict = predict
    try:
        yield
    finally:
        lgb.Booster.predict = original


def main(requested_folds=None, requested_cases=None, smoke=False):
    cfg = json.loads(CONFIG.read_bytes())
    if os.environ.get("PYTHONHASHSEED") != str(cfg["seed"]):
        raise ValueError("PYTHONHASHSEED_MUST_BE_42")
    inputs = BASE / "panel-v1"
    cases = requested_cases or ["rank_cap", "rnn_rank_cap", "tra_rank_cap"]
    augmented = cases == ["financial_rank_cap"]
    if "financial_rank_cap" in cases and not augmented:
        raise ValueError("RUN_FINANCIAL_CASE_SEPARATELY_TO_PRESERVE_FROZEN_CORE_INPUT")
    prefix = "" if augmented else "core-"
    manifest = json.loads((inputs / (prefix + "results.json")).read_bytes())
    if (
        core.sha256(inputs / (prefix + "panel.parquet")) != manifest["panel_sha256"]
        or core.sha256(CONFIG) != manifest["config_sha256"]
    ):
        raise ValueError("FROZEN_INPUT_CHANGED")
    panel = pd.read_parquet(inputs / (prefix + "panel.parquet"))
    calendar = json.loads((BASE / "data-v1/calendar.json").read_bytes())["days"]
    start = pd.Timestamp(cfg["evaluation_start"]).to_period("Q")
    end = pd.Timestamp(cfg["evaluation_end"]).to_period("Q")
    folds = [str(p) for p in pd.period_range(start, end, freq="Q")]
    if requested_folds:
        if set(requested_folds) - set(folds):
            raise ValueError("UNREGISTERED_FOLD")
        folds = requested_folds
    if set(cases) - {"rank_cap", "financial_rank_cap", "rnn_rank_cap", "tra_rank_cap"}:
        raise ValueError("UNREGISTERED_MODEL_CASE")
    output = BASE / ("training-smoke-v1" if smoke else "training-v1")
    output.mkdir(exist_ok=True)
    save(output / "protocol.json", cfg)
    source_files = [
        Path(__file__),
        Path(inspect.getsourcefile(core)),
        ROOT / "src/moneymore/research/three_year.py",
        ROOT / "src/moneymore/research/three_year_tra.py",
        CONFIG,
        ROOT / "configs/ashare_mainline_v1.json",
    ]
    hashes = {str(p.relative_to(ROOT)): core.sha256(p) for p in source_files}
    worker_id = "-".join(cases) + "-" + folds[0]

    def progress(phase, **extra):
        row = {"phase": phase, "pid": os.getpid(), "utc": datetime.now(UTC).isoformat(), **extra}
        save(output / ("progress-" + worker_id + ".json"), row)
        print(json.dumps(row), flush=True)

    from qlib.contrib.model.double_ensemble import DEnsembleModel

    native_config = json.loads((ROOT / "configs/ashare_mainline_v1.json").read_bytes())[
        "doubleensemble"
    ]
    for fold in folds:
        split = rolling_split(panel, fold, cfg["validation_sessions"], cfg["training_sessions"])
        good = panel.alpha_feature_fraction.ge(cfg["minimum_alpha_fraction"])
        train_mask = split["train"] & good & np.isfinite(panel.label_20)
        valid_mask = split["valid"] & good & np.isfinite(panel.label_20)
        test_mask = split["test"] & good
        train = panel.loc[train_mask]
        valid = panel.loc[valid_mask]
        test = panel.loc[test_mask]
        if min(train.date.nunique(), valid.date.nunique(), test.date.nunique()) < 1:
            raise ValueError("EMPTY_PARTITION")
        if not (
            train.label_end_20.max() < valid.date.min()
            and valid.label_end_20.max() < test.date.min()
        ):
            raise ValueError("LABEL_BOUNDARY_VIOLATION")
        contract = {
            k: {
                "first": str(f.date.min()),
                "last": str(f.date.max()),
                "last_label_end": str(f.label_end_20.dropna().max()),
                "days": f.date.nunique(),
                "rows": len(f),
            }
            for k, f in {"train": train, "valid": valid, "test": test}.items()
        }
        for case in cases:
            dest = output / fold / case
            result_file = dest / "results.json"
            if result_file.exists():
                previous = json.loads(result_file.read_bytes())
                if (
                    previous["source_hashes"] != hashes
                    or previous["panel_sha256"] != manifest["panel_sha256"]
                    or previous["predictions_sha256"] != core.sha256(dest / "predictions.parquet")
                ):
                    raise ValueError("COMPLETED_TRAINING_ARTIFACT_CHANGED")
                progress("CACHED", fold=fold, case=case)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            save(dest / "time-contract.json", contract)
            progress("FITTING", fold=fold, case=case)
            began = time.monotonic()
            names = manifest["feature_groups"][
                "financial" if case == "financial_rank_cap" else "full"
            ]
            processor = core.fit_processor(train, names, categorical=True)
            np.random.seed(cfg["seed"])
            if case in {"rank_cap", "financial_rank_cap"}:
                parts = {"train": train, "valid": valid, "test": test}
                xs = {k: core.transform(processor, v) for k, v in parts.items()}
                targets = {k: core.rank_target(parts[k]) for k in ["train", "valid"]}
                model = DEnsembleModel(
                    **{**native_config, **({"epochs": 2} if smoke else {})},
                    num_threads=cfg["threads"],
                    seed=cfg["seed"],
                )
                model.fit(
                    core.NativeDataset(
                        {k: core.learning_frame(xs[k], targets[k]) for k in ["train", "valid"]}
                    )
                )
                scores = core.predict(model, "doubleensemble", xs["test"])
                artifact = {
                    "model": model,
                    "processor": processor,
                    "feature_names": names,
                    "config": cfg,
                    "time_contract": contract,
                }
                with (dest / "model.pkl").open("wb") as stream:
                    pickle.dump(artifact, stream)
                with (dest / "model.pkl").open("rb") as stream:
                    restored = pickle.load(stream)
                second = core.predict(
                    restored["model"], "doubleensemble", core.transform(restored["processor"], test)
                )
                reload_error = float(np.max(np.abs(scores - second)))
                output_frame = test[
                    ["date", "symbol", "industry", "label_20", "label_end_20"]
                ].copy()
                output_frame["score"] = scores
                importance = model.get_feature_importance(importance_type="gain").rename("gain")
                importance.rename_axis("feature").to_frame().to_parquet(
                    dest / "feature-importance.parquet"
                )
                del xs, model, artifact, restored
            else:
                import torch
                from qlib.contrib.model.pytorch_tra import TRAModel

                from moneymore.research.three_year_tra import (
                    CausalSequenceBatches,
                    predict_without_labels,
                    routing_diagnostics,
                )

                torch.set_num_threads(cfg["threads"])
                torch.manual_seed(cfg["seed"])
                device = "cuda" if torch.cuda.is_available() else "cpu"
                spec = cfg["tra"]
                first_index = max(0, calendar.index(train.date.min()) - spec["sequence_length"])
                frame = (
                    panel.loc[panel.date.between(calendar[first_index], test.date.max())]
                    .copy()
                    .reset_index(drop=True)
                )
                x = core.transform(processor, frame).to_numpy(dtype="float32")
                x = np.concatenate([np.zeros((1, x.shape[1]), dtype="float32"), x])
                sequences = sequence_indices(frame, calendar, spec["sequence_length"])
                mask_train = (
                    frame.date.isin(train.date.unique())
                    & frame.alpha_feature_fraction.ge(cfg["minimum_alpha_fraction"])
                    & frame.label_20.notna()
                )
                mask_valid = (
                    frame.date.isin(valid.date.unique())
                    & frame.alpha_feature_fraction.ge(cfg["minimum_alpha_fraction"])
                    & frame.label_20.notna()
                )
                mask_test = frame.date.isin(test.date.unique()) & frame.alpha_feature_fraction.ge(
                    cfg["minimum_alpha_fraction"]
                )
                target = pd.Series(np.nan, index=frame.index)
                # Same eligible cross-sectional ranks as the DoubleEnsemble cases.
                # Labels outside learning/validation are never exposed to Qlib.
                learn_mask = mask_train | mask_valid
                target.loc[learn_mask] = core.rank_target(frame.loc[learn_mask])
                states = spec["num_states"] if case == "tra_rank_cap" else 1
                datasets = {
                    k: CausalSequenceBatches(
                        x, sequences, frame, m, target, spec["batch_size"], states, device
                    )
                    for k, m in {
                        "train": mask_train,
                        "valid": mask_valid,
                        "test": mask_test,
                    }.items()
                }
                model = TRAModel(
                    model_config={
                        "input_size": x.shape[1],
                        "hidden_size": spec["hidden_size"],
                        "num_layers": spec["num_layers"],
                        "dropout": spec["dropout"],
                    },
                    tra_config={"num_states": states, "src_info": "LR"},
                    model_type="RNN",
                    lr=spec["lr"],
                    n_epochs=spec["epochs"],
                    early_stop=spec["early_stop"],
                    lamb=spec["lamb"],
                    rho=spec["rho"],
                    alpha=1.0,
                    seed=cfg["seed"],
                    transport_method="router" if states > 1 else "none",
                    eval_train=False,
                    eval_test=False,
                )
                best = -np.inf
                best_params = None
                stale = 0
                epochs = []
                for epoch in range(1 if smoke else spec["epochs"]):
                    loss = model.train_epoch(epoch, datasets["train"])
                    validation = predict_without_labels(model, datasets["valid"])
                    check = frame.loc[validation.index, ["date", "label_20"]].copy()
                    check["score"] = validation
                    ic = float(
                        check.groupby("date")
                        .apply(
                            lambda g: g.score.rank().corr(g.label_20.rank()), include_groups=False
                        )
                        .mean()
                    )
                    if not np.isfinite(loss) or not np.isfinite(ic):
                        raise ValueError("NONFINITE_TRAINING_OR_VALIDATION")
                    if ic > best:
                        best = ic
                        best_params = {
                            "model": copy.deepcopy(model.model.state_dict()),
                            "tra": copy.deepcopy(model.tra.state_dict()),
                        }
                        stale = 0
                    else:
                        stale += 1
                    epochs.append(
                        {"epoch": epoch, "loss": loss, "validation_rank_ic": ic, "best": best}
                    )
                    save(dest / "epochs.json", epochs)
                    progress(
                        "EPOCH_COMPLETE",
                        fold=fold,
                        case=case,
                        epoch=epoch,
                        maximum_epochs=spec["epochs"],
                        validation_rank_ic=ic,
                    )
                    if stale >= spec["early_stop"]:
                        break
                model.model.load_state_dict(best_params["model"])
                model.tra.load_state_dict(best_params["tra"])
                prediction = predict_without_labels(model, datasets["test"])
                save(
                    dest / "routing-diagnostics.json", routing_diagnostics(model, datasets["test"])
                )
                torch.save(
                    {
                        "weights": best_params,
                        "config": cfg,
                        "processor": processor,
                        "features": names,
                        "time_contract": contract,
                    },
                    dest / "model.pt",
                )
                loaded = torch.load(dest / "model.pt", map_location=device, weights_only=False)
                model.model.load_state_dict(loaded["weights"]["model"])
                model.tra.load_state_dict(loaded["weights"]["tra"])
                # Arbitrary test labels cannot alter predictions, and reload is verified.
                datasets["test"].labels = np.full(len(frame), 12345.0, dtype="float32")
                second = predict_without_labels(model, datasets["test"])
                reload_error = float(np.max(np.abs(prediction - second)))
                output_frame = frame.loc[
                    prediction.index, ["date", "symbol", "industry", "label_20", "label_end_20"]
                ].copy()
                output_frame["score"] = prediction
                save(
                    dest / "native-runtime.json",
                    {
                        "torch": torch.__version__,
                        "device": device,
                        "native_source_sha256": core.sha256(Path(inspect.getsourcefile(TRAModel))),
                        "router": "LR",
                        "error_memory_used": False,
                        "test_label_invariance_error": reload_error,
                        "epochs": len(epochs),
                    },
                )
                del datasets, frame, x, sequences, model, loaded, best_params
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if reload_error > 1e-6 or not np.isfinite(output_frame.score).all():
                raise ValueError("PREDICTION_RELOAD_NOT_VERIFIED")
            output_frame["case"] = case
            output_frame.to_parquet(dest / "predictions.parquet", index=False)
            diag = core.prediction_diagnostics(output_frame, cfg["evaluation_end"])
            diag.to_parquet(dest / "daily-diagnostics.parquet", index=False)
            result = {
                "status": "COMPLETE_NOT_RETURN_ACCEPTANCE",
                "fold": fold,
                "case": case,
                "smoke": smoke,
                "rows": len(output_frame),
                "test_days": output_frame.date.nunique(),
                "reload_error": reload_error,
                "minimum_distinct_scores": int(diag.distinct_scores.min()),
                "elapsed_seconds": time.monotonic() - began,
                "panel_sha256": manifest["panel_sha256"],
                "source_hashes": hashes,
                "predictions_sha256": core.sha256(dest / "predictions.parquet"),
                "production_modified": False,
            }
            save(result_file, result)
            progress("CASE_COMPLETE", **{k: result[k] for k in ["fold", "case", "elapsed_seconds"]})
            gc.collect()
    save(
        output / ("completed-" + "-".join(cases) + ".json"),
        {"status": "COMPLETE", "folds": folds, "cases": cases, "smoke": smoke},
    )
    progress("REQUESTED_MATRIX_COMPLETE", folds=folds, cases=cases)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", nargs="+")
    p.add_argument("--cases", nargs="+")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    try:
        with threadpool_limits(limits=2), bounded_lightgbm_prediction(2):
            main(a.folds, a.cases, a.smoke)
    except Exception as exc:
        save(
            BASE / "training-failure.json",
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pid": os.getpid(),
            },
        )
        raise
