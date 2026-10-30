from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import qlib
import torch
import yaml
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.pytorch_gru import GRU
from qlib.workflow import R

from moneymore.data.store import ParquetStore
from moneymore.portfolio_constructor import adjusted_close_panel, trailing_return_correlation
from moneymore.qlib_challenger import (
    QlibPanelDataset,
    build_challenger_dataset,
    challenger_universe,
    evaluate_predictions,
    metrics_payload,
    rank_blend_predictions,
)
from moneymore.qlib_governance import register_qlib_candidate
from moneymore.training_time_contract import weekly_time_contract
from moneymore.challenger_diagnostics import trading_dates
from moneymore.weekly_market_context import market_context

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "configs" / "qlib_challenger.yaml").read_text(encoding="utf-8")
)
ISOLATED = os.environ.get("MONEYMORE_ISOLATED_RESEARCH", "") == "1"
ARTIFACTS = (ROOT / "state" / "weekly-learning" / "isolated-research") if ISOLATED else ROOT / "state" / "qlib-challenger"
CANDIDATE_TAG = os.environ.get("MONEYMORE_CANDIDATE_TAG", "").strip()
MODELS = (
    ARTIFACTS / "candidates" / CANDIDATE_TAG / "models"
    if CANDIDATE_TAG
    else ARTIFACTS / "models"
)
if CANDIDATE_TAG and (MODELS.parent / "_RESEARCH_COMPLETE").exists():
    raise FileExistsError("completed candidate is frozen; choose a new candidate tag")
MODELS.mkdir(parents=True, exist_ok=True)

store = ParquetStore(ROOT / "data")
universe = challenger_universe(ROOT, store)
price_panel = adjusted_close_panel(store, list(universe))
frame = build_challenger_dataset(
    store,
    universe,
    sequence_length=int(CONFIG["sequence_length"]),
    label_horizon=int(CONFIG["label_horizon"]),
    feature_count=int(CONFIG["model"]["d_feat"]),
)
segments = {
    name: (values["start"], values["end"])
    for name, values in CONFIG.items()
    if name in {"train", "valid", "test", "forward"}
}
time_contract = None
selected_rounds = {}
if CANDIDATE_TAG:
    # The frame already drops immature labels. Removing another horizon from
    # frame dates incorrectly delays training twice. Use exchange sessions.
    # Conservatively use yesterday until a completed-session service contract
    # is wired in; an intraday partial daily bar is never a training close.
    as_of = (pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai"))).normalize() - pd.Timedelta(days=1)).strftime("%Y%m%d")
    raw_dates = store.read("daily", columns=["trade_date"])["trade_date"].astype(str)
    available_end = raw_dates.loc[raw_dates <= as_of].max()
    dates = pd.to_datetime([
        d for d in trading_dates(store, available_end) if d <= available_end
    ])
    horizon = int(CONFIG["label_horizon"])
    time_contract = weekly_time_contract(
        dates, as_of=datetime.now(UTC).date(), horizon=horizon
    )
    context_bars = store.read("daily", columns=["ts_code", "trade_date", "close"],
        filters=[("ts_code", "in", list(universe)), ("trade_date", ">=", "20260101"), ("trade_date", "<=", as_of)])
    market_summary, market_history = market_context(context_bars,
        [d for d in trading_dates(store, as_of) if "20260101" <= d <= as_of],
        universe, as_of=as_of)
    (MODELS.parent / "market-context.json").write_text(
        json.dumps(market_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    market_history.to_parquet(MODELS.parent / "market-history.parquet")
    segments = {
        name: tuple(bounds)
        for name, bounds in time_contract["research_segments"].items()
    }
    segments["forward"] = (
        time_contract["unmatured_features_only"][0], "2099-12-31"
    )
    # Research artifacts remain research-only. The latest-data refit needs its
    # own fitter/manifest and cannot inherit this evaluation model's OOS scores.
    (MODELS.parent / "time-contract.json").write_text(
        json.dumps(time_contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
dataset = QlibPanelDataset(frame, segments)
test_dates = frame.loc[(slice(segments["test"][0], segments["test"][1]), slice(None)), :].index.get_level_values("datetime").unique()
evaluation_correlations = {
    pd.Timestamp(date): trailing_return_correlation(
        price_panel, date, int(CONFIG["portfolio_policy"]["correlation_lookback"])
    )
    for date in test_dates
}
provider_dir = ARTIFACTS / "qlib-provider"
provider_dir.mkdir(parents=True, exist_ok=True)
tracking_db = (ARTIFACTS / "mlflow.db").resolve().as_posix()
qlib.init(
    provider_uri=str(provider_dir),
    region=REG_CN,
    exp_manager={
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": f"sqlite:///{tracking_db}",
            "default_exp_name": "moneymore_qlib_challenger",
        },
    },
)

def build_gru(seed: int) -> GRU:
    return GRU(
        d_feat=int(CONFIG["model"]["d_feat"]),
        hidden_size=int(CONFIG["model"]["hidden_size"]),
        num_layers=int(CONFIG["model"]["num_layers"]),
        dropout=float(CONFIG["model"]["dropout"]),
        n_epochs=int(CONFIG["model"]["n_epochs"]),
        lr=float(CONFIG["model"]["learning_rate"]),
        batch_size=int(CONFIG["model"]["batch_size"]),
        early_stop=int(CONFIG["model"]["early_stop"]),
        GPU=0 if torch.cuda.is_available() else -1,
        seed=seed,
    )


def fit_and_evaluate(model_id: str, model: object) -> tuple[dict[str, object], object]:
    with R.start(experiment_name="moneymore_qlib_challenger"):
        save_path = MODELS / f"{model_id}.bin"
        if isinstance(model, GRU):
            evaluations = {}
            model.fit(dataset, save_path=str(save_path), evals_result=evaluations)
            valid_scores = pd.Series(evaluations.get("valid", []), dtype=float)
            if valid_scores.empty or not valid_scores.notna().any():
                raise RuntimeError("no finite validation scores for epoch selection")
            selected_rounds[model_id] = int(valid_scores.idxmax()) + 1
        else:
            model.fit(dataset)
            selected_rounds[model_id] = int(model.model.best_iteration or model.model.current_iteration())
        predictions = model.predict(dataset, "test")
        labels = dataset.prepare("test", "label").iloc[:, 0]
        pd.concat([predictions.rename("score"), labels.rename("sector_relative_label")], axis=1).to_parquet(
            MODELS.parent / f"{model_id}-test-predictions.parquet")
        result = evaluate_predictions(
            predictions,
            labels,
            model_id,
            "test",
            int(CONFIG["portfolio_policy"]["top_k"]),
            None,
            rebalance_interval=int(CONFIG["rebalance_interval"]),
            exit_rank_per_sector=int(CONFIG["portfolio_policy"]["exit_rank"]),
            max_replacements=int(CONFIG["portfolio_policy"]["max_replacements"]),
            minimum_weight=float(CONFIG["portfolio_policy"]["minimum_weight"]),
            maximum_weight=float(CONFIG["portfolio_policy"]["maximum_weight"]),
            price_panel=price_panel,
            correlation_lookback=int(CONFIG["portfolio_policy"]["correlation_lookback"]),
            correlation_penalty=float(CONFIG["portfolio_policy"]["correlation_penalty"]),
            cluster_correlation_threshold=float(CONFIG["portfolio_policy"]["cluster_correlation_threshold"]),
            maximum_cluster_members=int(CONFIG["portfolio_policy"]["maximum_cluster_members"]),
            weighting_method=str(CONFIG["portfolio_policy"].get("weighting_method", "rank_linear")),
            confidence_temperature=float(CONFIG["portfolio_policy"].get("confidence_temperature", 1.0)),
            confidence_shrinkage=float(CONFIG["portfolio_policy"].get("confidence_shrinkage", 0.35)),
            correlations=evaluation_correlations,
        )
        with (MODELS / f"{model_id}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    return metrics_payload(result), predictions


metrics = []
lightgbm = LGBModel(
    loss="mse",
    num_leaves=64,
    learning_rate=0.03,
    n_estimators=500,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
)
lightgbm_metrics, lightgbm_predictions = fit_and_evaluate(
    "qlib_lightgbm_alpha360_v1",
    lightgbm,
)
metrics.append(lightgbm_metrics)

base_model_id = str(CONFIG["model_id"])
seed_metrics = []
seed_predictions = []
deployment_models = []
for seed in CONFIG["model"]["seeds"]:
    seed_model_id = f"{base_model_id}_seed{int(seed)}"
    result, predictions = fit_and_evaluate(
        seed_model_id,
        build_gru(int(seed)),
    )
    metrics.append(result)
    seed_metrics.append(result)
    seed_predictions.append(predictions.rename(str(seed)))
    deployment_models.append(f"{seed_model_id}.pkl")

ensemble_config = CONFIG.get("ensemble", {})
ensemble_predictions = rank_blend_predictions([
    (lightgbm_predictions, float(ensemble_config.get("lightgbm_weight", 0.5))),
    *[
        (prediction, float(ensemble_config.get("gru_weight", 0.5)) / len(seed_predictions))
        for prediction in seed_predictions
    ],
])
labels = dataset.prepare("test", "label").iloc[:, 0]
ensemble_metrics = metrics_payload(
    evaluate_predictions(
        ensemble_predictions,
        labels,
        base_model_id,
        "test",
        int(CONFIG["portfolio_policy"]["top_k"]),
        None,
        rebalance_interval=int(CONFIG["rebalance_interval"]),
        exit_rank_per_sector=int(CONFIG["portfolio_policy"]["exit_rank"]),
        max_replacements=int(CONFIG["portfolio_policy"]["max_replacements"]),
        minimum_weight=float(CONFIG["portfolio_policy"]["minimum_weight"]),
        maximum_weight=float(CONFIG["portfolio_policy"]["maximum_weight"]),
        price_panel=price_panel,
        correlation_lookback=int(CONFIG["portfolio_policy"]["correlation_lookback"]),
        correlation_penalty=float(CONFIG["portfolio_policy"]["correlation_penalty"]),
        cluster_correlation_threshold=float(CONFIG["portfolio_policy"]["cluster_correlation_threshold"]),
        maximum_cluster_members=int(CONFIG["portfolio_policy"]["maximum_cluster_members"]),
        weighting_method=str(CONFIG["portfolio_policy"].get("weighting_method", "rank_linear")),
        confidence_temperature=float(CONFIG["portfolio_policy"].get("confidence_temperature", 1.0)),
        confidence_shrinkage=float(CONFIG["portfolio_policy"].get("confidence_shrinkage", 0.35)),
        correlations=evaluation_correlations,
    )
)
metrics.append(ensemble_metrics)
positive_seed_count = sum(float(row["rank_ic"]) > 0 for row in seed_metrics)
stability = {
    "seed_count": len(seed_metrics),
    "positive_seed_count": positive_seed_count,
    "positive_seed_ratio": positive_seed_count / len(seed_metrics),
    "rank_ic_mean": sum(float(row["rank_ic"]) for row in seed_metrics)
    / len(seed_metrics),
    "rank_ic_min": min(float(row["rank_ic"]) for row in seed_metrics),
    "rank_ic_max": max(float(row["rank_ic"]) for row in seed_metrics),
}
ensemble_artifact = MODELS / f"{base_model_id}_ensemble.json"
ensemble_artifact.write_text(
    json.dumps({
        "method": "daily_rank_blend",
        "models": [
            {"filename": "qlib_lightgbm_alpha360_v1.pkl", "weight": float(ensemble_config.get("lightgbm_weight", 0.5))},
            *[
                {"filename": filename, "weight": float(ensemble_config.get("gru_weight", 0.5)) / len(deployment_models)}
                for filename in deployment_models
            ],
        ],
    }, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

payload = {
    "created_at": datetime.now(UTC).isoformat(),
    "data_cutoff": (
        str(segments["test"][1])
        if CANDIDATE_TAG
        else str(frame.index.get_level_values("datetime").max().date())
    ),
    "universe_size": len(universe),
    "samples": len(frame),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "protocol": CONFIG,
    "effective_segments": segments,
    "time_contract": time_contract,
    "research_selected_rounds": selected_rounds,
    "candidate_tag": CANDIDATE_TAG or None,
    "metrics": metrics,
    "stability": stability,
}
payload["candidate_release"] = {"lifecycle": "ISOLATED_RESEARCH_NOT_REGISTERED"} if ISOLATED else register_qlib_candidate(
    ROOT,
    model_id=base_model_id,
    artifact_path=ensemble_artifact,
    data_cutoff=str(payload["data_cutoff"]),
)
research_path = (
    ARTIFACTS / "candidates" / CANDIDATE_TAG / "research.json"
    if CANDIDATE_TAG
    else ARTIFACTS / "latest-research.json"
)
research_path.parent.mkdir(parents=True, exist_ok=True)
research_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if CANDIDATE_TAG:
    (research_path.parent / "_RESEARCH_COMPLETE").write_text(
        f"completed_at={datetime.now(UTC).isoformat()}\n",
        encoding="utf-8",
    )
print(json.dumps(payload, ensure_ascii=False, indent=2))

# On the Windows CUDA runtime used by the local service, Python's native-library
# teardown can abort in ucrtbase.dll with 0xc0000409 after every artifact has
# already been durably written.  Candidate research runs are isolated child
# processes, so flush their output and avoid that unsafe native teardown path.
if CANDIDATE_TAG and os.name == "nt":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
