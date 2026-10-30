"""Isolated, explicit latest-data refit. Never registers a trading account."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore
from moneymore.latest_model_refit import fit_gru_fixed_rounds
from moneymore.qlib_challenger import QlibPanelDataset, build_challenger_dataset, challenger_universe, rank_blend_predictions
from moneymore.training_time_contract import weekly_time_contract
from moneymore.weekly_market_context import market_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--suspension-proof", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("isolated output must be new")
    report_path = args.research_dir / "research.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = report.get("research_selected_rounds", {})
    config = report["protocol"]
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    universe = challenger_universe(root, store)
    sessions = [d for d in trading_dates(store, args.as_of) if d <= args.as_of]
    bars = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=[
        ("ts_code", "in", list(universe)), ("trade_date", ">=", "20260101"),
        ("trade_date", "<=", args.as_of)])
    context, history = market_context(bars, [d for d in sessions if d >= "20260101"], universe, as_of=args.as_of)
    contract = weekly_time_contract(sessions, as_of=args.as_of, horizon=int(config["label_horizon"]))
    ensemble_path = args.research_dir / "models" / f'{config["model_id"]}_ensemble.json'
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    blockers = []
    unexplained = set(context["missing_symbols"])
    if args.suspension_proof:
        proof = pd.read_parquet(args.suspension_proof)
        full_day = proof.loc[(proof.trade_date.astype(str) == context["expected_session"])
            & proof.suspend_type.eq("S") & proof.suspend_timing.fillna("").eq("")]
        unexplained -= set(full_day.ts_code.astype(str))
        context["suspension_evidence_sha256"] = hashlib.sha256(args.suspension_proof.read_bytes()).hexdigest()
        context["confirmed_suspensions"] = sorted(set(context["missing_symbols"]) - unexplained)
    if unexplained:
        blockers.append("unexplained missing current-session bars: " + ",".join(sorted(unexplained)))
    if not report.get("time_contract"):
        blockers.append("source research has no causal time contract")
    for row in ensemble["models"]:
        model_id = Path(row["filename"]).stem
        if not isinstance(selected.get(model_id), int) or selected[model_id] < 1:
            blockers.append(f"missing validation-selected rounds: {model_id}")
    args.output.mkdir(parents=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(), "status": "BLOCKED" if blockers else "PREFLIGHT_READY",
        "blockers": blockers, "time_contract": contract,
        "research_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "context": context, "registered_account": False, "has_oos_score": False,
        "model_input": "unchanged price-volume features; market context observation only",
        "scope": "isolated_refit_not_publication", "trained_models": [],
    }
    output_manifest = args.output / "manifest.json"

    def save():
        output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    history.to_parquet(args.output / "market-history.parquet")
    save()
    if blockers or not args.execute:
        print(json.dumps({"status": manifest["status"], "blockers": blockers}, ensure_ascii=False))
        return 2 if blockers else 0
    # No whole-file pickle of untrusted remote data: source models are local
    # artifacts explicitly named by the source research ensemble.
    try:
        import lightgbm as lgb
        frame = build_challenger_dataset(store, universe,
            sequence_length=int(config["sequence_length"]), label_horizon=int(config["label_horizon"]),
            feature_count=int(config["model"]["d_feat"]))
        dataset = QlibPanelDataset(frame, {"train": tuple(contract["deployment_refit"])})
        train = dataset.prepare("train", ["feature", "label"]).dropna()
        manifest.update(status="RUNNING", actual_train_end=str(train.index.get_level_values("datetime").max().date()), samples=len(train))
        if manifest["actual_train_end"] != contract["latest_matured_signal"]:
            raise ValueError("actual training end does not match latest matured session")
        models_dir = args.output / "models"
        models_dir.mkdir()
        save()
        for row in ensemble["models"]:
            name = row["filename"]
            if Path(name).name != name:
                raise ValueError("unsafe artifact filename")
            model_id = Path(name).stem
            with (args.research_dir / "models" / name).open("rb") as stream:
                source = pickle.load(stream)
            if "seed" in model_id:
                from qlib.contrib.model.pytorch_gru import GRU
                seed = int(model_id.rsplit("seed", 1)[1])
                settings = config["model"]
                model = GRU(d_feat=int(settings["d_feat"]), hidden_size=int(settings["hidden_size"]),
                    num_layers=int(settings["num_layers"]), dropout=float(settings["dropout"]),
                    batch_size=int(settings["batch_size"]), lr=float(settings["learning_rate"]),
                    seed=seed, GPU=0 if source.use_gpu else -1)
                fit_gru_fixed_rounds(model, train, epochs=selected[model_id])
            else:
                from qlib.contrib.model.gbdt import LGBModel
                model = LGBModel(**source.params)
                params = dict(source.params)
                # sklearn n_estimators alias must not override selected rounds.
                params.pop("n_estimators", None)
                model.model = lgb.train(params, lgb.Dataset(train["feature"].values,
                    label=train["label"].iloc[:, 0].values), num_boost_round=selected[model_id])
            target = models_dir / name
            with target.open("wb") as stream:
                pickle.dump(model, stream)
            manifest["trained_models"].append({"filename": name, "rounds": selected[model_id],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256((args.research_dir / "models" / name).read_bytes()).hexdigest()})
            save()
        (models_dir / ensemble_path.name).write_text(json.dumps(ensemble, indent=2), encoding="utf-8")
        manifest["status"] = "VERIFYING_CURRENT_PREDICTIONS"
        save()
        live = build_challenger_dataset(store, universe,
            sequence_length=int(config["sequence_length"]), label_horizon=int(config["label_horizon"]),
            feature_count=int(config["model"]["d_feat"]), require_label=False, live_as_of=args.as_of)
        if live.empty or live.index.get_level_values("datetime").max() != pd.Timestamp(args.as_of):
            raise ValueError("latest prediction cross-section unavailable")
        live_dataset = QlibPanelDataset(live, {"forward": (args.as_of, args.as_of)})
        def predict_ensemble(folder):
            members = []
            for row in ensemble["models"]:
                with (folder / row["filename"]).open("rb") as stream:
                    fitted = pickle.load(stream)
                pred = fitted.predict(live_dataset, "forward")
                if pred.isna().any() or len(pred) != len(live):
                    raise ValueError("incomplete model prediction")
                members.append((pred, float(row["weight"])))
            return rank_blend_predictions(members)
        research_scores = predict_ensemble(args.research_dir / "models")
        refit_scores = predict_ensemble(models_dir)
        comparison = pd.concat([research_scores.rename("research_score"), refit_scores.rename("latest_refit_score")], axis=1)
        comparison.to_parquet(args.output / "current-score-comparison.parquet")
        research_top = set(research_scores.nlargest(10).index.get_level_values("instrument"))
        refit_top = set(refit_scores.nlargest(10).index.get_level_values("instrument"))
        manifest["current_comparison"] = {"date": args.as_of, "stocks": len(comparison),
            "rank_correlation": float(comparison.corr(method="spearman").iloc[0, 1]),
            "top10_overlap": len(research_top & refit_top),
            "research_top10": sorted(research_top), "latest_refit_top10": sorted(refit_top),
            "meaning": "same architecture/inputs, research training cutoff versus latest mature labels; not performance proof"}
        manifest["status"] = "COMPLETED_ISOLATED_REFIT_NOT_RELEASED"
        save()
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
