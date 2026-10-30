"""Offline model/style/optimizer comparisons; no portfolio registration or orders."""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore
from moneymore.qlib_challenger import QlibPanelDataset, build_challenger_dataset, challenger_universe, rank_blend_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("review output already exists")
    root = Path(__file__).resolve().parents[1]
    store = ParquetStore(root / "data")
    report = json.loads((args.research_dir / "research.json").read_text(encoding="utf-8"))
    config = report["protocol"]
    models_dir = args.models_dir or args.research_dir / "models"
    universe = challenger_universe(root, store)
    frame = build_challenger_dataset(store, universe, sequence_length=int(config["sequence_length"]),
        label_horizon=int(config["label_horizon"]), feature_count=int(config["model"]["d_feat"]),
        require_label=False, live_as_of=args.as_of)
    if frame.empty or frame.index.get_level_values("datetime").max() != pd.Timestamp(args.as_of):
        raise ValueError("current cross-section missing")
    dataset = QlibPanelDataset(frame, {"forward": (args.as_of, args.as_of)})
    ensemble = json.loads((models_dir / f'{config["model_id"]}_ensemble.json').read_text(encoding="utf-8"))
    predictions, loaded = {}, []
    for member in ensemble["models"]:
        filename = member["filename"]
        if Path(filename).name != filename:
            raise ValueError("invalid model filename")
        with (models_dir / filename).open("rb") as stream:
            model = pickle.load(stream)
        loaded.append(model)  # retain CUDA objects until explicit process exit
        score = model.predict(dataset, "forward")
        if len(score) != len(frame) or not np.isfinite(score).all():
            raise ValueError("invalid predictions")
        predictions[Path(filename).stem] = score
    predictions["ensemble_50_50"] = rank_blend_predictions([
        (predictions[Path(m["filename"]).stem], float(m["weight"])) for m in ensemble["models"]])
    gru = [(s, 1.) for n, s in predictions.items() if "seed" in n]
    predictions["gru_only"] = rank_blend_predictions(gru)
    calendar = [d for d in trading_dates(store, args.as_of) if d <= args.as_of][-121:]
    bars = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=[
        ("ts_code", "in", list(universe)), ("trade_date", ">=", calendar[0]), ("trade_date", "<=", args.as_of)])
    adj = store.read("adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=[
        ("ts_code", "in", list(universe)), ("trade_date", ">=", calendar[0]), ("trade_date", "<=", args.as_of)])
    joined = bars.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    joined["adjusted"] = joined.close * joined.adj_factor
    prices = joined.pivot(index="trade_date", columns="ts_code", values="adjusted").reindex(calendar)
    returns = prices.pct_change(fill_method=None)
    basic = store.read("daily_basic", columns=["ts_code", "total_mv"], filters=[("trade_date", "==", args.as_of)])
    size = basic.drop_duplicates("ts_code").set_index("ts_code").total_mv
    style = pd.DataFrame({"momentum20": prices.pct_change(20, fill_method=None).iloc[-1],
        "volatility20": returns.tail(20).std(), "log_size": np.log(size.where(size > 0))})
    rows = []
    flat_scores = {}
    for name, score in predictions.items():
        score = score.copy()
        score.index = score.index.get_level_values("instrument")
        flat_scores[name] = score
        top = score.nlargest(10).index.tolist()
        rows.append({"model": name, "top10": [{"symbol": s, "industry": universe[s]} for s in top],
            "growth_board_count": sum(s.startswith(("30", "68")) for s in top),
            "industry_counts": pd.Series([universe[s] for s in top]).value_counts().to_dict(),
            "style_rank_correlations": {c: score.corr(style[c], method="spearman") for c in style}})
    risk = {"status": "NOT_RUN"}
    try:
        from moneymore.research.enhanced_indexing import EnhancedPolicy, optimize_snapshot
        scores = flat_scores["ensemble_50_50"]
        history = returns.iloc[1:].dropna(axis=1)
        eligible = history.columns.intersection(scores.index).intersection(size.dropna().index)
        history = history[eligible]
        w0 = pd.Series(0., index=eligible)
        top = scores.reindex(eligible).nlargest(10).index
        w0.loc[top] = 1 / len(top)
        risk = optimize_snapshot(scores.reindex(eligible), history, size.reindex(eligible), w0,
            EnhancedPolicy(), solver_max_iters=200)
        risk["comparison_scope"] = "same eligible risk universe; diagnostic default policy, not validated live allocation"
        risk["excluded_from_risk_universe"] = sorted(set(scores.index) - set(eligible))
        risk["reference_top10"] = list(top)
    except Exception as exc:
        risk = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    result = {"as_of": args.as_of, "models_directory": str(models_dir), "orders_enabled": False,
        "scope": "OFFLINE_DIAGNOSIS_NOT_PROMOTION", "models": rows,
        "score_rank_correlations": pd.DataFrame(flat_scores).corr(method="spearman").to_dict(),
        "research_metrics": report["metrics"], "stability": report["stability"],
        "target_audit": {"current_target": "5-day return minus same-day sector mean",
            "absolute_return_forecast": False,
            "warning": "relative outperformance may coexist with absolute losses",
            "alternative_target_training": "NOT_IMPLEMENTED_REQUIRES_SEPARATE_PROTOCOL"},
        "risk_optimizer": risk}
    args.output.mkdir(parents=True)
    pd.DataFrame(flat_scores).to_parquet(args.output / "member-scores.parquet")
    # Pandas converts missing floating diagnostics into strict JSON null.
    (args.output / "review.json").write_text(pd.Series(result).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "REVIEW_WRITTEN", "models": len(rows), "risk": risk["status"]}))
    if os.name == "nt":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
