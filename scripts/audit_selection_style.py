"""Read-only strategy style audit; writes diagnostics only, never recommendations/orders."""
import json
from pathlib import Path

import pandas as pd
import yaml

from moneymore.data.store import ParquetStore
from moneymore.research.sector_model import build_global_factor_snapshot
from moneymore.strategy_universe import active_strategy_universe

ROOT = Path(__file__).resolve().parents[1]
STORE = ParquetStore(ROOT / "data")
ASOF = "20261026"


def board(symbol):
    return "STAR" if symbol.startswith(("688", "689")) else "ChiNext" if symbol.startswith(("300", "301")) else "Main"


def summary(frame, symbol="symbol", score="score"):
    f = frame.copy()
    f["board"] = f[symbol].map(board)
    result = f.groupby("board")[score].agg(["count", "mean", "median", "std"]).round(6).reset_index().to_dict("records")
    ordered = f.sort_values([score, symbol], ascending=[False, True])
    return {"boards": result, "top_counts": {str(k): ordered.head(k)["board"].value_counts().to_dict() for k in (10, 20, 50, 100)}}


def main():
    universe = active_strategy_universe(STORE)
    symbols = universe.symbol.astype(str).tolist()
    instruments = STORE.read("instruments").drop_duplicates("ts_code").set_index("ts_code")
    saved = STORE.read("global_factor_recommendation")
    saved = saved.loc[pd.to_datetime(saved.date) == pd.to_datetime(saved.date).max()].copy()
    weights = yaml.safe_load((ROOT / "configs/sector_models.yaml").read_text(encoding="utf-8"))["global_selection"]["factors"]
    # Rebuilt after market-gap repair: diagnostic counterfactual, NOT an as-run factor record.
    rebuilt = build_global_factor_snapshot(STORE, symbols, weights)
    financial = STORE.read("fina_indicator")
    financial_coverage = {"stored_symbols": int(financial.ts_code.nunique()), "latest_announcement": str(financial.ann_date.max()), "pool_with_stored_records": int(financial.loc[financial.ts_code.isin(symbols), "ts_code"].nunique()), "roe_available": int(rebuilt.roe.notna().sum()), "growth_available": int(rebuilt.quarterly_revenue_growth.notna().sum()), "active_weight_distribution": rebuilt.active_weight.round(2).value_counts().sort_index().to_dict()}
    bars = STORE.read("daily", columns=["ts_code", "trade_date", "close"], filters=[("ts_code", "in", symbols), ("trade_date", ">=", "20260501"), ("trade_date", "<=", ASOF)]).sort_values(["ts_code", "trade_date"])
    adjustments = STORE.read("adj_factor", filters=[("ts_code", "in", symbols), ("trade_date", ">=", "20260501"), ("trade_date", "<=", ASOF)])
    bars = bars.merge(adjustments, on=["ts_code", "trade_date"], how="inner")
    bars["adjusted"] = bars.close * bars.adj_factor
    bars["return20"] = bars.groupby("ts_code").adjusted.pct_change(20, fill_method=None)
    bars["volatility20"] = bars.groupby("ts_code").adjusted.transform(lambda s: s.pct_change(fill_method=None).rolling(20).std())
    characteristics = bars.groupby("ts_code").tail(1).set_index("ts_code")[["return20", "volatility20"]]
    characteristics["size"] = universe.set_index("symbol").total_mv
    comparable = saved.merge(rebuilt[["symbol", "score"]], on="symbol", suffixes=("_saved", "_rebuilt"))
    groups = {"value_dividend": ["earnings_yield", "book_to_price", "dividend_yield_ttm"], "quality_growth": ["roe", "quarterly_revenue_growth"], "momentum": ["momentum_252_21", "return_120"], "low_risk": ["volatility_60", "drawdown_120"]}
    contributions = []
    for group, names in groups.items():
        rebuilt[group] = sum(rebuilt[n].fillna(0) * weights[n] for n in names) / rebuilt.active_weight
        for b, f in rebuilt.groupby(rebuilt.symbol.map(board)):
            contributions.append({"group": group, "board": b, "mean_contribution": float(f[group].mean()), "factor_coverage": {n: float(f[n].notna().mean()) for n in names}})
    ablations = {}
    for group, names in groups.items():
        counter = rebuilt.copy()
        remaining = [n for n in weights if n not in names]
        denominator = sum(counter[n].notna() * weights[n] for n in remaining)
        counter["score"] = sum(counter[n].fillna(0) * weights[n] for n in remaining) / denominator.replace(0, float("nan"))
        ablations["without_" + group] = summary(counter)
    accounts = {}
    paths = {"baseline": ROOT / "state/multi-sector-shadow", "main_gru": ROOT / "state/qlib-challenger-shadow"}
    paths.update({p.name: p for p in (ROOT / "state/qlib-candidate-shadow").iterdir() if p.is_dir()})
    daily = STORE.read("daily", columns=["ts_code", "trade_date", "close"], filters=[("trade_date", "==", ASOF)])
    closes = daily.set_index("ts_code").close
    for name, directory in paths.items():
        path = directory / (ASOF + ".json")
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for position in report.get("portfolio", {}).get("positions", []):
            s = position["symbol"]
            if position["quantity"] <= 0:
                continue
            rows.append({"symbol": s, "name": instruments.loc[s, "name"] if s in instruments.index else s, "industry": instruments.loc[s, "industry"] if s in instruments.index else "unknown", "board": board(s), "equity_weight": float(position["quantity"] * closes.get(s, 0) / report["portfolio"]["equity"])})
        scores = pd.DataFrame(report.get("scores", []))
        history = []
        for historical in sorted(directory.glob("202610*.json")):
            if not "20261012" <= historical.stem <= ASOF:
                continue
            r = json.loads(historical.read_text(encoding="utf-8"))
            selected = r.get("selected", list(r.get("target_weights", {})))
            if selected:
                history.append({"date": historical.stem, "selected_count": len(selected), "star_chinext_count": sum(board(s) != "Main" for s in selected)})
                historical_scores = pd.DataFrame(r.get("scores", []))
                if not historical_scores.empty:
                    reference = bars.loc[bars.trade_date.astype(str) == historical.stem].set_index("ts_code")
                    joined = historical_scores.set_index("instrument").join(reference[["return20", "volatility20"]])
                    history[-1]["score_return20_spearman"] = float(joined.score.corr(joined.return20, method="spearman"))
                    history[-1]["raw_top10_star_chinext"] = sum(board(s) != "Main" for s in historical_scores.sort_values("score", ascending=False).head(10).instrument)
        accounts[name] = {"holdings": rows, "board_equity_weights": pd.DataFrame(rows).groupby("board").equity_weight.sum().to_dict() if rows else {}, "selection_history": history}
        if not scores.empty:
            accounts[name]["scores"] = summary(scores, "instrument")
            accounts[name]["top20"] = scores.sort_values("score", ascending=False).head(20).to_dict("records")
            linked = scores.set_index("instrument").join(characteristics)
            accounts[name]["score_style_spearman"] = {c: float(linked.score.corr(linked[c], method="spearman")) for c in characteristics}
            standardized = scores.copy()
            standardized["board"] = standardized.instrument.map(board)
            standardized["score"] = standardized.groupby("board").score.transform(lambda s: (s - s.mean()) / s.std())
            accounts[name]["board_standardized_diagnostic"] = summary(standardized, "instrument")
    saved["raw_rank"] = saved.score.rank(ascending=False, method="first")
    result = {"as_of": ASOF, "taxonomy": "Listing board is not technology industry; STAR includes 689. Current instrument industries, not historical point-in-time taxonomy.", "universe": pd.Series(symbols).map(board).value_counts().to_dict(), "saved_baseline": summary(saved), "saved_baseline_top20": saved.sort_values("score", ascending=False).head(20).to_dict("records"), "saved_baseline_best_growth_board": saved.loc[saved.symbol.map(board) != "Main"].sort_values("score", ascending=False).head(10).to_dict("records"), "rebuilt_baseline": summary(rebuilt), "rebuilt_vs_saved": {"matched": len(comparable), "max_abs_score_change": float((comparable.score_saved - comparable.score_rebuilt).abs().max()), "rank_correlation": float(comparable.score_saved.corr(comparable.score_rebuilt, method="spearman"))}, "rebuilt_contributions": contributions, "diagnostic_ablations": ablations, "accounts": accounts}
    result["financial_coverage"] = financial_coverage
    out = ROOT / "state/challenger-diagnostics/selection-style-audit-20261026.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"accounts", "saved_baseline_top20", "saved_baseline_best_growth_board", "diagnostic_ablations"}}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
