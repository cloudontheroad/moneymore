"""Audited daily cross-sectional rebuild for the US open-to-open selector.

Uses only information available by t-1 close.  This is research-only and never
submits orders.  It compares random selection, transparent factor composites,
LightGBM and CatBoost on raw and neutralized targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_corrected/features/standard_daily_regime/panel.parquet"
GROUPS = ROOT / "data/us_overnight_corrected/features/factor_groups.json"
OUT = ROOT / "state/us-overnight-open2open-v1/signal-rebuild"
TEST_START = pd.Timestamp("2026-05-07", tz="UTC")
VALID_START = pd.Timestamp("2025-11-07", tz="UTC")
SEED = 20261107


def metric(r: pd.Series) -> dict:
    r = pd.Series(r).dropna().astype(float)
    eq = (1 + r).cumprod(); sd = r.std(ddof=1)
    return {"sessions": len(r), "mean_bps": r.mean() * 1e4,
            "cumulative": eq.iloc[-1] - 1, "win_rate": (r > 0).mean(),
            "volatility": sd * np.sqrt(252),
            "sharpe": r.mean() / sd * np.sqrt(252) if sd else 0,
            "max_drawdown": (eq / eq.cummax() - 1).min()}


def evaluate(frame: pd.DataFrame, score: np.ndarray) -> dict:
    z = frame[["timestamp", "symbol", "label_next_open_net"]].copy(); z["score"] = score
    z["rank"] = z.groupby("timestamp")["score"].rank(method="first", ascending=False)
    slots = z[z["rank"] <= 10].pivot(index="timestamp", columns="rank", values="label_next_open_net")
    rank = {str(i): metric(slots[i]) for i in range(1, 11)}
    topn = {str(i): metric(slots.loc[:, 1:i].mean(axis=1)) for i in range(1, 11)}
    means = [rank[str(i)]["mean_bps"] for i in range(1, 11)]
    top1 = z[z["rank"] == 1].set_index("timestamp")["label_next_open_net"]
    without_best5 = metric(top1.drop(top1.nlargest(5).index))
    concentration = z[z["rank"] == 1].symbol.value_counts(normalize=True)
    return {"rank": rank, "topn": topn, "adjacent_monotonic": sum(a >= b for a, b in zip(means, means[1:])),
            "strict_monotonic": all(a > b for a, b in zip(means, means[1:])),
            "top1_without_best_5_days": without_best5,
            "top1_max_symbol_share": float(concentration.iloc[0]),
            "top1_unique_symbols": int(len(concentration))}


def random_baseline(test: pd.DataFrame, trials: int = 10_000) -> dict:
    rng = np.random.default_rng(SEED)
    days = [g.label_next_open_net.dropna().to_numpy() for _, g in test.groupby("timestamp")]
    out = {}
    for n in (1, 5, 10):
        columns = []
        for v in days:
            if len(v) < n:
                continue
            chunks = []
            for start in range(0, trials, 500):
                size = min(500, trials - start)
                keys = rng.random((size, len(v)))
                pick = np.argpartition(keys, n - 1, axis=1)[:, :n]
                chunks.append(v[pick].mean(axis=1))
            columns.append(np.concatenate(chunks))
        daily = np.column_stack(columns)
        means = daily.mean(axis=1) * 1e4
        sharpes = daily.mean(axis=1) / daily.std(axis=1, ddof=1) * np.sqrt(252)
        cumulative = np.prod(1 + daily, axis=1) - 1
        out[str(n)] = {"mean_bps_p05_p50_p95": np.quantile(means, [.05, .5, .95]).tolist(),
                       "sharpe_p05_p50_p95": np.quantile(sharpes, [.05, .5, .95]).tolist(),
                       "cumulative_p05_p50_p95": np.quantile(cumulative, [.05, .5, .95]).tolist()}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))
    df = pd.read_parquet(PANEL)
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    meta = {"timestamp", "feature_asof", "symbol", "label_next_open_net"}
    features = [c for c in df.columns if c not in meta]
    df = df.dropna(subset=["label_next_open_net"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    train = df.timestamp < VALID_START; valid = (df.timestamp >= VALID_START) & (df.timestamp < TEST_START); test = df.timestamp >= TEST_START

    # Labels: raw cross-sectional rank; market residual; and market residual
    # standardized by lagged, known overnight volatility.
    daily_mean = df.groupby("timestamp").label_next_open_net.transform("mean")
    market_resid = df.label_next_open_net - daily_mean
    # At the decision immediately before open(t), label(t-1) still contains
    # open(t) and is unknown.  Shift by two so volatility uses only completed
    # open-to-open observations through t-2.
    vol = df.get("overnight_std_20", df.groupby("symbol").label_next_open_net.transform(lambda s: s.shift(2).rolling(20).std()))
    vol = vol.clip(lower=0.005).fillna(vol[train].median())
    labels = {"raw_rank": df.groupby("timestamp").label_next_open_net.rank(pct=True) - .5,
              "market_residual": market_resid,
              "vol_scaled_market_residual": market_resid / vol}

    # Factor audit uses development data only, split into three chronological blocks.
    audit_rows = []
    tr_dates = np.array(sorted(df.loc[train, "timestamp"].unique()))
    blocks = np.array_split(tr_dates, 3)
    family_of = {f: family for family, fs in groups.items() for f in fs}
    family_of.update({f: "standard_technical" for f in features if f not in family_of})
    for f in features:
        vals = []
        for block in blocks:
            q = df[train & df.timestamp.isin(block)][["timestamp", f, "label_next_open_net"]].dropna()
            corr = q.groupby("timestamp")[[f, "label_next_open_net"]].corr(method="spearman")
            ic = (corr.xs(f, level=1)["label_next_open_net"]
                  if isinstance(corr.index, pd.MultiIndex) else pd.Series(dtype=float))
            vals.append(float(ic.mean()))
        finite = [x for x in vals if np.isfinite(x)]
        audit_rows.append({"feature": f, "family": family_of[f], "ic_block1": vals[0], "ic_block2": vals[1],
                           "ic_block3": vals[2], "mean_ic": np.mean(finite) if finite else np.nan,
                           "same_sign_blocks": max(sum(x > 0 for x in vals), sum(x < 0 for x in vals))})
    audit = pd.DataFrame(audit_rows).sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)
    selected = audit[(audit.same_sign_blocks >= 2) & (audit.mean_ic.abs() >= .004)].head(40).feature.tolist()
    audit.to_csv(OUT / "factor_audit.csv", index=False)

    X = df[selected].replace([np.inf, -np.inf], np.nan)
    med = X.loc[train].median(); X = X.fillna(med).fillna(0).astype("float32")
    # Transparent baselines use daily percentile ranks and train-only IC signs/weights.
    xr = X.groupby(df.timestamp).rank(pct=True) - .5
    ic = audit.set_index("feature").loc[selected, "mean_ic"]
    scores = {"factor_equal": (xr * np.sign(ic)).mean(axis=1).to_numpy(),
              "factor_ic_weighted": (xr * ic / ic.abs().sum()).sum(axis=1).to_numpy()}
    results = {}
    for name, score in scores.items(): results[name] = evaluate(df[test], score[test])

    for label_name, y in labels.items():
        lgbm = lgb.LGBMRegressor(objective="huber", n_estimators=800, learning_rate=.03, num_leaves=23,
                                 min_child_samples=100, subsample=.8, colsample_bytree=.8,
                                 reg_lambda=10, random_state=SEED, n_jobs=16, verbosity=-1)
        lgbm.fit(X.loc[train], y[train], eval_set=[(X.loc[valid], y[valid])],
                 callbacks=[lgb.early_stopping(50, verbose=False)])
        results[f"lightgbm_{label_name}"] = evaluate(df[test], lgbm.predict(X.loc[test]))
        cat = CatBoostRegressor(loss_function="Huber:delta=1.0", iterations=800, depth=7, learning_rate=.03,
                                l2_leaf_reg=10, random_seed=SEED, verbose=False, thread_count=16,
                                od_type="Iter", od_wait=50)
        cat.fit(X.loc[train], y[train], eval_set=(X.loc[valid], y[valid]), verbose=False)
        results[f"catboost_{label_name}"] = evaluate(df[test], cat.predict(X.loc[test]))

    random = random_baseline(df[test])
    summary = []
    for name, res in results.items():
        row = {"model": name, "monotonic_pairs": res["adjacent_monotonic"],
               "top1_without_best5_bps": res["top1_without_best_5_days"]["mean_bps"]}
        for n in (1, 5, 10):
            row.update({f"top{n}_bps": res["topn"][str(n)]["mean_bps"],
                        f"top{n}_sharpe": res["topn"][str(n)]["sharpe"],
                        f"top{n}_cum": res["topn"][str(n)]["cumulative"],
                        f"top{n}_win": res["topn"][str(n)]["win_rate"],
                        f"top{n}_dd": res["topn"][str(n)]["max_drawdown"]})
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values(["top1_sharpe", "top10_sharpe"], ascending=False)
    summary.to_csv(OUT / "model_summary.csv", index=False)
    payload = {"contract": "daily data through t-1 close; buy open(t), sell open(t+1); label net 20bps",
               "split": {"train_end": str(VALID_START.date()), "valid_end": str(TEST_START.date()),
                         "test_sessions": int(df.loc[test, "timestamp"].nunique())},
               "available_families": {k: len(v) for k, v in groups.items()}, "selected_features": selected,
               "data_gaps": ["no point-in-time per-stock industry classification",
                             "no completed point-in-time SEC event table", "no complete after-hours news/analyst event feed"],
               "random_10000": random, "models": results, "summary": summary.to_dict("records")}
    (OUT / "full_report.json").write_text(
        json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
