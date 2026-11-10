"""Three-window expanding audit of the best rebuilt mature candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/us_overnight_corrected/features/standard_daily_regime/panel.parquet"
AUDIT = ROOT / "state/us-overnight-open2open-v1/signal-rebuild/factor_audit.csv"
OUT = ROOT / "state/us-overnight-open2open-v1/signal-rebuild/rolling_robustness.json"
SEED = 20261107


def metrics(r):
    r = pd.Series(r).dropna(); eq = (1 + r).cumprod(); sd = r.std(ddof=1)
    return {"n": len(r), "mean_bps": r.mean()*1e4, "win": (r > 0).mean(),
            "sharpe": r.mean()/sd*np.sqrt(252) if sd else 0,
            "cumulative": eq.iloc[-1]-1, "drawdown": (eq/eq.cummax()-1).min()}


def evaluate(part, score):
    z = part[["timestamp", "symbol", "label_next_open_net"]].copy(); z["score"] = score
    z["rank"] = z.groupby("timestamp").score.rank(method="first", ascending=False)
    p = z[z["rank"] <= 10].pivot(index="timestamp", columns="rank", values="label_next_open_net")
    means = [p[i].mean()*1e4 for i in range(1, 11)]
    return {"top1": metrics(p[1]), "top5": metrics(p.loc[:, 1:5].mean(axis=1)),
            "top10": metrics(p.mean(axis=1)), "rank_mean_bps": means,
            "monotonic_pairs": sum(a >= b for a,b in zip(means, means[1:])),
            "top1_without_best5": metrics(p[1].drop(p[1].nlargest(5).index)),
            "top1_symbols": z[z["rank"] == 1].symbol.value_counts().head(10).to_dict()}


def main():
    df = pd.read_parquet(PANEL); df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    df = df.dropna(subset=["label_next_open_net"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    dates = np.array(sorted(df.timestamp.unique())); windows = [dates[-378:-252], dates[-252:-126], dates[-126:]]
    # Re-screen once using only history before the first audited window, then
    # freeze the list for all three windows.
    candidates = pd.read_csv(AUDIT).feature.tolist()
    discovery = df[df.timestamp < windows[0][0]]
    ic_rows=[]
    blocks=np.array_split(np.array(sorted(discovery.timestamp.unique())),3)
    for f in candidates:
        vals=[]
        for block in blocks:
            q=discovery[discovery.timestamp.isin(block)][['timestamp',f,'label_next_open_net']].dropna()
            corr=q.groupby('timestamp')[[f,'label_next_open_net']].corr(method='spearman')
            vals.append(float(corr.xs(f,level=1).label_next_open_net.mean()) if isinstance(corr.index,pd.MultiIndex) else np.nan)
        finite=[v for v in vals if np.isfinite(v)]
        ic_rows.append((f,np.mean(finite) if finite else np.nan,max(sum(v>0 for v in finite),sum(v<0 for v in finite))))
    selected=[x[0] for x in sorted(ic_rows,key=lambda x:abs(x[1]) if np.isfinite(x[1]) else -1,reverse=True) if x[2]>=2 and abs(x[1])>=.004][:40]
    X = df[selected].replace([np.inf,-np.inf],np.nan)
    rows=[]
    for wi, test_dates in enumerate(windows, 1):
        start = test_dates[0]; prior = dates[dates < start]; valid_dates = prior[-63:]
        tr = df.timestamp < valid_dates[0]; va = df.timestamp.isin(valid_dates); te = df.timestamp.isin(test_dates)
        med=X.loc[tr].median(); xx=X.fillna(med).fillna(0).astype('float32')
        resid=df.label_next_open_net-df.groupby('timestamp').label_next_open_net.transform('mean')
        model=CatBoostRegressor(loss_function='Huber:delta=1.0',iterations=800,depth=7,learning_rate=.03,
            l2_leaf_reg=10,random_seed=SEED,verbose=False,thread_count=16,od_type='Iter',od_wait=50)
        model.fit(xx.loc[tr],resid[tr],eval_set=(xx.loc[va],resid[va]),verbose=False)
        result=evaluate(df[te],model.predict(xx.loc[te])); result.update(window=wi,start=str(pd.Timestamp(test_dates[0]).date()),end=str(pd.Timestamp(test_dates[-1]).date()))
        rows.append(result)
    agg={k:{"mean_bps":float(np.mean([r[k]['mean_bps'] for r in rows])),
            "positive_windows":int(sum(r[k]['mean_bps'] > 0 for r in rows)),
            "mean_sharpe":float(np.mean([r[k]['sharpe'] for r in rows]))} for k in ('top1','top5','top10')}
    OUT.write_text(json.dumps({"candidate":"CatBoost market residual","windows":rows,"aggregate":agg,
        "selected_features":selected,"limitations":["no point-in-time stock-sector membership, so sector concentration is unavailable"]},indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o)),encoding='utf-8')
    print(json.dumps(agg,indent=2))
    return 0

if __name__ == '__main__': raise SystemExit(main())
