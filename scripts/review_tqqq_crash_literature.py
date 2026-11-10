"""Check published industry run-ups and transfer crash-risk features to TQQQ."""

from __future__ import annotations

import argparse
import json
import re
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from review_tqqq_kelly_forecast import standardize
from review_tqqq_peak_exit import recent_high
from review_tqqq_reentry_payoff import forward_payoffs, inputs

BASE = ("gap", "change5", "change21", "vol63", "credit", "vol_term")
GROUPS = {
    "base": (),
    "runup": ("semi_runup", "semi_acceleration", "semi_vol_rise"),
    "disagreement": ("qqq_volume", "semi_volume", "qqq_runup", "negative_skew"),
    "options": ("skew_level", "skew_change", "vvix_level", "vvix_change"),
}


def french_table(path, header, *, title):
    with ZipFile(path) as archive:
        lines = archive.read(archive.namelist()[0]).decode("utf-8-sig").splitlines()
    line = next(i for i, row in enumerate(lines) if header in row) + int(title)
    finish = next(i for i in range(line + 1, len(lines)) if not re.match(r"^\s*\d{6},", lines[i]))
    table = pd.read_csv(StringIO("\n".join(lines[line:finish])), index_col=0, na_values=[-99.99, -999])
    table.index = pd.PeriodIndex(table.index.astype(str), freq="M")
    return table


def industry_replication(industry_path, market_path):
    returns = french_table(industry_path, "Average Value Weighted Returns -- Monthly", title=True).drop(columns="Other") / 100
    counts = french_table(industry_path, "Number of Firms in Portfolios", title=True)
    factors = french_table(market_path, "Mkt-RF", title=False)
    market = (factors["Mkt-RF"] + factors.RF).reindex(returns.index) / 100
    log = np.log1p(returns)
    market24 = np.expm1(np.log1p(market).rolling(24).sum())
    raw24 = np.expm1(log.rolling(24).sum())
    raw60 = np.expm1(log.rolling(60).sum())
    acceleration = raw24 - np.expm1(log.rolling(12).sum().shift(12))
    volatility = log.rolling(12).std().diff(12)
    five_year = raw60.ge(0.50) | (returns.index.year < 1931)[:, None]
    output = {}
    for threshold in (0.50, 1.00, 1.50):
        eligible = raw24.ge(threshold) & raw24.sub(market24, axis=0).ge(threshold)
        eligible &= five_year & counts.reindex_like(returns).ge(10)
        records = []
        for name, mask in eligible.items():
            last = -24
            for i in np.flatnonzero(mask.to_numpy()):
                if i - last < 24:
                    continue
                future = returns[name].iloc[i + 1:i + 25]
                if len(future) < 24 or future.isna().any():
                    continue
                last = i
                nav = np.r_[1, (1 + future.to_numpy()).cumprod()]
                maximum = np.maximum.accumulate(nav)
                low = np.argmin(nav / maximum)
                top = np.argmax(nav[:low + 1])
                records.append({
                    "date": str(returns.index[i]), "industry": name, "crash": nav[low] / maximum[low] <= 0.60,
                    "drawdown": float(nav[low] / maximum[low] - 1),
                    "return12": float(nav[12] - 1), "return24": float(nav[-1] - 1),
                    "gain_until_peak": float(nav[top] - 1), "months_until_peak": int(top),
                    "acceleration": float(acceleration[name].iloc[i]),
                    "volatility_change": float(volatility[name].iloc[i]),
                })
        table = pd.DataFrame(records).sort_values("date")
        periods = {}
        for label, subset in (("original_window", table.loc[table.date.le("2012-03")]),
                              ("extension", table.loc[table.date.gt("2012-03")])):
            crashed, other = subset.loc[subset.crash], subset.loc[~subset.crash]
            periods[label] = {
                "episodes": len(subset), "crashes": len(crashed),
                "mean_return12": float(subset.return12.mean()) if len(subset) else None,
                "mean_return24": float(subset.return24.mean()) if len(subset) else None,
                "crash_gain_until_peak": float(crashed.gain_until_peak.mean()) if len(crashed) else None,
                "crash_months_until_peak": float(crashed.months_until_peak.mean()) if len(crashed) else None,
                "crash_acceleration": float(crashed.acceleration.mean()) if len(crashed) else None,
                "other_acceleration": float(other.acceleration.mean()) if len(other) else None,
                "crash_volatility_change": float(crashed.volatility_change.mean()) if len(crashed) else None,
                "other_volatility_change": float(other.volatility_change.mean()) if len(other) else None,
            }
        output[f"{threshold:.0%}"] = {"groups": periods, "episodes": json.loads(table.to_json(orient="records"))}
    return output


def market_features(history, context, volume, options):
    close = np.log(history.qqqClose)
    semi = np.log(context.SOXX.reindex(history.index))
    spy = np.log(context.SPY.reindex(history.index))
    volume = np.log(volume.reindex(history.index).where(lambda values: values.gt(0)))
    abnormal_volume = volume.rolling(126).mean() - volume.shift(126).rolling(378).mean()
    options = options.reindex(history.index)
    semi_vol = semi.diff().rolling(63).std()
    return pd.DataFrame({
        "gap": close - close.rolling(63).max(),
        "change5": close.diff(5), "change21": close.diff(21), "vol63": close.diff().rolling(63).std(),
        "credit": np.log(history.HYG / history.IEI).diff(5),
        "vol_term": np.log(history.VIX / history.VIX3M).diff(5),
        "semi_runup": (semi - spy).diff(504),
        "semi_acceleration": np.expm1(semi.diff(504)) - np.expm1(semi.diff(252).shift(252)),
        "semi_vol_rise": semi_vol.diff(252),
        "qqq_volume": abnormal_volume.QQQ, "semi_volume": abnormal_volume.SOXX,
        "qqq_runup": close.diff(756), "negative_skew": -close.diff().rolling(126).skew(),
        "skew_level": options.SKEW, "skew_change": options.SKEW.diff(5),
        "vvix_level": np.log(options.VVIX / history.VIX), "vvix_change": np.log(options.VVIX).diff(5),
        "recent": recent_high(close).le(21),
    })


def predict(data, labels, threshold, *, train_until=None):
    names = [*BASE, *(name for block in GROUPS.values() for name in block)]
    eligible = data.recent & data[names].notna().all(axis=1)
    result = pd.DataFrame(index=data.index, columns=[*GROUPS, "combined"], dtype=float)
    for date in data.index[eligible]:
        cutoff = min(date, train_until) if train_until is not None else date
        past = eligible & labels.settled.le(cutoff)
        y = labels.loc[past, "loss"].ge(-np.log1p(-threshold))
        if len(y) < 78 or min(y.sum(), (~y).sum()) < 8:
            continue
        for group, variables in (*GROUPS.items(), ("combined", tuple(names[len(BASE):]))):
            keys = [*BASE, *variables]
            x, query = standardize(data.loc[past, keys].to_numpy(), data.loc[date, keys].to_numpy())
            model = LogisticRegression(C=0.15, max_iter=500).fit(x, y)
            result.at[date, group] = model.predict_proba(query[None])[0, 1]
    return result


def elevated(scores):
    return scores.gt(scores.shift().rolling(104, min_periods=26).quantile(0.80))


def nonoverlapping_alerts(alerts, settled):
    selected = []
    previous_end = pd.Timestamp.min
    for date, end in settled.loc[alerts].items():
        if date >= previous_end:
            selected.append(date)
            previous_end = end
    return selected


def evaluate(scores, labels, data, threshold):
    common = scores.notna().all(axis=1) & labels.settled.notna()
    tail = labels.loss.ge(-np.log1p(-threshold))
    flags = elevated(scores)
    cost = np.log((1 + 0.0025) / (1 - 0.0025))
    net_exit = np.expm1(-cost) - np.expm1(labels.terminal)
    near_high = common & data.gap.ge(np.log(0.97))
    before_2026 = data.index.year < 2026
    groups = {
        "all": common, "near_high": near_high, "before_2026": common & before_2026,
        "near_high_before_2026": near_high & before_2026, "2026": common & ~before_2026,
    }
    output = {}
    for name, mask in groups.items():
        y = tail.loc[mask]
        result = {"weeks": len(y), "crash_weeks": int(y.sum()),
                  "start": str(y.index.min().date()) if len(y) else None,
                  "end": str(y.index.max().date()) if len(y) else None}
        for model in scores:
            alert = flags[model] & mask
            selected = nonoverlapping_alerts(alert, labels.settled)
            result[model] = {
                "auc": float(roc_auc_score(y, scores.loc[mask, model])) if y.nunique() == 2 else None,
                "brier": float(np.square(scores.loc[mask, model] - y).mean()) if len(y) else None,
                "alerts": int(alert.sum()), "tail_hits": int((alert & tail).sum()),
                "mean_net_exit": float(net_exit.loc[alert].mean()) if alert.any() else None,
                "nonoverlapping_alerts": len(selected),
                "nonoverlapping_hits": int(tail.loc[selected].sum()),
                "nonoverlapping_mean_net_exit": float(net_exit.loc[selected].mean()) if selected else None,
            }
        output[name] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history", "context", "alternate", "volume", "skew", "vvix", "industries", "factors"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()

    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    history, context, alternate, volume = map(read, (args.history, args.context, args.alternate, args.volume))
    options = pd.concat([read(args.skew).SKEW, read(args.vvix).VVIX], axis=1)
    _, weekly = inputs(history, context, alternate)
    signals = market_features(history, context, volume, options).reindex(weekly.index)
    realized = forward_payoffs(history, weekly.index, 4)
    scores = predict(signals, realized, 0.20)
    date = "2022-10-14"
    prefix = market_features(history.loc[:date], context.loc[:date], volume.loc[:date],
                             options.loc[:date]).reindex(weekly.loc[:date].index)
    pd.testing.assert_frame_equal(prefix, signals.loc[:date])
    pd.testing.assert_frame_equal(predict(prefix, forward_payoffs(history.loc[:date], prefix.index, 4), 0.20),
                                  scores.loc[:date])
    frozen = predict(signals, realized, 0.20, train_until=pd.Timestamp("2025-12-31"))
    print(json.dumps({
        "industry_replication": industry_replication(args.industries, args.factors),
        "tqqq": evaluate(scores, realized, signals, 0.20),
        "frozen_2026": evaluate(frozen, realized, signals, 0.20)["2026"],
        "tail_sensitivity": {f"{cut:.0%}": evaluate(predict(signals, realized, cut), realized, signals, cut)
                             for cut in (0.15, 0.25)},
        "october": json.loads(signals.join([frozen.add_prefix("score_"), elevated(frozen).add_prefix("alert_")])
                              .loc["2026-09-15":]
                              .reset_index().to_json(orient="records", date_format="iso")),
    }, indent=2))


if __name__ == "__main__":
    main()
