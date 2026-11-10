"""Walk-forward weekly persistence forecasts and their TQQQ execution checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from review_tqqq_breadth import PEAK_WINDOWS
from review_tqqq_kelly_forecast import standardize
from review_tqqq_sharpe import daily_returns, paired_interval, sharpe
from review_tqqq_swing import replay
from review_tqqq_swing_round_trips import path_metrics
from review_tqqq_turning_points import price_signals

PRICE = ("gap26", "change4", "change13", "change26", "below10_share", "vol13")
CROSS = ("chips4", "chips13", "consumer4", "consumer13", "breadth13", "credit4", "vix_curve", "yield13")


def weekly_features(history, context, alternate):
    weekly = history.groupby(history.index.to_period("W-FRI")).tail(1)
    if weekly.index[-1].dayofweek != 4:
        # Earlier holiday-shortened weeks are already complete. Drop the
        # final week when the last input date could be mid-week.
        weekly = weekly.iloc[:-1]
    close = weekly.qqqClose
    qqq = np.log(close)
    fast_average = close.rolling(10).mean()
    below_fast = close.lt(fast_average).where(fast_average.notna())
    above_fast = close.gt(fast_average)
    source = context.reindex(weekly.index)
    alt = alternate.reindex(weekly.index)
    chips = np.log(source.SOXX / close)
    consumer = np.log(alt.RSPD / alt.RSPS)
    features = pd.DataFrame({
        "gap26": qqq - qqq.rolling(26).max(),
        "change4": qqq.diff(4), "change13": qqq.diff(13), "change26": qqq.diff(26),
        "below10_share": below_fast.rolling(4).mean(),
        "vol13": qqq.diff().rolling(13).std(),
        "chips4": chips.diff(4), "chips13": chips.diff(13),
        "consumer4": consumer.diff(4), "consumer13": consumer.diff(13),
        "breadth13": np.log(source.RSP / source.SPY).diff(13),
        "credit4": np.log(weekly.HYG / weekly.IEI).diff(4),
        "vix_curve": np.log(weekly.VIX / weekly.VIX3M),
        "yield13": source["^TNX"].diff(13),
    })
    state = pd.DataFrame({
        "close": close, "high26": close.rolling(26).max(), "sma10": fast_average,
        "fast_off": below_fast.eq(True) & qqq.diff(4).lt(0),
        "repair": above_fast & above_fast.shift(fill_value=False),
    })
    return features, state


def labels(state, horizon, persistent_weeks):
    result = pd.DataFrame(index=state.index, columns=["persistent", "weeks_under"], dtype=float)
    result["settled"] = pd.NaT
    for i in range(len(state) - horizon):
        high = state.high26.iloc[i]
        if not np.isfinite(high):
            continue
        path = state.close.iloc[i + 1:i + horizon + 1] / high - 1
        weeks_under = int(path.le(-0.05).sum())
        result.iloc[i] = float(weeks_under >= persistent_weeks and path.min() <= -0.10), weeks_under, path.index[-1]
    return result


def train(data, outcome):
    valid = data[[*PRICE, *CROSS]].notna().all(axis=1)
    result = pd.DataFrame(index=data.index, columns=["base_rate", "gap_only", "price", "cross"], dtype=float)
    for date in data.index[valid]:
        past = valid & outcome.settled.le(date)
        target = outcome.loc[past, "persistent"].astype(int)
        if len(target) < 156 or min(target.sum(), (1 - target).sum()) < 12:
            continue
        result.at[date, "base_rate"] = target.mean()
        for name, columns in (("gap_only", ["gap26"]), ("price", list(PRICE)), ("cross", [*PRICE, *CROSS])):
            x, query = standardize(data.loc[past, columns].to_numpy(), data.loc[date, columns].to_numpy())
            model = LogisticRegression(C=0.30, max_iter=500).fit(x, target)
            result.at[date, name] = model.predict_proba(query[None])[0, 1]
    return result


def signal_metrics(outcome, scores, gap, alerts):
    known = scores.notna().all(axis=1) & outcome.persistent.notna()
    groups = {"all": known, "near_high": known & gap.ge(np.log(0.97)),
              "under_3_to_10": known & gap.lt(np.log(0.97)) & gap.ge(np.log(0.90)),
              "under_10_plus": known & gap.lt(np.log(0.90))}
    result = {}
    for group, mask in groups.items():
        target = outcome.loc[mask, "persistent"]
        metrics = {"weeks": len(target), "positives": int(target.sum())}
        for name, forecast in scores.loc[mask].items():
            chosen = mask & alerts[name] if name in alerts else pd.Series(False, index=mask.index)
            metrics[name] = {
                "brier": float(np.square(forecast - target).mean()),
                "auc": float(roc_auc_score(target, forecast)) if target.nunique() == 2 else None,
                "alerts": int(chosen.sum()), "alert_positives": int(outcome.loc[chosen, "persistent"].sum()),
            }
        result[group] = metrics
    return result


def execution_state(history, weekly):
    return weekly.reindex(history.index).ffill().shift()


def allocation(base, state, name):
    target = []
    waiting = False
    previously_invested = False
    reentry_only = name in ("technical_reentry", "price_reentry", "cross_reentry")
    risk_column = "price_high" if name.startswith("price") else "cross_high"
    for invested, row in zip(base.to_numpy(), state.itertuples(), strict=True):
        active = row.ready == True
        high = row.fast_off == True if name.startswith("technical") else getattr(row, risk_column) == True
        repaired = row.repair == True
        if not active or not invested:
            waiting = False
        elif not previously_invested:
            waiting = not repaired if name.startswith("technical") else high
        elif not reentry_only and high and row.fast_off == True:
            waiting = True
        elif waiting and not high and repaired:
            waiting = False
        target.append(float(bool(invested) and not waiting))
        previously_invested = bool(invested)
    return pd.Series(target, index=base.index)


def trade_comparison(history, state, alerts, ready, bps, start):
    weekly = state[["fast_off", "repair"]].assign(
        ready=ready, price_high=alerts.price, cross_high=alerts.cross,
    )
    aligned = execution_state(history, weekly)
    base = price_signals(history).trend
    rules = {"hold": pd.Series(1.0, index=history.index), "sma175": base}
    for name in ("technical_reentry", "price_reentry", "cross_reentry", "technical_cycle", "price_cycle", "cross_cycle"):
        rules[name] = allocation(base, aligned, name)
    prices = history.loc[start:]
    reference, _ = replay(prices, base.loc[start:], bps)
    base_returns = daily_returns(reference.nav)
    recent_peak = prices.qqqClose.tail(63).idxmax()
    result = {}
    for name, target in rules.items():
        frame, fills = replay(prices, target.loc[start:], bps)
        returns = daily_returns(frame.nav)
        include = ~prices.index.year.isin([2020, 2022])
        recent = target.loc[recent_peak:]
        exits = recent.index[recent.eq(0)]
        first_exit = exits[0] if len(exits) else None
        later_risk = recent.loc[first_exit:].gt(0).sum() if first_exit is not None else 0
        result[name] = {
            **path_metrics(frame, fills),
            "sharpe_without_2020_2022": float(sharpe(returns[include])),
            "sharpe_delta_interval": paired_interval(returns, base_returns),
            "latest_target": float(target.iloc[-1]),
            "recent_peak": str(recent_peak.date()),
            "return_since_recent_peak": float(frame.nav.iloc[-1] / frame.nav.at[recent_peak] - 1),
            "first_cash_day_since_peak": str(first_exit.date()) if first_exit is not None else None,
            "days_reinvested_since_first_cash": int(later_risk),
        }
    return result


def reentry_opportunities(history, state, alerts, ready, start, bps):
    base = price_signals(history).trend
    weekly = state[["repair"]].assign(ready=ready, price_high=alerts.price, cross_high=alerts.cross)
    available = execution_state(history, weekly)
    rows = []
    cost = np.log((1 + bps / 10_000) / (1 - bps / 10_000))
    starts = base.eq(1) & base.shift().eq(0) & available.ready.eq(True)
    for date in base.index[starts & (base.index >= start)]:
        i = base.index.get_loc(date)
        later = np.flatnonzero(base.iloc[i + 1:i + 21].eq(0).to_numpy())
        end = i + 1 + later[0] if len(later) else i + 20
        if end >= len(base):
            continue
        buy_edge = np.log(history.tqqqOpen.iloc[end] / history.tqqqOpen.iloc[i]) - cost
        rows.append({
            "date": str(date.date()), "end": str(base.index[end].date()),
            "buy_net_log_edge": float(buy_edge),
            "cross_blocks": bool(available.at[date, "cross_high"]),
            "price_blocks": bool(available.at[date, "price_high"]),
        })
    return rows


def wave_checks(history, alerts, scores):
    output = {}
    for name, (start, end) in PEAK_WINDOWS.items():
        peak = history.qqqClose.loc[start:end].idxmax()
        if pd.isna(peak):
            continue
        before = alerts.loc[:peak].tail(4)
        after = alerts.loc[peak + pd.offsets.Day(1):].head(4)
        output[name] = {
            "peak": str(peak.date()),
            "before": [str(d.date()) for d in before.index[before.cross]],
            "after": [str(d.date()) for d in after.index[after.cross]],
            "pre_peak_score": float(scores.loc[:peak, "cross"].iloc[-1]),
        }
    return output


def evaluate(history, data, state, horizon, weeks):
    outcome = labels(state, horizon, weeks)
    scores = train(data, outcome)
    cutoff = scores[["price", "cross"]].shift().rolling(104, min_periods=52).quantile(0.80)
    ready = cutoff.notna().all(axis=1) & scores.notna().all(axis=1)
    alerts = scores[["price", "cross"]].gt(cutoff) & ready.to_numpy()[:, None]
    start = ready[ready].index[0]
    last = data.join(state).join(scores.add_prefix("p_")).join(cutoff.add_prefix("threshold_"))
    last = last.join(alerts.add_prefix("alert_")).loc["2026-09-01":]
    return {
        "start": str(start.date()),
        "forecast": signal_metrics(outcome, scores, data.gap26, alerts),
        "portfolio": {bps: trade_comparison(history, state, alerts, ready, bps, start) for bps in (25, 50)},
        "reentry_opportunities": reentry_opportunities(history, state, alerts, ready, start, 25),
        "waves": wave_checks(history, alerts, scores),
        "latest": json.loads(last.reset_index().to_json(orient="records", date_format="iso")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-11-06")
    for name in ("history-cache", "context-cache", "alternate-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).loc[:args.end]

    history, context, alternate = map(read, (args.history_cache, args.context_cache, args.alternate_cache))
    data, state = weekly_features(history, context, alternate)
    full = train(data, labels(state, 13, 8))
    for cutoff in ("2018-12-21", "2020-03-20", "2022-10-14", "2025-04-04"):
        p, s = weekly_features(history.loc[:cutoff], context.loc[:cutoff], alternate.loc[:cutoff])
        pd.testing.assert_frame_equal(p, data.loc[:cutoff])
        pd.testing.assert_frame_equal(train(p, labels(s, 13, 8)), full.loc[:cutoff])
    inputs = {name: hashlib.sha256(frame.to_csv().encode()).hexdigest() for name, frame in
              (("market", history), ("context", context), ("alternate", alternate))}
    print(json.dumps({"input_sha256": inputs, "results": {
        str(horizon): evaluate(history, data, state, horizon, weeks) for horizon, weeks in ((13, 8), (8, 5))
    }}, indent=2))


if __name__ == "__main__":
    main()
