"""Frozen open-to-open research replay with matched stock-selection controls."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_us_corrected_market_hurdle as market
import train_us_overnight_lightgbm as ranker
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/us_open2open_v1.json"
LABEL = "label_next_open_net"


def training_panel():
    panel = pd.read_parquet(market.PANEL)
    if "label_end" not in panel:
        ends = pd.read_parquet(
            market.REPORT_DIR / "investability_eligibility_history.parquet",
            columns=["timestamp", "symbol", "label_end"],
        )
        panel = panel.merge(ends, on=["timestamp", "symbol"], validate="one_to_one", how="left")
    if panel.label_end.isna().any() or (panel.label_end <= panel.timestamp).any():
        raise ValueError("Invalid label settlement timestamps")
    return panel


def daily_context(panel):
    fields = [c for c in panel if c.startswith(("market_", "breadth_", "calendar_"))]
    daily = panel.groupby("timestamp").agg(
        {**dict.fromkeys(fields, "first"), LABEL: "mean", "label_end": "max"}
    )
    return daily, fields


def settled_days(daily, cutoff, count):
    return daily.loc[(daily.index < cutoff) & (daily.label_end < cutoff)].tail(count)


def fit_ranker(panel, daily, cutoff, spec):
    ranker.ACTIVE_FEATURE_COLUMNS = spec["stock_features"]
    ranker.ACTIVE_OBJECTIVE = "huber"
    ranker.ACTIVE_TARGET_COLUMN = LABEL
    dates = settled_days(daily, cutoff, spec["stock_train_sessions"]).index
    return ranker._fit(panel[panel.timestamp.isin(dates)], **spec["stock_parameters"])


def fit_market(daily, fields, cutoff, spec):
    train = settled_days(daily, cutoff, spec["market_train_sessions"])
    estimator = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1000),
    )
    return estimator.fit(train[fields], train[LABEL] * 10_000)


def metrics(values):
    series = pd.Series(values, dtype=float)
    equity = (1 + series).cumprod()
    equity = pd.concat([pd.Series([1.0]), equity], ignore_index=True)
    return market.stats(series) | {
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    panel = training_panel()
    daily, fields = daily_context(panel)
    daily_positions = {date: i for i, date in enumerate(daily.index)}
    groups = panel.groupby("timestamp")
    dates = daily.index[-252:]
    rng = np.random.default_rng(621)
    rows, simulations = [], []
    last_stock = last_market = None
    for date in dates:
        i = daily_positions[date]
        stock_anchor = daily.index[i - i % spec["stock_retrain"]]
        market_anchor = daily.index[i - i % spec["market_retrain"]]
        if stock_anchor != last_stock:
            stock = fit_ranker(panel, daily, stock_anchor, spec)
            last_stock = stock_anchor
        if market_anchor != last_market:
            gate = fit_market(daily, fields, market_anchor, spec)
            last_market = market_anchor
        forecast = gate.predict(daily.loc[[date], fields])[0]
        day = groups.get_group(date).copy()
        day["score"] = stock.predict(day[spec["stock_features"]])
        top = day.nlargest(10, "score")
        reversal = day.nsmallest(10, "ret_3d")
        active = forecast > spec["market_threshold_bps"]
        row = {"timestamp": date.isoformat(), "active": bool(active)}
        for name, basket in (("ranker", top), ("reversal", reversal)):
            for n in (1, 10):
                row[f"{name}_{n}"] = float(basket[LABEL].iloc[:n].mean())
        rows.append(row)
        # Neighbor controls match the pre-open cross-sectional volatility and
        # turnover ranks of each selected stock, excluding the selected names.
        pool = day.loc[~day.symbol.isin(top.symbol)].dropna(
            subset=["realized_vol_20", "log_dollar_volume"]
        )
        dimensions = ["realized_vol_20", "log_dollar_volume"]
        distance = np.square(
            top[dimensions].to_numpy()[:, None, :] - pool[dimensions].to_numpy()[None, :, :]
        ).sum(axis=2)
        nearest = np.argsort(distance, axis=1, kind="stable")
        draws = np.empty((10, 1000))
        for trial in range(draws.shape[1]):
            used = set()
            for slot, neighbors in enumerate(nearest):
                choices = [index for index in neighbors[:30] if index not in used]
                chosen = rng.choice(choices[:20])
                used.add(chosen)
                draws[slot, trial] = pool[LABEL].iloc[chosen]
        simulations.append(np.stack([draws[0], draws.mean(axis=0)]) * active)

    frame = pd.DataFrame(rows)
    trials = np.array(simulations)
    report = {
        "strategy_id": spec["id"], "sessions": len(frame),
        "historical_membership": "retrospective_current_snapshot_only",
        "first_session": rows[0]["timestamp"], "last_session": rows[-1]["timestamp"],
        "traded_sessions": int(frame.active.sum()), "matched_trials": trials.shape[2],
        "results": {}, "daily": rows,
    }
    for n, slot in ((1, 0), (10, 1)):
        active = frame.active
        values = frame[f"ranker_{n}"]
        gated = values.where(active, 0)
        control_means = trials[:, slot].mean(axis=0) * 10_000
        report["results"][str(n)] = {
            "ranker": metrics(values),
            "gated": metrics(gated) | {"trade_win_rate": float((values[active] > 0).mean())},
            "gated_last126": metrics(gated.iloc[-126:]),
            "reversal_gated": metrics(frame[f"reversal_{n}"].where(active, 0)),
            "matched_random_mean_bps_percentiles": np.percentile(control_means, [5, 50, 95]).tolist(),
            "fraction_of_matched_trials_beaten": float((control_means < gated.mean() * 1e4).mean()),
            "gated_cost_stress": {
                str(bps): metrics(gated - active.astype(float) * (bps - 20) / 10_000)
                for bps in (30, 50)
            },
        }
    path = market.REPORT_DIR / "open2open_v1_review_replay.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
