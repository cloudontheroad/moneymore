"""Compare simple and learned TQQQ sizing with causal, executable labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_execution import ROOT, bootstrap, metrics
from review_tqqq_signal_edge import raw_signals, targets

from moneymore.leveraged_etf import DATA_SYMBOLS, LeveragedEtfPaper
from moneymore.leveraged_etf_v3 import v3_target

HORIZON = 5
CANDIDATES = ("vol_budget", "trend_budget", "learned_budget")


def monthly_forecast(features: pd.DataFrame, label: pd.Series) -> tuple[pd.Series, list[dict]]:
    """Fit a small ridge model monthly, using only labels mature at the fit close."""
    predictions = np.full(len(features), np.nan)
    fit_log = []
    x = features.to_numpy()
    y = label.to_numpy()
    months = features.index.to_period("M")
    model = None
    for i, date in enumerate(features.index):
        if i == 0 or months[i] != months[i - 1]:
            last = i - HORIZON - 1
            first = max(0, last - 755)
            if last >= first:
                tx, ty = x[first:last + 1], y[first:last + 1]
                valid = np.isfinite(tx).all(axis=1) & np.isfinite(ty)
                if valid.sum() >= 252:
                    tx, ty = tx[valid], ty[valid]
                    center = tx.mean(axis=0)
                    scale = np.maximum(tx.std(axis=0), 1e-8)
                    z = np.clip((tx - center) / scale, -3, 3)
                    mean = np.clip(ty, -3, 3).mean()
                    penalty = 0.5 * len(z) * np.eye(z.shape[1])
                    beta = np.linalg.solve(z.T @ z + penalty, z.T @ (np.clip(ty, -3, 3) - mean))
                    model = center, scale, beta
                    fit_log.append({
                        "fit_close": str(date.date()), "latest_label_exit": str(features.index[last + HORIZON + 1].date()),
                        "rows": int(valid.sum()),
                    })
        if model is not None and np.isfinite(x[i]).all():
            center, scale, beta = model
            predictions[i] = np.clip((x[i] - center) / scale, -3, 3) @ beta
    return pd.Series(predictions, index=features.index), fit_log


def signal_inputs(data, dates):
    def column(symbol, key="close"):
        return pd.Series(data[symbol][key].to_numpy(dtype=float), index=dates)

    qqq, tqqq, opening = column("QQQ"), column("TQQQ"), column("TQQQ", "open")
    daily = tqqq.pct_change()
    sigma20 = daily.rolling(20).std()
    sigma = pd.concat([daily.ewm(span=n, min_periods=n).std() for n in (10, 60)], axis=1).max(axis=1)
    term = column("^VIX") / column("^VIX3M").replace(0, np.nan)
    ema100 = qqq.ewm(span=100, min_periods=100).mean()
    features = pd.DataFrame({
        "trend": qqq / ema100 - 1,
        "momentum_5": qqq.pct_change(5),
        "momentum_20": qqq.pct_change(20),
        "drawdown": qqq / qqq.rolling(63).max() - 1,
        "vol_ratio": daily.rolling(5).std() / sigma20,
        "fear_term": term,
    })
    # Close t decides a trade for open t+1; its 5-session payoff ends at open t+6.
    label = (opening.shift(-(HORIZON + 1)) / opening.shift(-1) - 1) / (sigma20 * np.sqrt(HORIZON))
    score, fit_log = monthly_forecast(features, label)
    # Structural sizing is deliberately fixed. Learned score changes only the
    # risk budget, and can neither borrow nor take an inverse position.
    annual_vol = sigma * np.sqrt(252)
    neutral_budget = 0.40
    trend_budget = pd.Series(np.where(qqq > ema100, 0.52, 0.25), index=dates)
    learned_budget = neutral_budget + (0.60 * score.fillna(0)).clip(-0.20, 0.20)
    desired = pd.DataFrame({
        "vol_budget": neutral_budget / annual_vol,
        "trend_budget": trend_budget / annual_vol,
        "learned_budget": learned_budget / annual_vol,
    }).clip(0, 1)
    return desired, fit_log


def execute_targets(raw: pd.Series, start: int, end: int, band: float = 0.05) -> pd.DataFrame:
    out = np.full(len(raw), np.nan)
    prior = None
    for i in range(start, end + 1):
        if np.isfinite(raw.iat[i]):
            prior = v3_target(raw.iat[i], prior, band)
            out[i] = prior
    return pd.DataFrame({"tqqq": out, "regime": raw.name})


def aggregate(folds: list[pd.DataFrame], cash: float = 0.0) -> dict:
    daily = pd.concat([frame.iloc[1:].copy() for frame in folds], ignore_index=True)
    returns = np.concatenate([frame.normalized_nav.pct_change().dropna().to_numpy() for frame in folds])
    daily["normalized_nav"] = np.cumprod(1 + returns)
    baseline = pd.DataFrame({"normalized_nav": [1.0]})
    return metrics(pd.concat([baseline, daily], ignore_index=True), cash)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/leveraged-etf")
    args = parser.parse_args()
    hashes, inputs = {}, {}
    for symbol in DATA_SYMBOLS:
        payload = (args.cache / f"{symbol.lstrip('^')}.csv").read_bytes()
        hashes[symbol] = hashlib.sha256(payload).hexdigest()
        frame = pd.read_csv(BytesIO(payload))
        inputs[symbol] = frame.loc[frame.trade_date <= args.as_of]
    data = LeveragedEtfPaper._align(inputs)
    dates = pd.DatetimeIndex(data["QQQ"].trade_date)
    desired, fit_log = signal_inputs(data, dates)
    # Check that truncating the input after a decision never changes its signal.
    for date in ("2022-07-07", "2023-01-09", "2025-04-04"):
        stop = dates.searchsorted(date, side="right")
        if 0 < stop < len(dates):
            truncated, _ = signal_inputs({symbol: frame.iloc[:stop] for symbol, frame in data.items()}, dates[:stop])
            pd.testing.assert_frame_equal(truncated, desired.iloc[:stop])
    signals = raw_signals(data, dates)

    def replay(target, start, end, bps=6, cash=0.0, *, hold=False):
        rows = LeveragedEtfPaper.replay(
            data, target, list(range(start, end + 1)), "BASELINE",
            slippage=(bps - 1) / 10_000, cash_yield=cash, buy_hold=hold,
        )
        return pd.DataFrame(rows).set_index("trade_date")

    def fixed(weight):
        return pd.DataFrame({"tqqq": weight, "regime": "STATIC"}, index=range(len(dates)))

    def match_weight(vol, start, end):
        low, high = 0.0, 1.0
        for _ in range(12):
            weight = (low + high) / 2
            if metrics(replay(fixed(weight), start, end))["volatility"] < vol:
                low = weight
            else:
                high = weight
        return (low + high) / 2

    folds, scenario_folds = {}, {}
    for year in range(2022, dates[-1].year + 1):
        start = int(dates.searchsorted(f"{year}-01-01"))
        end = int(dates.searchsorted(f"{year + 1}-01-01")) - 1
        if end <= start:
            break
        train_start = int(dates.searchsorted(f"{year - 2}-01-01"))
        fitted_weights = {}
        strategies = {name: targets(signals, start, end, name) for name in ("v2", "v3")}
        for name in CANDIDATES:
            train = execute_targets(desired[name], train_start, start - 1)
            vol = metrics(replay(train, train_start, start - 1))["volatility"]
            fitted_weights[name] = match_weight(vol, train_start, start - 1)
            strategies[name] = execute_targets(desired[name], start, end)
            strategies[name + "_static"] = fixed(fitted_weights[name])
        strategies["hold"] = fixed(1.0)
        strategies["fixed50"] = fixed(0.50)
        for bps, cash in ((6, 0.0), (25, 0.0), (6, 0.03)):
            case = (bps, cash)
            daily = {name: replay(target, start, end, bps, cash, hold=name == "hold")
                     for name, target in strategies.items()}
            scenario_folds.setdefault(case, []).append(daily)
            if case == (6, 0.0):
                folds[year] = {"matched_weights": fitted_weights, "metrics": {
                    name: metrics(frame) for name, frame in daily.items()
                }}
    summary = {
        f"{bps}bps_cash{cash:g}": {
            name: aggregate([fold[name] for fold in daily_folds], cash)
            for name in daily_folds[0]
        }
        for (bps, cash), daily_folds in scenario_folds.items()
    }
    baseline_folds = scenario_folds[6, 0.0]
    intervals = {}
    for name in CANDIDATES:
        refs = ("hold", name + "_static", "v3")
        if name != "vol_budget":
            refs += ("vol_budget",)
        intervals[name] = {
            ref: bootstrap([np.log(fold[name].normalized_nav / fold[ref].normalized_nav)
                            .diff().dropna().to_numpy() for fold in baseline_folds])
            for ref in refs
        }
    checks = {}
    base, stress = summary["6bps_cash0"], summary["25bps_cash0"]
    for name in CANDIDATES:
        checks[name] = {
            "stress_annualized_excess_vs_hold_at_least_2pp": stress[name]["annualized"] - stress["hold"]["annualized"] >= 0.02,
            "sharpe_gain_vs_matched_static_at_least_015": base[name]["sharpe"] - base[name + "_static"]["sharpe"] >= 0.15,
            "no_less_return_and_no_deeper_drawdown_vs_v3": (
                base[name]["annualized"] >= base["v3"]["annualized"]
                and base[name]["max_drawdown"] >= base["v3"]["max_drawdown"]
            ),
            "positive_excess_interval_vs_hold_and_matched_static": all(
                intervals[name][ref][0] > 0 for ref in ("hold", name + "_static")
            ),
        }
    print(json.dumps({
        "as_of": str(dates[-1].date()), "input_sha256": hashes, "monthly_fit_audit": fit_log,
        "folds": folds, "summary": summary, "annual_log_excess_intervals": intervals,
        "research_gates": {name: {"checks": gates, "passed": all(gates.values())} for name, gates in checks.items()},
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
