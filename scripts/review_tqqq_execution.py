"""Replay frozen TQQQ candidates, fit risk budgets on past data, and report held-out results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneymore.leveraged_etf import DATA_SYMBOLS, LeveragedEtfPaper
from moneymore.leveraged_etf_v3 import LeveragedEtfV3Paper
from moneymore.tqqq_research import fast_defense, layered_components, layered_target

BUDGETS = (0.35, 0.45, 0.55)
CANDIDATES = ("v2_fast", "v3_layered")


def metrics(daily: pd.DataFrame, cash_yield: float = 0.0) -> dict:
    nav = daily.normalized_nav
    returns = nav.pct_change().dropna()
    excess = returns - ((1 + cash_yield) ** (1 / 252) - 1)
    deviation = returns.std()
    return {
        "return": float(nav.iloc[-1] - 1),
        "annualized": float(nav.iloc[-1] ** (252 / len(returns)) - 1),
        "sharpe": float(np.sqrt(252) * excess.mean() / deviation) if deviation > 0 else None,
        "volatility": float(np.sqrt(252) * deviation),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "actual_exposure": float(daily.actual_tqqq.mean()),
        "fills": int(daily.fill_count.sum()),
        "turnover": float(daily.turnover.sum()),
        "cost_usd": float(daily.cost.sum()),
    }


def relative_attribution(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict:
    candidate_return = np.log(candidate.normalized_nav).diff().dropna()
    reference_return = np.log(reference.normalized_nav).diff().dropna()
    active = candidate_return - reference_return
    overnight = np.log1p(candidate.overnight_return) - np.log1p(reference.overnight_return)
    return {
        "up_days_log_excess": float(active[reference_return > 0].sum()),
        "down_days_log_excess": float(active[reference_return < 0].sum()),
        "overnight_log_excess": float(overnight.sum()),
        "session_and_cost_log_excess": float(active.sum() - overnight.sum()),
    }


def bootstrap(folds: list[np.ndarray]) -> list[float]:
    rng = np.random.default_rng(20261109)
    sums = np.zeros(2000)
    for values in folds:
        length = len(values)
        starts = rng.integers(0, length, size=(len(sums), (length + 19) // 20))
        offsets = (starts[..., None] + np.arange(20)) % length
        sums += values[offsets.reshape(len(sums), -1)[:, :length]].sum(axis=1)
    annual_log = sums * 252 / sum(map(len, folds))
    return np.quantile(annual_log, [0.025, 0.5, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="2026-11-06")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/leveraged-etf")
    args = parser.parse_args()
    paths = {symbol: args.cache / f"{symbol.lstrip('^')}.csv" for symbol in DATA_SYMBOLS}
    inputs = {}
    hashes = {}
    for symbol, path in paths.items():
        raw = path.read_bytes()
        hashes[symbol] = hashlib.sha256(raw).hexdigest()
        frame = pd.read_csv(BytesIO(raw))
        inputs[symbol] = frame.loc[frame.trade_date <= args.as_of]
    data = LeveragedEtfPaper._align(inputs)
    dates = pd.Index(data["QQQ"].trade_date.astype(str))
    qqq = data["QQQ"].close
    defense = fast_defense(qqq)
    components = layered_components(qqq, data["TQQQ"].open, data["TQQQ"].close)
    v2 = LeveragedEtfPaper._backtest_targets(data)
    v3 = LeveragedEtfV3Paper._backtest_targets(data)
    first_valid = dates[v2.tqqq.first_valid_index()]

    def replay(weights, start, end, bps=6, cash_yield=0.0, hold=False, drift_band=0.0):
        indices = np.flatnonzero((dates >= start) & (dates <= end)).tolist()
        if len(indices) < 2:
            raise ValueError(f"Insufficient sessions for {start} to {end}")
        target = pd.DataFrame({"tqqq": weights, "regime": "RESEARCH"}, index=pd.RangeIndex(len(dates)))
        return pd.DataFrame(LeveragedEtfPaper.replay(
            data, target, indices, "BASELINE", slippage=(bps - 1) / 10_000,
            cash_yield=cash_yield, buy_hold=hold, drift_band=drift_band,
        )).set_index("trade_date")

    stress = {}
    folds = {}
    chain = {}
    paired = {name: {"full": [], "risk_matched": []} for name in CANDIDATES}
    weights_by_year = {}
    for year in range(2022, int(args.as_of[:4]) + 1):
        start, end = f"{year}-01-01", min(f"{year}-12-31", args.as_of)
        train_start = max(first_valid, f"{year - 3}-01-01")
        train_end = f"{year - 1}-12-31"
        train_hold = metrics(replay(1.0, train_start, train_end, hold=True))["volatility"]
        budget = max(BUDGETS, key=lambda value: metrics(replay(
            layered_target(components, value, recovery=True), train_start, train_end,
        ))["sharpe"])
        first = int(dates.searchsorted(start))
        targets = {
            "v2": LeveragedEtfPaper._backtest_targets(data, first).tqqq,
            "v3": LeveragedEtfV3Paper._backtest_targets(data, first).tqqq,
            "v2_fast": LeveragedEtfPaper._backtest_targets(data, first).tqqq * defense,
            "v3_trend": components.trend,
            "v3_trend_risk": layered_target(components, budget, recovery=False),
            "v3_layered": layered_target(components, budget, recovery=True),
            "full": 1.0,
        }
        for name, training_target in (
            ("v2_fast", v2.tqqq * defense),
            ("v3_layered", targets["v3_layered"]),
        ):
            trained_vol = metrics(replay(training_target, train_start, train_end))["volatility"]
            targets[f"{name}_risk_match"] = min(1.0, trained_vol / train_hold)
        weights_by_year[year] = targets
        daily = {name: replay(w, start, end, hold=name == "full") for name, w in targets.items()}
        for name in ("v2", "v3"):
            for band in (0.01, 0.02, 0.05):
                daily[f"{name}_drift_{band:.0%}"] = replay(targets[name], start, end, drift_band=band)
        for name, frame in daily.items():
            chain.setdefault(name, []).append(frame.normalized_nav.pct_change().dropna())
        folds[year] = {
            "train": [train_start, train_end],
            "test": [start, daily["full"].index[-1]],
            "risk_budget": budget,
            "static_weights": {name: w for name, w in targets.items() if name.endswith("_match")},
            "metrics": {name: metrics(frame) for name, frame in daily.items()},
            "attribution": {
                name: relative_attribution(daily[name], daily[base])
                for name, base in (("v2_fast", "v2"), ("v3_layered", "v3"))
            },
        }
        for name in CANDIDATES:
            for label, ref in (("full", "full"), ("risk_matched", f"{name}_risk_match")):
                active = np.log(daily[name].normalized_nav / daily[ref].normalized_nav).diff().dropna()
                paired[name][label].append(active.to_numpy())
        for bps, rate in ((10, 0.0), (25, 0.0), (6, 0.03)):
            stress[f"{year}/{bps}bps/cash{rate}"] = {
                name: metrics(replay(targets[name], start, end, bps, rate, name == "full"), rate)
                for name in (*CANDIDATES, "full", *(f"{name}_risk_match" for name in CANDIDATES))
            }

    crises = {}
    for name, start, end in (
        ("2022_rates", "2022-01-03", "2022-12-30"),
        ("2025_selloff_rebound", "2025-02-19", "2025-06-30"),
    ):
        targets = weights_by_year[int(start[:4])]
        crises[name] = {
            strategy: metrics(replay(w, start, end, hold=strategy == "full"))
            for strategy, w in targets.items()
        }
    episodes = []
    entry = None
    for index in range(len(dates)):
        if defense.iloc[index] < 1 and entry is None:
            entry = index
        elif defense.iloc[index] == 1 and entry is not None:
            if dates[entry] >= "2022-01-01" and index + 1 < len(dates):
                sell_open = float(data["TQQQ"].open.iloc[entry + 1])
                buy_open = float(data["TQQQ"].open.iloc[index + 1])
                trough = float(data["TQQQ"].low.iloc[entry + 1:index + 1].min())
                episodes.append({
                    "reduce_open": dates[entry + 1], "restore_open": dates[index + 1],
                    "repurchase_premium": buy_open / sell_open - 1,
                    "underlying_decline_to_trough": trough / sell_open - 1,
                    "underlying_rebound_to_repurchase": buy_open / trough - 1,
                })
            entry = None
    intervals = {
        name: {label: bootstrap(values) for label, values in comparisons.items()}
        for name, comparisons in paired.items()
    }
    gates = {}
    for name, parent in (("v2_fast", "v2"), ("v3_layered", "v3")):
        checks = {
            "positive_excess_lower_bound_vs_full": intervals[name]["full"][0] > 0,
            "positive_excess_lower_bound_vs_risk_match": intervals[name]["risk_matched"][0] > 0,
            "sharpe_improves_vs_parent_in_majority": sum(
                fold["metrics"][name]["sharpe"] > fold["metrics"][parent]["sharpe"]
                for fold in folds.values()
            ) > len(folds) / 2,
            "no_worse_2022_drawdown_vs_parent": (
                crises["2022_rates"][name]["max_drawdown"] >= crises["2022_rates"][parent]["max_drawdown"]
            ),
        }
        gates[name] = {"checks": checks, "passed": all(checks.values())}
    chained_return = {name: float((1 + pd.concat(parts)).prod() - 1) for name, parts in chain.items()}
    print(json.dumps({
        "as_of": dates[-1], "input_start": dates[0], "first_v2_signal": first_valid,
        "input_sha256": hashes, "folds": folds, "cost_and_cash_stress": stress,
        "crises": crises, "fast_defense_episodes": episodes,
        "annual_log_excess_bootstrap": intervals,
        "chained_fold_returns": chained_return,
        "research_gate": gates,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
