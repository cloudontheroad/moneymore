"""Compare three independently implemented TQQQ swing approaches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from review_tqqq_execution import bootstrap
from review_tqqq_external import daily_prices, summary
from review_tqqq_switch_cost import leverage_rules

START = "2013-01-02"
COSTS = (10, 25)
WAVES = {
    "2018": ("2018-09-01", "2019-04-30"),
    "2020": ("2020-02-01", "2020-06-30"),
    "2022": ("2022-01-01", "2023-04-30"),
    "2025": ("2025-02-01", "2025-06-30"),
}


def monthly_states(close, rates):
    periods = close.index.to_period("M")
    monthly = close.groupby(periods).last()
    # Use only monthly rates old enough to have been published before the month began.
    cash = rates.reindex(monthly.index - 2).to_numpy() / 12
    excess = np.log(monthly).diff() - np.log1p(cash)
    fast = excess.gt(0)
    slow = excess.rolling(12).sum().gt(0)
    allocation = pd.Series(np.select([fast, slow], [1.0, 0.5], default=0.0), index=monthly.index)
    # A month's signal first becomes tradeable at next month's opening session.
    targets = {
        "four_states": allocation.shift(),
        "slow_only": slow.astype(float).shift(),
    }
    return {
        name: pd.Series(weights.reindex(periods).to_numpy(), index=close.index)
        for name, weights in targets.items()
    }


def weekly_slow_weights(close):
    weeks = pd.Series(close.index.to_period("W-FRI"), index=close.index)
    momentum = close.shift() / close.shift(253) - 1
    return momentum.gt(0).astype(float).where(momentum.notna() & weeks.ne(weeks.shift())).ffill().fillna(1)


def changepoint_weights(close, *, slow_fallback):
    """Weekly, causal two-mean change detector inspired by the upstream CPD design."""
    returns = np.log(close).diff().to_numpy()
    weights = np.ones(len(close))
    weeks = close.index.to_period("W-FRI")
    slow = weekly_slow_weights(close).to_numpy()
    weight = 1.0
    for i in range(127, len(close)):
        if weeks[i] != weeks[i - 1]:
            window = returns[i - 126:i]
            recent_lengths = np.arange(10, 22)
            split = len(window) - recent_lengths
            cumulative = np.cumsum(window)
            before = cumulative[split - 1] / split
            after = (cumulative[-1] - cumulative[split - 1]) / recent_lengths
            null_sse = ((window - window.mean()) ** 2).sum()
            gain = split * recent_lengths / len(window) * (before - after) ** 2
            # Two extra fitted parameters, including the searched breakpoint.
            if slow_fallback:
                weight = slow[i]
            if null_sse > 0:
                improvement = -len(window) * np.log1p(-np.minimum(gain / null_sse, 1 - 1e-12))
                improvement -= 2 * np.log(len(window))
                best = np.argmax(improvement)
                if improvement[best] > 0 and np.sign(before[best]) != np.sign(after[best]):
                    weight = float(after[best] > 0)
        weights[i] = weight
    return pd.Series(weights, index=close.index)


def replay(prices, weights, bps, *, alternative="cash", rebalance_on=()):
    """Trade only when the target changes. Allocate after costs at the known opening price."""
    targets = weights.to_numpy(dtype=float)
    opens = prices[["qqqOpen", "tqqqOpen"]].to_numpy()
    closes = prices[["qqqClose", "tqqqClose"]].to_numpy()
    shares = np.zeros(2)
    cash = 1.0
    fills = 0
    cost_rate = bps / 10_000
    scheduled = set(rebalance_on)
    rows = []
    for i, date in enumerate(prices.index):
        current = shares * opens[i]
        equity = current.sum() + cash
        prior_weight = current[1] / equity
        turnover = 0.0
        if not i or targets[i] != targets[i - 1] or date in scheduled:
            target = targets[i]
            desired = np.array([1 - target if alternative == "qqq" else 0, target])
            investable = equity
            for _ in range(8):
                position = desired * investable
                trade = np.abs(position - current)
                investable = equity - cost_rate * trade.sum()
            np.testing.assert_allclose(investable + cost_rate * trade.sum(), equity, rtol=1e-12)
            cash = investable - position.sum()
            shares = position / opens[i]
            fills += np.count_nonzero(trade > 1e-12)
            turnover = trade.sum() / equity
        open_nav = cash + (shares * opens[i]).sum()
        weight = shares[1] * opens[i, 1] / open_nav
        rows.append((cash + shares @ closes[i], weight, weight - prior_weight, turnover))
    columns = ["nav", "weight", "weight_change", "turnover"]
    return pd.DataFrame(rows, index=prices.index, columns=columns), int(fills)


def assess(frame, fills, hold, tqqq_open):
    active = np.log(frame.nav / hold.nav).diff()
    active.iloc[0] = np.log(frame.nav.iloc[0] / hold.nav.iloc[0])
    # Use fixed historical wave windows for attribution, then remove the most profitable one.
    waves = {}
    for name, (start, end) in WAVES.items():
        change = frame.weight_change.loc[start:end]
        reduction = change.index[change < -0.05]
        subsequent = change.loc[reduction[0]:] if len(reduction) else change.iloc[:0]
        reentry = subsequent.index[subsequent > 0.05]
        path = {}
        if len(reduction):
            sell_day = reduction[0]
            buy_day = reentry[0] if len(reentry) else tqqq_open.loc[start:end].index[-1]
            peak = tqqq_open.loc[start:sell_day].max()
            trough = tqqq_open.loc[sell_day:buy_day].min()
            path = {
                "peak_to_sell": float(tqqq_open.loc[sell_day] / peak - 1),
                "sell_to_trough": float(trough / tqqq_open.loc[sell_day] - 1),
                "trough_to_reentry": float(tqqq_open.loc[buy_day] / trough - 1) if len(reentry) else None,
            }
        waves[name] = {
            "log_excess": float(active.loc[start:end].sum()),
            "first_reduction": str(reduction[0].date()) if len(reduction) else None,
            "first_increase": next((str(d.date()) for d in change.index[change > 0.05]), None),
            "first_reentry_after_reduction": str(reentry[0].date()) if len(reentry) else None,
            "tqqq_open_path": path,
        }
    periods = {name: float(np.expm1(active.loc[start:end].sum())) for name, start, end in (
        ("2013–2019", "2013", "2019"), ("2020–2022", "2020", "2022"), ("2023+", "2023", None),
    )}
    best_wave = max(waves, key=lambda key: waves[key]["log_excess"])
    rolling = (frame.nav / hold.nav).pct_change(756).dropna()
    return {
        **summary([1, *frame.nav], fills),
        "relative_return": float(np.expm1(active.sum())),
        "relative_by_period": periods,
        "three_year_win_rate": float(rolling.gt(0).mean()),
        "average_tqqq_weight": float(frame.weight.mean()),
        "turnover": float(frame.turnover.sum()),
        "annual_log_excess_interval": bootstrap([active.to_numpy()]),
        "waves": waves,
        "best_wave": best_wave,
        "without_best_wave": float(np.expm1(active.sum() - max(0, waves[best_wave]["log_excess"]))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--end", default="2026-11-06")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        qqq, tqqq = pool.map(lambda symbol: daily_prices(symbol, args.end), ("QQQ", "TQQQ"))
    all_prices = pd.concat([
        qqq[["Open", "Close"]].add_prefix("qqq"), tqqq[["Open", "Close"]].add_prefix("tqqq"),
    ], axis=1, join="inner")
    rates = pd.read_json(args.dashboard / "data/rates-snapshot.json")
    rates = rates.set_index(pd.to_datetime(rates.date).dt.to_period("M")).value
    signals = monthly_states(all_prices.qqqClose, rates)
    signals["weekly_slow"] = weekly_slow_weights(all_prices.qqqClose)
    cpd_variants = {"cpd_slow_fallback": True, "cpd_persistent": False}
    for name, fallback in cpd_variants.items():
        signals[name] = changepoint_weights(all_prices.qqqClose, slow_fallback=fallback)
    signals["buffer"] = ((leverage_rules(all_prices.qqqClose).buffer_1pct - 1) / 2).shift()
    for cutoff in ("2018-12-24", "2020-03-16", "2022-10-13", "2025-04-09"):
        past = all_prices.qqqClose.loc[:cutoff]
        shorter = {
            **monthly_states(past, rates),
            "weekly_slow": weekly_slow_weights(past),
            **{name: changepoint_weights(past, slow_fallback=fallback) for name, fallback in cpd_variants.items()},
        }
        for name, candidate in shorter.items():
            pd.testing.assert_series_equal(candidate, signals[name].loc[:cutoff])
    prices = all_prices.loc[START:]
    payload = {
        "start": START, "end": args.end, "costs": COSTS,
        "prices": dict(zip(prices.index.strftime("%Y-%m-%d"), prices.to_dict("records"), strict=True)),
    }
    bridge = Path(__file__).with_name("review_tqqq_dashboard.cjs")
    proc = subprocess.run(["node", str(bridge), str(args.dashboard)], input=json.dumps(payload),
                          text=True, stdout=subprocess.PIPE, check=True)
    dashboard = json.loads(proc.stdout)
    results = {}
    for bps in COSTS:
        hold, _ = replay(prices, pd.Series(1.0, index=prices.index), bps)
        dash = dashboard["runs"][str(bps)]
        raw = pd.DataFrame(dash["rows"]).set_index("date")
        raw.index = pd.DatetimeIndex(raw.index)
        if not raw.index.equals(prices.index):
            raise ValueError("Dashboard sessions differ from the common comparison")
        results[bps] = {
            "hold": assess(hold, 1, hold, prices.tqqqOpen),
            "dashboard": assess(raw, dash["fills"], hold, prices.tqqqOpen),
        }
        for name, target in signals.items():
            aligned = target.loc[prices.index]
            if aligned.isna().any():
                raise ValueError(f"Missing initial signal for {name}")
            frame, fills = replay(prices, aligned, bps, alternative="qqq" if name == "buffer" else "cash")
            results[bps][name] = assess(frame, fills, hold, prices.tqqqOpen)
        results[bps]["dashboard_actions"] = raw.action.value_counts().to_dict()
    print(json.dumps({
        "start": START, "end": str(prices.index[-1].date()),
        "input_sha256": hashlib.sha256(all_prices.to_csv().encode()).hexdigest(),
        "snapshot": dashboard["snapshot"], "native": dashboard["native"],
        "native_actions": dashboard["nativeActions"], "results": results,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
