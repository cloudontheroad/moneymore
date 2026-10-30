"""Reproduce and audit MoneyMore's cycle-aware TQQQ candidate.

This is research-only.  It never opens the paper database.  Parameters are
chosen on pre-2010 history, frozen, and reported on the 2010+ holdout.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneymore.leveraged_etf import YahooDailyClient
from moneymore.tqqq_cycle import CycleParameters, cycle_decisions

OUTPUT = ROOT / "data" / "research" / "tqqq_cycle"
TRAIN_END = "2010-03-30"
TURNOVER_COST = 0.0006
ANNUAL_DRAG = 0.0095


def download(symbol: str, start: str = "1985-01-01") -> pd.Series:
    end = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    frame = YahooDailyClient(timeout=60).history(symbol, start, end)
    return frame.assign(trade_date=pd.to_datetime(frame.trade_date)).set_index("trade_date")["close"].astype(float)


def load_history() -> pd.DataFrame:
    series = {
        "ndx": download("^NDX"),
        "vix": download("^VIX"),
        "rate": download("^TNX"),
        "vix3m": download("^VIX3M"),
        "tqqq": download("TQQQ", "2010-01-01"),
    }
    frame = pd.concat(series, axis=1).sort_index()
    sessions = frame.index[frame.ndx.notna()]
    frame = frame.reindex(sessions)
    frame[["vix", "rate", "vix3m"]] = frame[["vix", "rate", "vix3m"]].ffill()
    return frame.dropna(subset=["ndx", "vix", "rate"])


def synthetic_tqqq_return(ndx: pd.Series) -> pd.Series:
    return (3.0 * ndx.pct_change().fillna(0.0) - ANNUAL_DRAG / 252).clip(lower=-0.99)


def equity(weight: pd.Series, returns: pd.Series) -> pd.Series:
    aligned = weight.reindex(returns.index).ffill().fillna(0.0)
    turnover = aligned.diff().abs().shift(1).fillna(0.0)
    daily = aligned.shift(1).fillna(0.0) * returns - turnover * TURNOVER_COST
    return (1.0 + daily.clip(lower=-0.99)).cumprod()


def metrics(curve: pd.Series, weight: pd.Series) -> dict[str, float]:
    curve = curve.dropna()
    daily = curve.pct_change().dropna()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1 / 252)
    drawdown = curve / curve.cummax() - 1.0
    return {
        "cagr": float(curve.iloc[-1] ** (1 / years) - 1),
        "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) else 0.0,
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(weight.reindex(curve.index).ffill().diff().abs().mean() * 252),
        "average_exposure": float(weight.reindex(curve.index).ffill().mean()),
    }


def whipsaw_audit(weight: pd.Series, price: pd.Series, horizon: int = 20) -> dict[str, float]:
    """Measure de-risk/re-risk pairs around troughs; future data is audit-only."""
    weight = weight.reindex(price.index).ffill().fillna(0.0)
    changes = weight.diff().fillna(0.0)
    pairs = []
    for offset in np.flatnonzero(changes.to_numpy() < -1e-9):
        later = changes.iloc[offset + 1 : offset + horizon + 1]
        buys = later[later > 1e-9]
        if buys.empty:
            continue
        buy_date = buys.index[0]
        sell_date = price.index[offset]
        window = price.iloc[offset : price.index.get_loc(buy_date) + 1]
        pairs.append(
            {
                "sell_date": sell_date,
                "buy_date": buy_date,
                "sell_price": float(price.loc[sell_date]),
                "buy_price": float(price.loc[buy_date]),
                "repurchase_premium": float(price.loc[buy_date] / price.loc[sell_date] - 1),
                "sold_near_local_bottom": bool(price.loc[sell_date] <= window.min() * 1.02),
            }
        )
    if not pairs:
        return {"whipsaw_pairs": 0, "mean_repurchase_premium": 0.0, "bottom_sell_rate": 0.0}
    table = pd.DataFrame(pairs)
    return {
        "whipsaw_pairs": len(table),
        "mean_repurchase_premium": float(table.repurchase_premium.mean()),
        "bottom_sell_rate": float(table.sold_near_local_bottom.mean()),
    }


def block_bootstrap(returns: pd.Series, samples: int = 2000, block: int = 20) -> pd.DataFrame:
    values = returns.dropna().to_numpy(dtype=float)
    random = np.random.default_rng(42)
    blocks = int(np.ceil(len(values) / block))
    rows = []
    for _ in range(samples):
        starts = random.integers(0, len(values) - block + 1, blocks)
        sample = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
        rows.append({"annual_return": float(np.mean(sample) * 252),
                     "sharpe": float(np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252))})
    return pd.DataFrame(rows)


def candidates() -> list[CycleParameters]:
    return [
        CycleParameters(base_exposure=base, cycle_slope=slope, rebalance_band=band,
                        crash_distance=distance)
        for base, slope, band, distance in itertools.product(
            (0.30, 0.45, 0.60), (0.6, 1.0, 1.4), (0.08, 0.12),
            (-0.20, -0.25, -0.30)
        )
    ]


def run() -> None:
    frame = load_history()
    returns = synthetic_tqqq_return(frame.ndx)
    train = frame.index <= TRAIN_END
    rows = []
    weights = {}
    for params in candidates():
        key = json.dumps(asdict(params), sort_keys=True)
        decision = cycle_decisions(frame.ndx, frame.vix, frame.rate, frame.vix3m, params)
        weight = decision.target_tqqq.fillna(0.0)
        weights[key] = weight
        curve = equity(weight, returns)
        rows.append({"parameters": key, "sample": "train", **metrics(curve.loc[train], weight.loc[train])})
        rows.append({"parameters": key, "sample": "holdout", **metrics(curve.loc[~train] / curve.loc[~train].iloc[0], weight.loc[~train]),
                     **whipsaw_audit(weight.loc[~train], frame.ndx.loc[~train])})

    canonical = CycleParameters()
    canonical_key = json.dumps(asdict(canonical), sort_keys=True)
    weight = weights[canonical_key]
    curve = equity(weight, returns)
    holdout_returns = curve.loc[~train].pct_change()
    bootstrap = block_bootstrap(holdout_returns)
    crisis_rows = []
    for name, start, end in (
        ("dotcom", "2000-03-01", "2002-10-31"),
        ("gfc", "2007-10-01", "2009-03-31"),
        ("covid", "2020-02-01", "2020-04-30"),
        ("rate_bear_2022", "2022-01-01", "2022-12-31"),
    ):
        part = curve.loc[start:end]
        if len(part) > 1:
            crisis_rows.append({"period": name, **metrics(part / part.iloc[0], weight.loc[part.index]),
                                **whipsaw_audit(weight.loc[part.index], frame.ndx.loc[part.index])})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "parameter_train_holdout.csv", index=False)
    pd.DataFrame(crisis_rows).to_csv(OUTPUT / "crisis_audit.csv", index=False)
    bootstrap.to_csv(OUTPUT / "holdout_block_bootstrap.csv", index=False)
    cycle_decisions(frame.ndx, frame.vix, frame.rate, frame.vix3m, canonical).to_csv(
        OUTPUT / "canonical_daily_signals.csv"
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "train_end": TRAIN_END,
        "holdout_start": str(frame.index[~train][0].date()),
        "canonical_parameters": asdict(canonical),
        "bootstrap_sharpe_95pct": bootstrap.sharpe.quantile([0.025, 0.975]).tolist(),
        "production_database_touched": False,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    run()
