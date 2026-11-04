"""Pre-TQQQ falsification for v3 using a transparent 3x QQQ reconstruction."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneymore.leveraged_etf import YahooDailyClient
from moneymore.tqqq_v3 import V3Parameters, signal_frame

OUTPUT = ROOT / "data" / "research" / "tqqq_v3"
ANNUAL_DRAG = 0.0095


def history(symbol: str) -> pd.Series:
    end = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    frame = YahooDailyClient(timeout=60).history(symbol, "1998-01-01", end)
    frame["trade_date"] = pd.to_datetime(frame.trade_date)
    return frame.set_index("trade_date").close.astype(float).sort_index()


def synthetic_tqqq(qqq: pd.Series) -> pd.Series:
    returns = (3.0 * qqq.pct_change() - ANNUAL_DRAG / 252).clip(lower=-0.99).fillna(0.0)
    return 100.0 * (1.0 + returns).cumprod()


def simulate(
    tqqq: pd.Series,
    target: pd.Series,
    cost: float = 0.0006,
    stop: float | None = None,
) -> pd.DataFrame:
    returns = tqqq.pct_change().fillna(0.0)
    desired = target.shift(1).reindex(tqqq.index)
    equity = 1.0
    weight = 1.0
    peak = 1.0
    cash_days = 0
    stopped = False
    rows = []
    for date in tqqq.index:
        requested = float(desired.loc[date]) if pd.notna(desired.loc[date]) else weight
        if stop is not None:
            if stopped:
                cash_days += 1
                if cash_days >= 5 and requested >= 0.80:
                    stopped = False
                    peak = equity
                else:
                    requested = 0.0
            elif equity / peak - 1.0 <= -stop:
                stopped = True
                cash_days = 0
                requested = 0.0
        turnover = abs(requested - weight)
        equity *= max(0.0, 1.0 - cost * turnover)
        weight = requested
        equity *= 1.0 + weight * float(returns.loc[date])
        peak = max(peak, equity)
        rows.append({"trade_date": date, "equity": equity, "weight": weight, "turnover": turnover})
    return pd.DataFrame(rows).set_index("trade_date")


def metrics(result: pd.DataFrame) -> dict[str, float]:
    daily = result.equity.pct_change().dropna()
    years = (result.index[-1] - result.index[0]).days / 365.25
    drawdown = result.equity / result.equity.cummax() - 1.0
    return {
        "cagr": float(result.equity.iloc[-1] ** (1 / years) - 1),
        "sharpe_rf0": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(result.turnover.mean() * 252),
        "average_exposure": float(result.weight.mean()),
    }


def normalized_slice(result: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    part = result.loc[start:end].copy()
    part.equity /= part.equity.iloc[0]
    return part


def main() -> None:
    qqq = history("QQQ")
    tqqq = synthetic_tqqq(qqq)
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    selected_params = V3Parameters(**manifest["selected_parameters"])
    periods = {
        "dotcom": ("2000-03-01", "2002-10-31"),
        "gfc": ("2007-10-01", "2009-03-31"),
        "pre_tqqq_full": ("1999-01-01", "2010-02-10"),
    }
    strategies = {
        "synthetic_buy_hold": simulate(tqqq, pd.Series(1.0, index=tqqq.index)),
        "selected_candidate": simulate(
            tqqq, signal_frame(qqq, tqqq, params=selected_params).target
        ),
        "wildwest_defaults": simulate(
            tqqq, signal_frame(qqq, tqqq, params=V3Parameters()).target, stop=0.25
        ),
    }
    rows = []
    for period, (start, end) in periods.items():
        benchmark = metrics(normalized_slice(strategies["synthetic_buy_hold"], start, end))
        for strategy, result in strategies.items():
            values = metrics(normalized_slice(result, start, end))
            rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    **values,
                    "excess_cagr": values["cagr"] - benchmark["cagr"],
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT / "long_history_falsification.csv", index=False)
    print(output.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
