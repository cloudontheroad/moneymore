"""Frozen, benchmark-first research for the WildWest-inspired TQQQ v3.

No paper database is opened.  Development/validation select a parameter plateau;
2020+ is reported as external replication (not pristine holdout because the source
project published that period).  2026-11-04 onward is reserved for prospective A/B.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneymore.leveraged_etf import YahooDailyClient
from moneymore.tqqq_v3 import V3Parameters, signal_frame

OUTPUT = ROOT / "data" / "research" / "tqqq_v3"
BASE_COST = 0.0006


def download(symbol: str, start: str) -> pd.DataFrame:
    end = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    frame = YahooDailyClient(timeout=60).history(symbol, start, end)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame.set_index("trade_date").sort_index()


def load_real_history() -> pd.DataFrame:
    frames = {
        "qqq": download("QQQ", "2008-01-01"),
        "tqqq": download("TQQQ", "2010-01-01"),
        "vix": download("^VIX", "2008-01-01"),
        "vix3m": download("^VIX3M", "2008-01-01"),
    }
    index = frames["tqqq"].index
    result = pd.DataFrame(index=index)
    for name in ("qqq", "tqqq"):
        result[f"{name}_open"] = frames[name]["open"].reindex(index)
        result[f"{name}_close"] = frames[name]["close"].reindex(index)
    result["vix_close"] = frames["vix"]["close"].reindex(index).ffill()
    result["vix3m_close"] = frames["vix3m"]["close"].reindex(index).ffill()
    return result.dropna(subset=["qqq_open", "qqq_close", "tqqq_open", "tqqq_close"])


def simulate_next_open(
    frame: pd.DataFrame,
    close_targets: pd.Series,
    cost: float = BASE_COST,
    trailing_stop: float | None = None,
    cooldown_days: int = 5,
) -> pd.DataFrame:
    desired_at_open = close_targets.shift(1).reindex(frame.index)
    equity = 1.0
    weight = 1.0
    peak = 1.0
    in_cash = False
    cooldown = 0
    rows = []
    previous_close = None
    for date, row in frame.iterrows():
        open_price = float(row.tqqq_open)
        close_price = float(row.tqqq_close)
        if previous_close is not None:
            equity *= 1.0 + weight * (open_price / previous_close - 1.0)
        peak = max(peak, equity)
        desired = float(desired_at_open.loc[date]) if pd.notna(desired_at_open.loc[date]) else weight
        if trailing_stop is not None:
            if in_cash:
                cooldown += 1
                if cooldown >= cooldown_days and desired >= 0.80:
                    in_cash = False
                    peak = equity
                else:
                    desired = 0.0
            elif equity / peak - 1.0 <= -trailing_stop:
                in_cash = True
                cooldown = 0
                desired = 0.0
        turnover = abs(desired - weight)
        equity *= max(0.0, 1.0 - turnover * cost)
        weight = desired
        equity *= 1.0 + weight * (close_price / open_price - 1.0)
        rows.append(
            {"trade_date": date, "equity": equity, "weight": weight, "turnover": turnover}
        )
        previous_close = close_price
    return pd.DataFrame(rows).set_index("trade_date")


def metrics(result: pd.DataFrame) -> dict[str, float]:
    returns = result.equity.pct_change().dropna()
    years = max((result.index[-1] - result.index[0]).days / 365.25, 1 / 252)
    drawdown = result.equity / result.equity.cummax() - 1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    return {
        "cagr": float(result.equity.iloc[-1] ** (1 / years) - 1.0),
        "sharpe_rf0": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
        "annualized_volatility": volatility,
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(result.turnover.mean() * 252),
        "average_exposure": float(result.weight.mean()),
    }


def slice_result(result: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    part = result.loc[start:end].copy()
    if part.empty:
        return part
    part["equity"] = part.equity / part.equity.iloc[0]
    return part


def run_candidate(
    frame: pd.DataFrame,
    params: V3Parameters,
    cost: float = BASE_COST,
    trailing_stop: float | None = None,
) -> pd.DataFrame:
    signals = signal_frame(
        frame.qqq_close,
        frame.tqqq_close,
        frame.vix_close,
        frame.vix3m_close,
        params,
    )
    return simulate_next_open(frame, signals.target, cost, trailing_stop)


def parameter_grid() -> list[V3Parameters]:
    return [
        V3Parameters(
            momentum_days=momentum,
            volatility_days=volatility,
            negative_momentum_scale=negative_scale,
            oversold_rsi=oversold,
            dip_buy_allocation=dip_allocation,
            rebalance_band=band,
        )
        for momentum, volatility, negative_scale, oversold, dip_allocation, band in itertools.product(
            (10, 20, 40, 60),
            (10, 20, 40),
            (0.20, 0.30, 0.50),
            (25.0, 30.0, 35.0),
            (0.60, 0.80, 1.00),
            (0.05, 0.10),
        )
    ]


def block_bootstrap_excess(
    candidate: pd.Series, benchmark: pd.Series, samples: int = 2000, block: int = 20
) -> dict[str, float]:
    excess = candidate.align(benchmark, join="inner")[0] - candidate.align(benchmark, join="inner")[1]
    values = excess.dropna().to_numpy(dtype=float)
    random = np.random.default_rng(42)
    blocks = int(np.ceil(len(values) / block))
    annual = []
    for _ in range(samples):
        starts = random.integers(0, len(values) - block + 1, blocks)
        sample = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
        annual.append(float(sample.mean() * 252))
    quantiles = np.quantile(annual, [0.025, 0.5, 0.975])
    return {
        "annual_excess_ci_low": float(quantiles[0]),
        "annual_excess_median": float(quantiles[1]),
        "annual_excess_ci_high": float(quantiles[2]),
        "probability_positive": float(np.mean(np.asarray(annual) > 0.0)),
    }


def main() -> None:
    frame = load_real_history()
    benchmark = simulate_next_open(
        frame,
        pd.Series(1.0, index=frame.index),
        BASE_COST,
    )
    periods = {
        "development": ("2010-03-01", "2016-12-31"),
        "validation": ("2017-01-01", "2019-12-31"),
        "external_replication": ("2020-01-01", str(frame.index[-1].date())),
        "full_real": ("2010-03-01", str(frame.index[-1].date())),
    }
    benchmark_metrics = {
        name: metrics(slice_result(benchmark, start, end))
        for name, (start, end) in periods.items()
    }

    rows = []
    cached_results: dict[str, pd.DataFrame] = {}
    for params in parameter_grid():
        key = json.dumps(asdict(params), sort_keys=True)
        result = run_candidate(frame, params)
        cached_results[key] = result
        record: dict[str, object] = {"parameters": key}
        for name, (start, end) in periods.items():
            candidate_metrics = metrics(slice_result(result, start, end))
            benchmark_row = benchmark_metrics[name]
            for metric_name, value in candidate_metrics.items():
                record[f"{name}_{metric_name}"] = value
            record[f"{name}_excess_cagr"] = candidate_metrics["cagr"] - benchmark_row["cagr"]
        rows.append(record)
    grid = pd.DataFrame(rows)
    eligible = grid[
        (grid.development_excess_cagr > 0.0)
        & (grid.validation_excess_cagr > 0.0)
    ].copy()
    if eligible.empty:
        selected_row = grid.sort_values(
            ["validation_excess_cagr", "development_excess_cagr"], ascending=False
        ).iloc[0]
        selection_status = "FAILED_NO_POSITIVE_DEV_VALIDATION"
    else:
        eligible["robust_score"] = eligible[
            ["development_excess_cagr", "validation_excess_cagr"]
        ].min(axis=1)
        selected_row = eligible.sort_values(
            ["robust_score", "validation_excess_cagr"], ascending=False
        ).iloc[0]
        selection_status = "SELECTED_ON_DEV_VALIDATION_ONLY"
    selected_key = str(selected_row.parameters)
    selected_params = V3Parameters(**json.loads(selected_key))
    selected = cached_results[selected_key]

    ablations = {
        "M0_buy_hold": None,
        "M1_momentum_only": replace(
            selected_params, use_relative_volatility=False, use_dip_buy=False
        ),
        "M2_momentum_relative_vol": replace(selected_params, use_dip_buy=False),
        "M3_plus_dip_buy": selected_params,
        "M4_vix_confirmed_dip": replace(selected_params, use_vix_confirmation=True),
        "M5_reversal_confirmed_dip": replace(
            selected_params, use_reversal_confirmation=True
        ),
    }
    ablation_rows = []
    for name, params in ablations.items():
        result = benchmark if params is None else run_candidate(frame, params)
        for period, (start, end) in periods.items():
            values = metrics(slice_result(result, start, end))
            ablation_rows.append(
                {
                    "module": name,
                    "period": period,
                    **values,
                    "excess_cagr": values["cagr"] - benchmark_metrics[period]["cagr"],
                }
            )

    wildwest = V3Parameters()
    wildwest_result = run_candidate(frame, wildwest, trailing_stop=0.25)
    for label, result in (
        ("M6_trailing_stop", run_candidate(frame, selected_params, trailing_stop=0.25)),
        ("M7_wildwest_defaults", wildwest_result),
    ):
        for period, (start, end) in periods.items():
            values = metrics(slice_result(result, start, end))
            ablation_rows.append(
                {
                    "module": label,
                    "period": period,
                    **values,
                    "excess_cagr": values["cagr"] - benchmark_metrics[period]["cagr"],
                }
            )

    cost_rows = []
    for cost_bps in (6, 10, 25, 50):
        result = run_candidate(frame, selected_params, cost=cost_bps / 10_000)
        for period in ("validation", "external_replication", "full_real"):
            start, end = periods[period]
            values = metrics(slice_result(result, start, end))
            cost_rows.append(
                {
                    "cost_bps": cost_bps,
                    "period": period,
                    **values,
                    "excess_cagr": values["cagr"] - benchmark_metrics[period]["cagr"],
                }
            )

    fold_rows = []
    for label, start, end in (
        ("2014_2016", "2014-01-01", "2016-12-31"),
        ("2017_2019", "2017-01-01", "2019-12-31"),
        ("2020_2022", "2020-01-01", "2022-12-31"),
        ("2023_now", "2023-01-01", str(frame.index[-1].date())),
    ):
        candidate_values = metrics(slice_result(selected, start, end))
        benchmark_values = metrics(slice_result(benchmark, start, end))
        fold_rows.append(
            {
                "fold": label,
                "candidate_cagr": candidate_values["cagr"],
                "benchmark_cagr": benchmark_values["cagr"],
                "excess_cagr": candidate_values["cagr"] - benchmark_values["cagr"],
                "candidate_max_drawdown": candidate_values["max_drawdown"],
                "benchmark_max_drawdown": benchmark_values["max_drawdown"],
            }
        )

    external_start, external_end = periods["external_replication"]
    candidate_returns = slice_result(selected, external_start, external_end).equity.pct_change()
    benchmark_returns = slice_result(benchmark, external_start, external_end).equity.pct_change()
    bootstrap = block_bootstrap_excess(candidate_returns, benchmark_returns)
    selected_external = metrics(slice_result(selected, external_start, external_end))
    selected_validation = metrics(slice_result(selected, *periods["validation"]))
    stress_25 = next(
        row for row in cost_rows if row["cost_bps"] == 25 and row["period"] == "external_replication"
    )
    folds = pd.DataFrame(fold_rows)
    gates = {
        "development_excess_positive": float(selected_row.development_excess_cagr) > 0,
        "validation_excess_positive": selected_validation["cagr"] > benchmark_metrics["validation"]["cagr"],
        "external_replication_excess_positive": selected_external["cagr"] > benchmark_metrics["external_replication"]["cagr"],
        "cost_25bps_external_excess_positive": float(stress_25["excess_cagr"]) > 0,
        "majority_walk_forward_folds_positive": int((folds.excess_cagr > 0).sum()) >= 3,
        "external_drawdown_not_worse": selected_external["max_drawdown"] >= benchmark_metrics["external_replication"]["max_drawdown"],
        "bootstrap_probability_positive_at_least_95pct": bootstrap["probability_positive"] >= 0.95,
    }
    gate_passed = all(gates.values())

    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT / "parameter_grid.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUTPUT / "ablation.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUTPUT / "cost_stress.csv", index=False)
    folds.to_csv(OUTPUT / "walk_forward_folds.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": selected.index,
            "candidate_equity": selected.equity,
            "candidate_weight": selected.weight,
            "benchmark_equity": benchmark.equity.reindex(selected.index),
        }
    ).to_csv(OUTPUT / "selected_daily.csv", index=False)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_status": selection_status,
        "selected_parameters": asdict(selected_params),
        "benchmark_metrics": benchmark_metrics,
        "selected_external_metrics": selected_external,
        "bootstrap": bootstrap,
        "gates": gates,
        "historical_gate_passed": gate_passed,
        "paper_account_created": False,
        "production_v2_changed": False,
        "true_prospective_start": "2026-11-04",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
