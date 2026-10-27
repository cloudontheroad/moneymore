"""Lagged peer-matched diagnostics, not a trading strategy or causal estimate."""

from __future__ import annotations

import json
import sqlite3
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .challenger_diagnostics import prediction_reports, trading_dates, write_diagnostic


def board(symbol: str) -> str:
    return (
        "STAR"
        if symbol.startswith("688")
        else "CHINEXT"
        if symbol.startswith(("300", "301"))
        else "MAIN"
    )


def matched_day(
    positions: list[dict[str, Any]],
    equity: float,
    actual_return: float,
    fee: float,
    prices: pd.Series,
    returns: pd.Series,
    features: pd.DataFrame,
    pool: list[str],
    neighbors: int = 5,
) -> dict[str, Any]:
    if equity <= 0:
        return {"status": "INVALID_EQUITY"}
    held = {str(row["symbol"]) for row in positions if row["quantity"] > 0}
    universe = sorted(set(pool))
    features = features.reindex(sorted(set(universe) | held)).replace([np.inf, -np.inf], np.nan)
    ranks = features.rank(pct=True)
    complete = set(ranks.dropna().index)
    stock_return = peer_return = exposure = 0.0
    matches = []
    for position in positions:
        symbol, quantity = str(position["symbol"]), int(position["quantity"])
        if quantity <= 0:
            continue
        if (
            symbol not in ranks.index
            or ranks.loc[symbol].isna().any()
            or not np.isfinite(prices.get(symbol, np.nan))
            or not np.isfinite(returns.get(symbol, np.nan))
        ):
            return {"status": "MISSING_HELD_STYLE_OR_PRICE", "symbol": symbol}
        peers = [
            s for s in universe if s not in held and board(s) == board(symbol) and s in complete
        ]
        if len(peers) < neighbors:
            return {"status": "INSUFFICIENT_STYLE_PEERS", "symbol": symbol}
        distances = ((ranks.loc[peers] - ranks.loc[symbol]) ** 2).sum(axis=1)
        # Match on t-1 only. Never select peers based on t return availability.
        chosen = distances.sort_values(kind="stable").head(neighbors).index.tolist()
        peer_returns = returns.reindex(chosen)
        if not np.isfinite(peer_returns).all():
            return {"status": "MISSING_MATCHED_PEER_RETURN", "symbol": symbol}
        weight = quantity * float(prices[symbol]) / equity
        exposure += weight
        stock_return += weight * float(returns[symbol])
        peer_return += weight * float(peer_returns.mean())
        matches.append(
            {
                "symbol": symbol,
                "weight": weight,
                "peers": chosen,
                "distance": float(distances.loc[chosen].mean()),
            }
        )
    market_returns = returns.reindex(universe)
    coverage = float(market_returns.notna().mean()) if universe else 0
    if coverage < 0.95:
        return {"status": "INSUFFICIENT_MARKET_COVERAGE", "coverage": coverage}
    market = exposure * float(market_returns.mean())
    costs = -fee / equity
    return {
        "status": "MATCHED",
        "actual_return": actual_return,
        "market": market,
        "style": peer_return - market,
        "selection": stock_return - peer_return,
        "trading_and_other": actual_return - stock_return - costs,
        "costs": costs,
        "prior_exposure": exposure,
        "matches": matches,
        "market_coverage": coverage,
    }


def linked_attribution(daily: list[dict[str, Any]]) -> dict[str, Any]:
    components = {
        name: 0.0 for name in ("market", "style", "selection", "trading_and_other", "costs")
    }
    wealth = 1.0
    for row in daily:
        for name in components:
            components[name] += wealth * row[name]
        wealth *= 1 + row["actual_return"]
    total = wealth - 1
    return {
        **components,
        "total_return": total,
        "reconciliation_error": sum(components.values()) - total,
    }


def run_style_attribution(root: Path, store: Any, as_of: str) -> dict[str, Any]:
    from .qlib_candidate_observer import candidate_catalog

    candidates = [row for row in candidate_catalog(root) if not row["archived"]]
    sources = [
        ("multi_sector_shadow", root / "state/multi-sector-shadow"),
        ("qlib_gru_shadow", root / "state/qlib-challenger-shadow"),
    ]
    sources.extend(
        (row["account_id"], root / "state/qlib-candidate-shadow" / row["candidate_tag"])
        for row in candidates
    )
    histories = {
        "multi_sector_shadow": store.read("multi_sector_account_daily"),
        "qlib_gru_shadow": store.read("qlib_challenger_account_daily"),
    }
    all_history = store.read("qlib_candidate_account_daily")
    histories.update(
        {
            row["account_id"]: all_history.loc[all_history.account_id == row["account_id"]]
            for row in candidates
        }
    )
    history_series = {
        key: value.assign(trade_date=value.trade_date.astype(str))
        .drop_duplicates("trade_date", keep="last")
        .set_index("trade_date")
        .equity.sort_index()
        for key, value in histories.items()
        if not value.empty
    }
    common = sorted(set.intersection(*(set(frame.index) for frame in history_series.values())))
    # Diagnose current 1000-stock deployment, not the earlier hand-picked universe.
    dates = [day for day in common if "20261008" <= day <= as_of][-21:]
    if len(dates) < 2:
        payload = {"as_of_date": as_of, "status": "INSUFFICIENT_COMMON_HISTORY", "accounts": []}
        write_diagnostic(root / "state/challenger-diagnostics/style-attribution.json", payload)
        return payload
    reports: dict[str, dict[str, Any]] = {}
    for account, directory in sources:
        by_date: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            date = str(row.get("trade_date", ""))
            if date in dates and row.get("portfolio"):
                previous = by_date.get(date)
                if previous is None or str(row.get("created_at", "")) > str(
                    previous.get("created_at", "")
                ):
                    by_date[date] = row
        reports[account] = by_date
    market_reports = {
        row["trade_date"]: row
        for row in prediction_reports(root / "state/qlib-challenger-shadow", as_of)
    }
    pools = {
        date: [str(score["instrument"]) for score in market_reports.get(date, {}).get("scores", [])]
        for date in dates
    }
    symbols = sorted(
        {symbol for pool in pools.values() for symbol in pool}
        | {
            str(position["symbol"])
            for rows in reports.values()
            for row in rows.values()
            for position in row["portfolio"].get("positions", [])
        }
    )
    start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=75)).strftime("%Y%m%d")
    filters = [("trade_date", ">=", start), ("trade_date", "<=", as_of), ("ts_code", "in", symbols)]
    market = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    factors = store.read(
        "adj_factor", columns=["ts_code", "trade_date", "adj_factor"], filters=filters
    )
    basic = store.read(
        "daily_basic", columns=["ts_code", "trade_date", "total_mv"], filters=filters
    )
    calendar = [day for day in trading_dates(store, as_of) if start <= day <= as_of]

    def pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
        frame = frame.assign(trade_date=frame.trade_date.astype(str))
        return (
            frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
            .pivot(index="trade_date", columns="ts_code", values=value)
            .reindex(index=calendar, columns=symbols)
        )

    close = pivot(market, "close")
    adjusted = close * pivot(factors, "adj_factor")
    returns = adjusted.pct_change(fill_method=None)
    momentum = adjusted.pct_change(20, fill_method=None)
    volatility = returns.rolling(20, min_periods=20).std()
    size = np.log(pivot(basic, "total_mv").where(lambda frame: frame > 0))
    with sqlite3.connect(
        f"file:{(root / 'state/paper_orders.sqlite3').as_posix()}?mode=ro", uri=True
    ) as connection:
        fees = pd.read_sql_query(
            "SELECT account_id,trade_date,SUM(fee) fee FROM fills WHERE trade_date>=? AND trade_date<=? GROUP BY account_id,trade_date",
            connection,
            params=(dates[0], as_of),
        )
    fee_map = {
        (str(row.account_id), str(row.trade_date)): float(row.fee) for row in fees.itertuples()
    }
    accounts = []
    for account, _ in sources:
        if account not in history_series:
            continue
        rows, skipped = [], []
        sensitivity: dict[int, list[dict[str, Any]]] = {3: [], 10: []}
        for prior, date in pairwise(dates):
            # Reject multi-day gaps rather than treating them as one trading day.
            if calendar.index(date) != calendar.index(prior) + 1:
                skipped.append({"trade_date": date, "status": "NONCONSECUTIVE_COMMON_DATES"})
                continue
            report = reports[account].get(prior)
            equity = float(history_series[account].loc[prior])
            if not report or abs(float(report["portfolio"]["equity"]) - equity) > 0.02:
                skipped.append(
                    {"trade_date": date, "status": "MISSING_OR_MISMATCHED_HOLDING_SNAPSHOT"}
                )
                continue
            actual = float(history_series[account].loc[date]) / equity - 1
            features = pd.DataFrame(
                {
                    "size": size.loc[prior],
                    "momentum": momentum.loc[prior],
                    "volatility": volatility.loc[prior],
                }
            )
            result = matched_day(
                report["portfolio"].get("positions", []),
                equity,
                actual,
                fee_map.get((account, date), 0),
                close.loc[prior],
                returns.loc[date],
                features,
                pools[prior],
            )
            result.update(trade_date=date, prior_date=prior)
            (rows if result["status"] == "MATCHED" else skipped).append(result)
            for neighbors, matched_rows in sensitivity.items():
                alternative = matched_day(
                    report["portfolio"].get("positions", []),
                    equity,
                    actual,
                    fee_map.get((account, date), 0),
                    close.loc[prior],
                    returns.loc[date],
                    features,
                    pools[prior],
                    neighbors=neighbors,
                )
                if alternative["status"] == "MATCHED":
                    matched_rows.append(alternative)
        summary = linked_attribution(rows)
        full_return = float(
            history_series[account].loc[dates[-1]] / history_series[account].loc[dates[0]] - 1
        )
        accounts.append(
            {
                "account_id": account,
                "status": "COMPLETE" if not skipped else "PARTIAL",
                "start_date": dates[0],
                "end_date": dates[-1],
                "expected_return_days": len(dates) - 1,
                "matched_days": len(rows),
                "skipped_days": skipped,
                "actual_period_return": full_return,
                "contributions": summary,
                "sensitivity": {
                    str(k): {
                        "matched_days": len(values),
                        "contributions": linked_attribution(values),
                    }
                    for k, values in sensitivity.items()
                },
                "daily": rows,
            }
        )
    payload = {
        "as_of_date": as_of,
        "status": "READY",
        "start_date": dates[0],
        "end_date": dates[-1],
        "method": "LAGGED_BOARD_SIZE_MOMENTUM_VOLATILITY_5_NEAREST_PEERS",
        "accounts": accounts,
        "writes_orders": False,
        "limitations": [
            "Diagnostic counterfactual, not causal identification or executable backtest.",
            "Peer selection uses t-1 ranks only; current industry labels do not enter matching.",
            "Market equals the prior recorded GRU scored pool, not a broad index.",
            "Trading-and-other includes intraday position changes, execution, dividends/tax and cash effects; not pure execution skill.",
            "Adjusted-price total returns approximate corporate actions; actual costs separated. Missing styles never filled from future.",
            "PARTIAL components describe matched days only; cannot be presented as decomposition of the full-period return.",
        ],
    }
    write_diagnostic(root / "state/challenger-diagnostics/style-attribution.json", payload)
    return payload
