"""Read-only turnover/drawdown diagnosis of the completed isolated rank study.

Per-name PnL is accounting attribution, not a causal effect of a mechanism or
tax-realized gains. No return labels or new market data are needed.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "state/ashare-mainline-20261112"
SOURCE = BASE / "n3r-rank-v2"
KINDS = ("rank_cap", "rank_equal", "topk")
CAPITAL = 10_000_000.0


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def turnover_summary(frame):
    if frame.empty:
        return {"fills": 0, "notional": 0.0, "fees": 0.0, "slippage": 0.0,
                "observed_cost": 0.0, "half_gross_turnover": 0.0}
    return {"fills": len(frame), "notional": frame.notional.sum(), "fees": frame.fee.sum(),
            "slippage": frame.slippage.sum(), "observed_cost": frame.observed_cost.sum(),
            "half_gross_turnover": (frame.notional / frame.previous_nav).sum() / 2,
            "median_notional": frame.notional.median(),
            "notional_quantiles": frame.notional.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
            "nav_bps_quantiles": frame.trade_nav_bps.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
            "cost_bps_of_notional": frame.observed_cost.sum() / frame.notional.sum() * 10000}


def max_drawdown(equity):
    nav = pd.concat([pd.Series([CAPITAL], index=["20251231"]), equity.set_index("date").equity])
    peak = nav.cummax()
    drawdown = nav / peak - 1
    trough_date = drawdown.idxmin()
    peak_value = peak.loc[trough_date]
    peak_date = nav.loc[:trough_date].loc[lambda x: np.isclose(x, peak_value, rtol=0, atol=1e-7)].index[-1]
    recovery = nav.loc[nav.index > trough_date].loc[lambda x: x >= peak_value]
    return {"maximum_drawdown": -drawdown.min(), "peak_date": peak_date,
            "trough_date": trough_date, "peak_nav": peak_value, "trough_nav": nav.loc[trough_date],
            "net_pnl": nav.loc[trough_date] - peak_value,
            "recovery_date": recovery.index[0] if len(recovery) else None,
            "return_dates": "strictly after peak date through trough date inclusive"}


def attach_industry(frame, panel):
    result = frame.merge(panel, on=["date", "symbol"], how="left", validate="many_to_one")
    missing = result.industry.isna()
    # Classifications absent after a pool exit use only previously known records.
    history = {symbol: group.sort_values("date") for symbol, group in panel.groupby("symbol")}
    for index in result.index[missing]:
        row = result.loc[index]
        past = history.get(row.symbol)
        if past is not None:
            past = past.loc[past.date.le(row.date) & past.industry.notna()]
            if len(past):
                result.loc[index, "industry"] = past.iloc[-1].industry
    result["industry"] = result.industry.fillna("UNKNOWN")
    return result


def position_trade_partition(fills, ledger, holdings, days):
    """Reconstruct exact physical quantities including listed stock dividends."""
    positions, rows, snapshots = {}, [], []
    if fills.duplicated(["date", "symbol"]).any():
        raise ValueError("MULTIPLE_SAME_DAY_NAME_FILLS_REQUIRE_EXPLICIT_ORDER")
    for day in days:
        events = ledger.loc[ledger.trade_date.eq(day)]
        for event in events.loc[events.action_type.eq("STOCK_DIVIDEND")].itertuples():
            positions[event.symbol] = positions.get(event.symbol, 0) + int(event.share_quantity)
        trades = fills.loc[fills.date.eq(day)].sort_values(["side", "symbol"], ascending=[False, True])
        for trade in trades.to_dict("records"):
            symbol, quantity = trade["symbol"], int(trade["quantity"])
            before = positions.get(symbol, 0)
            if trade["side"] == "BUY":
                category = ("INITIAL_BUILD" if day == days[0] else "NEW_ENTRY") if before == 0 else "RETAINED_BUY_RESIZE"
                after = before + quantity
            elif trade["side"] == "SELL":
                after = before - quantity
                if after < 0:
                    raise ValueError("NEGATIVE_RECONSTRUCTED_POSITION")
                category = "FULL_EXIT" if after == 0 else "RETAINED_SELL_RESIZE"
            else:
                raise ValueError("UNKNOWN_FILL_SIDE")
            positions[symbol] = after
            rows.append({**trade, "quantity_before": before, "quantity_after": after,
                         "trade_category": category})
        expected = set(holdings.loc[holdings.date.eq(day), "symbol"])
        actual = {s for s, q in positions.items() if q > 0}
        if expected != actual:
            raise ValueError("RECONSTRUCTED_HOLDINGS_SYMBOLS_MISMATCH:" + day)
        snapshots.extend({"date": day, "symbol": s, "quantity": q} for s, q in positions.items() if q > 0)
    return pd.DataFrame(rows), pd.DataFrame(snapshots)


def case_diagnostic(bps, kind, panel, input_hashes):
    name = "doubleensemble_full--" + kind
    folder = SOURCE / f"cost{bps}/bps{bps}"
    case = folder / name
    equity = pd.read_parquet(case / "equity.parquet").sort_values("date").reset_index(drop=True)
    fills = pd.read_parquet(case / "fills.parquet")
    holdings = pd.read_parquet(folder / "holdings.parquet")
    holdings = holdings.loc[holdings.kind.eq(name)].copy()
    database = case / "resume.sqlite3"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", connection)
        database_fills = pd.read_sql_query(
            "SELECT symbol,side,quantity,price,fee,trade_date AS date FROM fills", connection
        )
    pd.testing.assert_frame_equal(
        fills[database_fills.columns].sort_values(["date", "symbol"]).reset_index(drop=True),
        database_fills.sort_values(["date", "symbol"]).reset_index(drop=True), check_dtype=False,
    )
    days = equity.date.tolist()
    fills, quantities = position_trade_partition(fills, ledger, holdings, days)
    previous_nav = equity.set_index("date").equity.shift(1)
    previous_nav.iloc[0] = CAPITAL
    fills["previous_nav"] = fills.date.map(previous_nav)
    fills["trade_nav_bps"] = fills.notional / fills.previous_nav * 10000
    fills["observed_cost"] = fills.fee + fills.slippage
    fills["small_trade_bucket"] = pd.cut(
        fills.trade_nav_bps, bins=[-np.inf, 5, 10, 25, 50, 100, np.inf], right=False,
        labels=["under_5bps", "5_to_10bps", "10_to_25bps", "25_to_50bps", "50_to_100bps", "at_least_100bps"],
    ).astype(str)
    holding_values = holdings.merge(equity[["date", "equity"]], on="date", validate="many_to_one")
    holding_values["physical_value"] = holding_values.weight * holding_values.equity
    values = holding_values.pivot(index="date", columns="symbol", values="physical_value").reindex(days).fillna(0)
    # Here zero means an audited zero position, not a missing stock price or return.
    gap = equity.set_index("date").equity - equity.set_index("date").cash - values.sum(axis=1)
    symbols = sorted(set(values.columns) | set(fills.symbol) | set(ledger.symbol))
    values = values.reindex(columns=symbols, fill_value=0)
    cashflow = pd.DataFrame(0.0, index=days, columns=symbols)
    fees = pd.DataFrame(0.0, index=days, columns=symbols)
    slippage = pd.DataFrame(0.0, index=days, columns=symbols)
    dividends = pd.DataFrame(0.0, index=days, columns=symbols)
    for row in fills.itertuples():
        cashflow.loc[row.date, row.symbol] += row.notional * (1 if row.side == "SELL" else -1) - row.fee
        fees.loc[row.date, row.symbol] += row.fee
        slippage.loc[row.date, row.symbol] += row.slippage
    for row in ledger.loc[ledger.action_type.eq("CASH_DIVIDEND")].itertuples():
        if row.trade_date in dividends.index:
            dividends.loc[row.trade_date, row.symbol] += row.cash_amount
    changes = values - values.shift(1, fill_value=0)
    pnl = changes + cashflow + dividends
    expected_change = equity.set_index("date").equity - previous_nav
    errors = pnl.sum(axis=1) - expected_change
    cash_change = equity.set_index("date").cash.diff()
    cash_change.iloc[0] = equity.cash.iloc[0] - CAPITAL
    cash_errors = cashflow.sum(axis=1) + dividends.sum(axis=1) - cash_change
    reconciled = bool(gap.abs().max() < 0.01 and errors.abs().max() < 0.01 and cash_errors.abs().max() < 0.01)
    if not reconciled:
        # Pending rights would require extra asset-level reconstruction. Never
        # silently allocate the residual to cash or pretend it is stock PnL.
        raise ValueError("PER_NAME_ATTRIBUTION_NOT_RECONCILED:" + name + ":" + str(bps)
                         + ":" + str({"asset_gap": gap.abs().max(), "pnl_error": errors.abs().max()}))
    contributions = pnl.rename_axis(index="date", columns="symbol").stack().rename("net_pnl").reset_index()
    for field, matrix in [("fees", fees), ("slippage", slippage), ("cash_dividends", dividends)]:
        contributions[field] = matrix.rename_axis(index="date", columns="symbol").stack().to_numpy()
    contributions["previous_nav"] = contributions.date.map(previous_nav)
    contributions["net_return_contribution"] = contributions.net_pnl / contributions.previous_nav
    contributions = attach_industry(contributions, panel)
    dd = max_drawdown(equity)
    periods = {"full_period": contributions,
               "max_drawdown": contributions.loc[contributions.date.gt(dd["peak_date"])
                                                    & contributions.date.le(dd["trough_date"])]}
    attribution = {}
    for period, frame in periods.items():
        anchor = CAPITAL if period == "full_period" else dd["peak_nav"]
        sections = {}
        for dimension in ["industry", "symbol"]:
            aggregate = frame.groupby(dimension)[["net_pnl", "fees", "slippage", "cash_dividends"]].sum()
            aggregate["contribution_of_anchor_nav"] = aggregate.net_pnl / anchor
            aggregate = aggregate.sort_values("net_pnl")
            sections[dimension] = aggregate.reset_index().to_dict("records")
        attribution[period] = {"anchor_nav": anchor, "net_pnl": frame.net_pnl.sum(), **sections}
    decisions = []
    decision_files = sorted((SOURCE / f"cost{bps}/decisions" / name).glob("*.json"))
    for path in decision_files:
        decision = json.loads(path.read_bytes())
        decisions.append({"date": path.stem, "status": decision.get("status"),
                          "reason": decision.get("reason"),
                          "retry_next_session": decision.get("retry_next_session", False),
                          "allocation_status": (decision.get("allocation") or {}).get("status"),
                          "n_open_clips": len(decision.get("open_clips", decision.get("clips", [])))})
    decision_frame = pd.DataFrame(decisions)
    fills = fills.merge(decision_frame[["date", "status", "reason"]], on="date", how="left", validate="many_to_one")
    trade_days = fills.groupby("date").size()
    trade_days_without_plan = fills.loc[fills.status.isna()]
    if len(trade_days_without_plan):
        raise ValueError("TRADE_WITHOUT_DECISION")
    daily_audit = equity[["date", "equity", "cash", "fills", "fees", "slippage"]].copy()
    daily_audit["physical_asset_gap"] = gap.to_numpy()
    daily_audit["pnl_reconciliation_error"] = errors.to_numpy()
    daily_audit["cash_reconciliation_error"] = cash_errors.to_numpy()
    for column, expected in [("fee", "fees"), ("slippage", "slippage")]:
        actual = fills.groupby("date")[column].sum().reindex(days, fill_value=0).to_numpy()
        if not np.allclose(actual, equity[expected], rtol=0, atol=1e-7):
            raise ValueError("DAILY_TRADE_COST_MISMATCH")
    refresh = fills.loc[fills.trade_category.isin(["NEW_ENTRY", "FULL_EXIT"])]
    resize = fills.loc[fills.trade_category.str.startswith("RETAINED")]
    after_initial = fills.loc[fills.date.ne(days[0])]
    result = {"case": kind, "bps": bps, "sessions": len(days),
              "net_return": equity.equity.iloc[-1] / CAPITAL - 1,
              "maximum_drawdown": dd, "trading_days": len(trade_days),
              "decision_status_counts": decision_frame.status.value_counts(dropna=False).to_dict(),
              "decision_reason_counts": decision_frame.reason.dropna().value_counts().to_dict(),
              "turnover_all": turnover_summary(fills), "turnover_after_initial": turnover_summary(after_initial),
              "turnover_by_category": {category: turnover_summary(g) for category, g in fills.groupby("trade_category")},
              "turnover_by_size": {category: turnover_summary(g) for category, g in fills.groupby("small_trade_bucket")},
              "retained_resize": turnover_summary(resize), "name_refresh": turnover_summary(refresh),
              "resize_under_10bps_nav": turnover_summary(resize.loc[resize.trade_nav_bps.lt(10)]),
              "resize_under_25bps_nav": turnover_summary(resize.loc[resize.trade_nav_bps.lt(25)]),
              "reconciliation": {"matched": reconciled, "max_daily_pnl_error": errors.abs().max(),
                                 "max_daily_cash_error": cash_errors.abs().max(),
                                 "max_daily_nonphysical_assets": gap.abs().max(),
                                 "unknown_industry_nonzero_contribution_rows": int((contributions.industry.eq("UNKNOWN") & contributions.net_pnl.ne(0)).sum())},
              "attribution": attribution}
    for path in [case / "equity.parquet", case / "fills.parquet", folder / "holdings.parquet", database, *decision_files]:
        input_hashes[str(path.relative_to(ROOT))] = sha256(path)
    for frame in [fills, quantities, contributions, decision_frame, daily_audit]:
        frame["case"], frame["bps"] = kind, bps
    return result, fills, quantities, contributions, decision_frame, daily_audit


def main(output):
    output = output.resolve()
    if not output.is_relative_to(BASE.resolve()) or output == BASE.resolve():
        raise ValueError("ISOLATED_RESEARCH_OUTPUT_REQUIRED")
    if output.exists():
        raise FileExistsError("IMMUTABLE_OUTPUT:" + str(output))
    source_result = json.loads((SOURCE / "final/results.json").read_bytes())
    if source_result["status"] != "FULL_MATRIX_COMPLETE_NOT_RELEASED":
        raise ValueError("COMPLETED_STUDY_REQUIRED")
    panel_path = BASE / "panel-v2/panel.parquet"
    panel = pd.read_parquet(panel_path, columns=["date", "symbol", "industry"])
    hashes = {str(panel_path.relative_to(ROOT)): sha256(panel_path),
              str((SOURCE / "final/results.json").relative_to(ROOT)): sha256(SOURCE / "final/results.json")}
    all_results, fill_frames, quantity_frames, contribution_frames, decision_frames, audit_frames = [], [], [], [], [], []
    for bps in (10, 20):
        for kind in KINDS:
            result, fills, quantities, contributions, decisions, audit = case_diagnostic(bps, kind, panel, hashes)
            all_results.append(result)
            for target, frame in [(fill_frames, fills), (quantity_frames, quantities),
                                  (contribution_frames, contributions), (decision_frames, decisions), (audit_frames, audit)]:
                target.append(frame)
            print(json.dumps(json_safe({"case": kind, "bps": bps, "max_dd": result["maximum_drawdown"],
                                        "retained_resize": result["retained_resize"],
                                        "name_refresh": result["name_refresh"]})), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    for name, frames in [("partitioned-fills", fill_frames), ("reconstructed-quantities", quantity_frames),
                         ("daily-stock-contributions", contribution_frames), ("decisions", decision_frames),
                         ("daily-reconciliation", audit_frames)]:
        pd.concat(frames, ignore_index=True).to_parquet(output / (name + ".parquet"), index=False)
    report = {"status": "RECONCILED_HISTORICAL_MECHANISM_DIAGNOSTIC_NOT_NEW_SELECTION_TEST",
              "source": str(SOURCE), "cases": all_results,
              "method": {
                  "net_stock_pnl": "Change in actual physical end-close holding value + sell notional - buy notional - fees + actual cash dividends; sums to daily NAV change within 0.01 currency units.",
                  "slippage": "Already embedded in actual execution prices, displayed separately for cost diagnosis and never subtracted twice.",
                  "positions": "Physical quantities reconstructed from fills and stock-dividend settlement ledger; every daily holding symbol set is verified.",
                  "asset_sufficiency": "Verify NAV minus cash equals physical holding values each day; otherwise fail closed because pending rights require separate attribution.",
                  "industry": "Date-specific panel classification, falling back only to latest previously known classification for exited names. Unknown remains UNKNOWN.",
                  "turnover": "Half gross actual trade notional divided by previous close NAV; initial build shown separately; resize means the stock remains held after a partial buy/sell.",
                  "drawdown": "Includes initial NAV anchor; contributions sum over dates after the preceding peak through the trough, divided by peak NAV.",
              },
              "limitations": [
                  "Attribution is reconciled historical mark-to-market accounting, including unrealized gains; not tax-realized gains or a causal claim about why a mechanism wins.",
                  "Avoided observed resize costs are only a gross mechanical ceiling; suppressing trades changes later holdings, refresh feasibility and future costs, so a prospective fixed mechanism replay is required.",
                  "No future-return labels are used or new stocks chosen here. The six historical ledgers share their original universe, signal and data limitations.",
                  "No benchmark-relative alpha is claimed; the full-pool benchmark remains incomplete in the prior diagnostic.",
              ],
              "production_modified": False, "new_data_fetched": False, "real_orders_enabled": False,
              "input_hashes": hashes, "script_sha256": sha256(Path(__file__)),
              "artifact_hashes": {p.name: sha256(p) for p in output.iterdir()}}
    save(output / "results.json", report)
    print("OUTPUT " + str(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
