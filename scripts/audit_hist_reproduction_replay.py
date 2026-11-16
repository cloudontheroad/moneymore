"""Independently rebuild cash/positions/equity from filled-order journals."""
import argparse
import json

import numpy as np
import pandas as pd
from prepare_hist_reproduction import CFG, OUT, init_qlib, write_json


def audit(directory):
    from qlib.data import D
    from replay_hist_reproduction import EXECUTION
    audits = {}
    for lane in EXECUTION["lanes"]:
        orders = pd.read_parquet(directory / f"{lane}-orders.parquet")
        report = pd.read_parquet(directory / f"{lane}-daily.parquet")
        symbols = sorted(orders.instrument.unique())
        bars = D.features(symbols, ["$close", "$open", "$factor", "$volume", "$change"],
                          start_time=report.index.min(), end_time=report.index.max()).sort_index()
        close = bars["$close"].unstack("instrument").reindex(report.index).ffill()
        cash = float(CFG["initial_capital"])
        quantity = pd.Series(0., index=symbols)
        max_equity_error = max_cash_error = 0.
        prior_date = {}
        fills = 0
        for date, snapshot in report.iterrows():
            today = orders.loc[orders.date.eq(date)]
            for row in today.itertuples(index=False):
                if row.value == 0:
                    assert row.filled_adjusted_shares == 0 and row.cost == 0
                    continue
                assert row.value > 0 and row.cost >= 0 and row.filled_adjusted_shares > 0
                price = bars.loc[(row.instrument, date)]
                assert np.isfinite(price["$close"]) and price["$volume"] > 0
                assert abs(row.value-row.price_adjusted*row.filled_adjusted_shares) < .01
                execution_field = "$open" if lane.startswith("next_open") else "$close"
                assert abs(price[execution_field]-row.price_adjusted) < 1e-6
                buy = row.direction == 1
                rate = CFG["buy_cost"] if buy else CFG["sell_cost"]
                assert abs(row.cost-max(CFG["minimum_cost"], rate*row.value)) < .01
                assert prior_date.get(row.instrument) != (date, not buy), "Intraday round trip"
                prior_date[row.instrument] = (date, buy)
                quantity[row.instrument] += row.filled_adjusted_shares * (1 if buy else -1)
                cash += (-row.value if buy else row.value)-row.cost
                assert quantity.min() > -1e-5 and cash >= -.01
                fills += 1
            held = quantity[quantity.abs() > 1e-5]
            marks = close.loc[date].reindex(held.index)
            assert marks.notna().all(), "Missing held-position mark"
            equity = cash + float((held*marks).sum())
            max_equity_error = max(max_equity_error, abs(equity-snapshot.account))
            max_cash_error = max(max_cash_error, abs(cash-snapshot.cash))
        assert max_equity_error < .1 and max_cash_error < .1
        audits[lane] = {"days": len(report), "fills_checked": fills, "max_equity_error_yuan": max_equity_error,
                         "max_cash_error_yuan": max_cash_error, "passed": True}
    write_json(directory / "independent-audit.json", audits)
    print(json.dumps(audits, indent=2), flush=True)
    return audits


if __name__ == "__main__":
    from replay_hist_reproduction import REPLAY_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    init_qlib()
    if args.smoke:
        audit(OUT / "smoke-not-HIST-results")
    else:
        for seed in ([args.seed] if args.seed is not None else CFG["seeds"]):
            path = OUT / "runs" / f"seed-{seed:02d}" / REPLAY_DIR
            if (path / "replay.json").exists():
                audit(path)
