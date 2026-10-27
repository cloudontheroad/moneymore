"""Read-only historical forecast calibration, separate from portfolio performance."""

import argparse
import hashlib
from pathlib import Path

from moneymore.challenger_diagnostics import prediction_reports, trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.signal_calibration import build_observations, walk_forward_calibration


def run(root, as_of):
    store = ParquetStore(root / "data")
    calendar = trading_dates(store, as_of)
    accounts = {"main_gru": root / "state/qlib-challenger-shadow"}
    accounts.update({"candidate_" + p.name: p for p in (root / "state/qlib-candidate-shadow").iterdir() if p.is_dir()})
    reports = {name: prediction_reports(path, as_of) for name, path in accounts.items()}
    dates = [r["trade_date"] for rs in reports.values() for r in rs]
    filters = [("trade_date", ">=", min(dates)), ("trade_date", "<=", as_of)]
    daily = store.read("daily", columns=["ts_code", "trade_date", "close"], filters=filters)
    adj = store.read("adj_factor", filters=filters)
    bars = daily.merge(adj, on=["ts_code", "trade_date"], validate="one_to_one")
    bars["adjusted"] = bars.close * bars.adj_factor
    prices = bars.pivot(index="trade_date", columns="ts_code", values="adjusted")
    output = {"as_of_date": as_of, "orders_enabled": False, "automatic_promotion": False,
              "protocol": "rank_OLS_date_balanced_min20_purge_maturity_before_test_minus1_session_v1",
              "label": "5_session_adjusted_close_return_minus_recorded_pool_mean",
              "warning": "Calibration diagnostics only; overlapping test labels, no independent portfolio PnL or parameter selection.",
              "accounts": {}}
    output["market_input_sha256"] = {name: hashlib.sha256((store.curated / (name + ".parquet")).read_bytes()).hexdigest()
                                     for name in ("daily", "adj_factor", "trade_calendar")}
    for name, rs in reports.items():
        observations = build_observations(rs, prices, calendar, as_of=as_of)
        folds = walk_forward_calibration(observations, calendar)
        tested = [r for r in folds if "out_of_sample_mse" in r]
        output["accounts"][name] = {
            "recorded_dates": len(rs), "complete_mature_dates": sum(r["valid_label"] and r["prospective"] for r in observations),
            "tested_dates": len(tested), "latest": folds[-1] if folds else None, "folds": folds,
            "label_audit": [{k: v for k, v in r.items() if k not in ("x", "y")} for r in observations],
            "mean_mse_improvement": sum(r["zero_forecast_mse"] - r["out_of_sample_mse"] for r in tested) / len(tested) if tested else None}
    write_diagnostic(root / "state/enhanced-indexing-research/calibration.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.as_of)
    for account, row in result["accounts"].items():
        print(account, row["complete_mature_dates"], row["tested_dates"], (row["latest"] or {}).get("status"))
