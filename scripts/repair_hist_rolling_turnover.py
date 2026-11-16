"""Diagnose/correct float summation in turnover statistics, never trading decisions."""
import argparse
import json
import math
from pathlib import Path

import finalize_hist_rolling as finalizer
import pandas as pd
from prepare_hist_reproduction import digest, write_json

DEST = finalizer.FINAL/"turnover-precision"


def compensated_turnover(self, value):
    # Reset clears .to; a new account must not inherit a prior compensation.
    correction = getattr(self, "_hist_turnover_compensation", 0.) if self.to else 0.
    adjusted = float(value)-correction
    updated = self.to+adjusted
    self._hist_turnover_compensation = (updated-self.to)-adjusted
    self.to = updated


def run(seed, mode):
    import qlib.backtest as backtest_module
    from qlib.backtest.account import AccumulatedInfo

    folder = DEST/f"seed-{seed:02d}"/mode
    folder.mkdir(parents=True, exist_ok=True)
    original = backtest_module.backtest
    if mode == "stable":
        AccumulatedInfo.add_turnover = compensated_turnover

    def observed_backtest(*args, **kwargs):
        result = original(*args, **kwargs)
        report, positions = result[0]["1day"]
        orders = pd.DataFrame(kwargs["executor"].journal)
        lane = kwargs["exchange_kwargs"]["exchange"].buy_price
        # Calls are deterministically open equal-weight, close equal-weight, close dropout.
        number = len(list(folder.glob("*-totals.json")))
        name = f"{number:02d}"
        exact = math.fsum(orders.value)
        detail = {"lane_price": str(lane), "orders": len(orders), "fsum_turnover": exact,
                  "reported_turnover": float(report.total_turnover.iloc[-1]),
                  "sum_error_yuan": float(report.total_turnover.iloc[-1]-exact),
                  "pandas_sum_error_yuan": float(orders.value.sum()-exact),
                  "replay_statistics_only": True}
        write_json(folder/f"{name}-totals.json", detail)
        report.to_parquet(folder/f"{name}-daily.parquet")
        orders.to_parquet(folder/f"{name}-orders.parquet", index=False)
        pd.DataFrame([{"date": d, "instrument": s, "quantity": p.get_stock_amount(s)}
                      for d, p in positions.items() for s in p.get_stock_list()]).to_parquet(folder/f"{name}-holdings.parquet", index=False)
        return result

    backtest_module.backtest = observed_backtest
    finalizer.finalize_one(seed, "frozen")
    if mode == "stable":
        old = folder.parent/"inspect"
        comparisons = []
        for path in old.glob("*-totals.json"):
            name = path.name.removesuffix("-totals.json")
            for kind in ("orders", "holdings"):
                a, b = [pd.read_parquet(d/f"{name}-{kind}.parquet") for d in (old, folder)]
                keys = ["date", "instrument"]+(["direction"] if kind == "orders" else [])
                pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True), check_exact=True)
            a, b = [pd.read_parquet(d/f"{name}-daily.parquet") for d in (old, folder)]
            finalizer.compare_replay_tables(a.drop(columns=["turnover", "total_turnover"]), b.drop(columns=["turnover", "total_turnover"]), "daily")
            comparisons.append(name)
        assert comparisons, "No original failure evidence"
        proof = {"source_sha256": digest(Path(__file__)), "patch": "Kahan sum for reported turnover only",
                 "unchanged_orders_holdings_equity": comparisons,
                 "before": [json.loads(p.read_text()) for p in old.glob("*-totals.json")],
                 "after": [json.loads(p.read_text()) for p in folder.glob("*-totals.json")]}
        assert all(abs(r["sum_error_yuan"]) < .001 for r in proof["after"])
        write_json(folder.parent/"verification.json", proof)
        complete = finalizer.FINAL/"frozen"/f"seed-{seed:02d}"/"complete.json"
        data = json.loads(complete.read_text())
        for row in data["runs"].values():
            row["turnover_accumulator_repair"] = proof
        write_json(complete, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "stable"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.seed, args.mode)
