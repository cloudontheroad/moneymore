"""One bounded real integer-solver probe, no broker/paper intents generated."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moneymore.config import BacktestConfig
from moneymore.data.store import ParquetStore
from moneymore.research.integer_execution import plan_integer_execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable evidence")
    root = Path(__file__).resolve().parents[1]
    source = root / "state/style-balance/coverage-rounding-20261102-a"
    styles = pd.read_parquet(source / "estimated-controls-with-unknown-flags.parquet")
    weights = pd.read_parquet(source / "weights-0.1.parquet").reindex(styles.index)
    bars = ParquetStore(root / "data").read("daily", columns=["ts_code", "trade_date", "close"],
        filters=[("trade_date", "==", "20261030"), ("ts_code", "in", list(styles.index))])
    prices = bars.set_index("ts_code").close.reindex(styles.index)
    capital = 1_000_000.
    config = BacktestConfig(initial_cash=capital, commission_rate=.00015, minimum_commission=5,
        stamp_duty_rate=.0005, transfer_fee_rate=.00001, slippage_bps=10, lot_size=100,
        max_position_weight=.15, max_gross_exposure=1., max_drawdown=.2)
    limits = {c: 0. if c.startswith("unknown_") else .03 if c.startswith(("industry_", "board_")) else .10 for c in styles.columns}
    # Conic solver signed-zero residue is not a permitted negative target.
    # Repair only <1e-8 numerical dust, then recheck the complete contract.
    target = weights.weight.clip(lower=0)
    target /= target.sum()
    repair = float(abs(target-weights.weight).max())
    if (repair > 1e-8 or target.max() > .15+1e-6
            or abs(target-weights.benchmark).sum() > .20+1e-6
            or (np.abs((target-weights.benchmark).to_numpy() @ styles.to_numpy()) > pd.Series(limits).reindex(styles.columns).to_numpy()+1e-6).any()):
        raise ValueError("NUMERICAL_REPAIR_CHANGES_CONSTRAINT_CONTRACT")
    result = plan_integer_execution(target, prices, {}, {}, capital, capital, config, styles, limits,
        turnover=.20, time_limit=10., solver_threads=2, benchmark_weights=weights.benchmark,
        style_controls_prestandardized=True)
    args.output.mkdir(parents=True)
    report = {"status": "INTEGER_PROBE_NOT_RELEASE", "orders_enabled": False,
        "capital": capital, "source": str(source), "risk_budget_selected": False,
        "max_target_numerical_repair": repair, "numerical_repair_full_contract_rechecked": True,
        "source_weights_sha256": hashlib.sha256((source / "weights-0.1.parquet").read_bytes()).hexdigest(),
        "style_limits": limits, "result": result,
        "limitations": ["fresh-cash synthetic account, not existing live positions", "no limit-up/down or intraday quote acceptance",
            "zero unknown-active budget is a test, not a chosen production policy", "never submit returned reference orders"]}
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "reason": result.get("reason"),
        "gross": result.get("estimated_gross"), "order_count": len(result["orders"])}))


if __name__ == "__main__":
    main()
