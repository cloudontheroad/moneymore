"""Verify accelerated constructor against archived pre-acceleration real-account targets."""

import json
from pathlib import Path

import pandas as pd

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.baseline_v2_daily_readiness import construct_daily_targets
from moneymore.research.baseline_v2_daily_replay import frozen_prices

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "state/baseline-v2-validation/daily-inputs"
    source = base / "daily-replays/20261028T123054Z-aca37d53"
    frozen = json.loads((source / "frozen-protocol.json").read_text(encoding="utf-8"))["scoring"]
    scores_path = next(
        Path(p)
        for p in json.loads((source / "frozen-protocol.json").read_text(encoding="utf-8"))[
            "input_hashes"
        ]
        if Path(p).name == "daily-scores.parquet"
    )
    scores = pd.read_parquet(scores_path)
    pool = pd.read_parquet(
        Path(frozen["identity_audit"]["output_directory"]) / "identity-top1000-preview.parquet"
    )
    store = ParquetStore(root / "data")
    prices = frozen_prices(root, store, frozen, sorted(pool.security_id.unique()))
    calendar = trading_dates(store, "20261028")
    count, maximum = 0, 0.0
    for variant in ("fixed", "learned"):
        decisions = json.loads(
            (source / (variant + "-closing-decisions.json")).read_text(encoding="utf-8")
        )["events"]
        for event in decisions:
            date = event["signal_date"]
            frame = scores.loc[scores.date.eq(pd.Timestamp(date))].rename(
                columns={"score_" + variant: "score"}
            )
            selected, targets, _ = construct_daily_targets(
                frame,
                event["actual_incumbents"],
                prices.reindex(columns=frame.symbol),
                date,
                calendar[calendar.index(date) + 1],
                frozen["policy"],
                universe_members=set(pool.loc[pool.signal_date.eq(date), "security_id"]),
            )
            if selected != event["selected"] or set(targets) != set(event["weights"]):
                raise ValueError("ACCELERATED_CONSTRUCTOR_CHANGED_SELECTION")
            error = max(abs(targets[s] - event["weights"][s]) for s in targets)
            if error > 1e-12:
                raise ValueError("ACCELERATED_CONSTRUCTOR_CHANGED_WEIGHTS")
            count += 1
            maximum = max(maximum, error)
    result = {
        "status": "ARCHIVED_ACTUAL_INCUMBENT_TARGET_EQUIVALENCE_PASS",
        "checked_closings": count,
        "maximum_weight_difference": maximum,
        "source_run": str(source),
        "real_orders_enabled": False,
    }
    write_diagnostic(base / "constructor-equivalence-audit.json", result)
    print(json.dumps(result))
