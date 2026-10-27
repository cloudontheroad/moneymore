"""Isolated one-step feasibility interventions; never execute probe plans."""

import importlib.util
import json
from pathlib import Path

from moneymore.challenger_diagnostics import trading_dates, write_diagnostic
from moneymore.data.store import ParquetStore
from moneymore.research.integer_execution import plan_integer_execution


def run(root):
    spec = importlib.util.spec_from_file_location(
        "probe_style_replay", root / "scripts/replay_candidate_styles.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    store = ParquetStore(root / "data")
    preflight = json.loads(
        (root / "state/enhanced-indexing-research/replay-preflight.json").read_text(
            encoding="utf-8"
        )
    )
    dates = [
        d
        for d in trading_dates(store, preflight["as_of_date"])
        if preflight["start_date"] <= d <= preflight["as_of_date"]
    ]
    close_quotes = store.read("daily").pivot(index="trade_date", columns="ts_code", values="close")
    calls = 0

    def probe(**kwargs):
        nonlocal calls
        signal_date = dates[calls % (len(dates) - 1)]
        calls += 1
        original = plan_integer_execution(**kwargs)
        if original["status"] != "PASS_REFERENCE_QUOTES_ONLY":
            item = {
                "case": len(cases) + 1,
                "original_reason": original.get("reason"),
                "cash": kwargs["cash"],
                "equity": kwargs["equity"],
                "probes": {},
            }
            old_quotes = close_quotes.loc[signal_date]
            old_equity = kwargs["cash"] + sum(
                q * old_quotes[s] for s, q in kwargs["quantities"].items()
            )
            for name, overrides in (
                (
                    "signal_close_prices_DIAGNOSTIC_NOT_EXECUTABLE_TODAY",
                    {"prices": old_quotes, "equity": old_equity},
                ),
                ("turnover_25pct_DIAGNOSTIC_NOT_ALLOWED_POLICY", {"turnover": 0.25}),
                (
                    "no_required_style_reduction_DIAGNOSTIC",
                    {"style_limits": {k: v / 0.75 for k, v in kwargs["style_limits"].items()}},
                ),
            ):
                candidate = plan_integer_execution(**{**kwargs, **overrides})
                item["probes"][name] = {
                    k: v for k, v in candidate.items() if k not in ("orders", "target_quantities")
                }
            # The exact date and signal are assigned after replay from event order.
            item["input_signal_date"] = signal_date
            cases.append(item)
        return original  # Probes never replace orders or alter the replay path.

    module.plan_integer_execution = probe
    replay = module.run(root, "candidate_20261010", integer=True)
    events = [
        e
        for row in replay["runs"]
        for e in row.get("summary", {}).get("opening_plan_audits", [])
        if e["plan"]["status"] == "BLOCKED"
    ]
    if len(events) != len(cases):
        raise ValueError("PROBE_EVENT_ALIGNMENT_FAILED")
    for case, event in zip(cases, events, strict=True):
        if case["input_signal_date"] != event["plan"]["signal_date"]:
            raise ValueError("PROBE_SIGNAL_DATE_ALIGNMENT_FAILED")
        case.update(date=event["trade_date"], signal_date=event["plan"]["signal_date"])
    output = {
        "orders_enabled": False,
        "cases": cases,
        "warning": "One-step constraint interventions, no probe trades submitted. Relaxed constraints are not production permission.",
    }
    write_diagnostic(
        root / "state/enhanced-indexing-research/integer-feasibility-probes-1010.json", output
    )
    print([(r["date"], {k: v["status"] for k, v in r["probes"].items()}) for r in cases])
    return output


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
