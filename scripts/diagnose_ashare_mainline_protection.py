"""Identify first held-input blockers without changing an active research run."""

import argparse
import json

import numpy as np
import pandas as pd
from replay_ashare_mainline import BASE, Planner, engine
from threadpoolctl import threadpool_limits

from moneymore.research.enhanced_indexing import carry_confirmed_suspensions


def main(version):
    run = BASE / f"n3-full-matrix-{version}/cost10"
    checkpoint = json.loads((run / "bps10/checkpoint.json").read_bytes())
    holdings = pd.DataFrame(checkpoint["holdings"])
    contract = json.loads((run / "contract.json").read_bytes())
    planner = Planner(run, contract, engine.load_inputs())
    rows = []
    for kind in sorted(holdings.kind.unique()):
        failures = [
            (p, json.loads(p.read_bytes()))
            for p in sorted((run / "decisions" / kind).glob("*.json"))
        ]
        failures = [
            (p, d)
            for p, d in failures
            if d.get("reason") in ("HELD_RISK_INPUT_MISSING", "HELD_RISK_HISTORY_MISSING")
        ]
        if not failures:
            continue
        path, decision = failures[0]
        signal = decision["signal_date"]
        held = set(holdings.loc[holdings.kind.eq(kind) & holdings.date.eq(signal), "symbol"])
        dates = [d for d in planner.calendar if d <= signal][-121:]
        raw = planner.prices.reindex(index=dates, columns=sorted(planner.membership[signal] | held))
        proof = planner.proofs.loc[planner.proofs.trade_date.astype(str).le(signal)]
        history, _ = carry_confirmed_suspensions(raw, proof)
        returns = history.pct_change(fill_method=None).iloc[1:]
        attrs = planner.attributes[signal]
        _, eligible, _ = planner.risk(signal)
        missing = decision.get(
            "missing_symbols", sorted((held & planner.membership[signal]) - set(eligible.index))
        )
        for symbol in missing:
            before = planner.prices.loc[planner.prices.index < dates[0], symbol].dropna()
            score_missing = [
                case for case in contract["models"] if symbol not in planner.scores[case, signal]
            ]
            row = {
                "case": kind,
                "execution": path.stem,
                "signal": signal,
                "symbol": symbol,
                "risk_start": dates[0],
                "last_pre_window_close_date": before.index[-1] if len(before) else None,
                "raw_missing_dates": raw.index[raw[symbol].isna()].tolist(),
                "unresolved_missing_dates": history.index[history[symbol].isna()].tolist(),
                "bad_return_dates": returns.index[~np.isfinite(returns[symbol])].tolist(),
                "missing_scores": score_missing,
                "attrs": attrs.loc[symbol].to_dict() if symbol in attrs.index else None,
                "halt_rows": proof.loc[proof.ts_code.eq(symbol)].astype(str).to_dict("records"),
            }
            rows.append(row)
    dest = BASE / f"n3-first-blockers-{version}.json"
    dest.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rows, indent=2, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.version)
