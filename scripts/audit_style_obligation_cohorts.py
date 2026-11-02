"""Track original obligations separately from new recurring alerts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
from moneymore.challenger_diagnostics import clean_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--continued", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable cohort audit")
    old_path = args.original/"next-open-repair-chain.parquet"
    new_path = args.continued/"next-open-repair-chain.parquet"
    old = pd.read_parquet(old_path).set_index("obligation_key", verify_integrity=True)
    new = pd.read_parquet(new_path).set_index("obligation_key", verify_integrity=True)
    if set(old.index)-set(new.index):
        raise ValueError("ORIGINAL_OBLIGATIONS_LOST_AFTER_CONTINUATION")
    stable = old.state.isin(["REPAIRED_AFTER_EXECUTED_TRADES", "NATURAL_REENTRY_BEFORE_EXECUTION"])
    if not old.loc[stable, "state"].eq(new.loc[old.index[stable], "state"]).all():
        raise ValueError("PAST_OBLIGATION_STATE_REWRITTEN")
    previous_pending = old.index[old.state.str.startswith("PENDING")]
    result = {"status":"COHORT_AUDITED_NOT_NEW_TRADE_APPROVAL", "orders_enabled":False,
        "original_obligations":len(old), "original_pending":len(previous_pending),
        "original_states_after":dict(Counter(new.loc[old.index, "state"])),
        "original_unresolved_after":int(new.loc[old.index, "state"].str.startswith("PENDING").sum()),
        "previous_pending_results":new.loc[previous_pending].reset_index().to_dict("records"),
        "new_alerts":new.loc[new.index.difference(old.index)].reset_index().to_dict("records"),
        "source_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old_path,new_path]}}
    args.output.mkdir(parents=True)
    (args.output/"results.json").write_text(json.dumps(clean_json(result),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["original_obligations","original_pending","original_states_after","original_unresolved_after"]}))


if __name__ == "__main__":
    main()
