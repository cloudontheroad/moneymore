"""Official corporate actions for the isolated style experiment, not live ingestion."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv

from moneymore.challenger_diagnostics import write_diagnostic
from moneymore.data.store import ParquetStore


def run(root):
    preflight = json.loads((root / "state/enhanced-indexing-research/replay-preflight.json").read_text(encoding="utf-8"))
    evidence = Path(preflight["evidence_directory"])
    dates = preflight["suspension_response_dates"]
    load_dotenv(root / ".env")
    from moneymore.data.tushare_provider import TushareProvider
    client = TushareProvider()._client
    requests, frames = [], []
    for date in dates:
        for field in ("ex_date", "record_date"):
            path = evidence / f"dividend-{field}-{date}.parquet"
            try:
                cached = path.exists()
                data = pd.read_parquet(path) if cached else client.dividend(**{field: date})
                if data is None:
                    raise ValueError("NULL_RESPONSE")
                if not data.empty and not data[field].astype(str).eq(date).all():
                    raise ValueError("PROVIDER_FILTER_NOT_HONORED")
                if not cached:
                    data.to_parquet(path, index=False)
                    time.sleep(1.1)
                frames.append(data)
                requests.append(dict(date=date, field=field, rows=len(data), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            except Exception as error:
                requests.append(dict(date=date, field=field, error_type=type(error).__name__))
    actions = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
    action_file = evidence / "dividend-window.parquet"
    actions.to_parquet(action_file, index=False)
    store = ParquetStore(root / "data")
    # Detect material adjustment changes not explained by official ex-date events.
    factors = store.read("adj_factor", filters=[("trade_date", ">=", preflight["start_date"]),
                                                ("trade_date", "<=", preflight["as_of_date"])])
    factors = factors.sort_values(["ts_code", "trade_date"])
    factors["ratio"] = factors.groupby("ts_code").adj_factor.pct_change(fill_method=None)
    # Input rounding (4 decimals vs source float) is not a corporate action.
    jumps = factors.loc[factors.ratio.abs().gt(.0001)]
    explained = set(zip(actions.ts_code, actions.ex_date)) if not actions.empty else set()
    unexplained = jumps.loc[[tuple(x) not in explained for x in jumps[["ts_code", "trade_date"]].itertuples(index=False, name=None)]]
    result = dict(as_of_date=preflight["as_of_date"], orders_enabled=False,
                  requests=requests, official_action_rows=len(actions), action_file=str(action_file),
                  material_adjustment_jumps=len(jumps), unexplained_adjustment_jumps=unexplained.to_dict("records"),
                  status="READY_FOR_CA_REPLAY_AUDIT" if not any("error_type" in r for r in requests) else "BLOCKED_PROVIDER_REQUESTS",
                  tolerance=.0001, warning="Adjustment jumps are a cross-check, not proof of full action coverage. No production data changed.")
    write_diagnostic(root / "state/enhanced-indexing-research/corporate-action-audit.json", result)
    print({k: v for k, v in result.items() if k not in ("requests", "unexplained_adjustment_jumps")})
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run(Path(__file__).resolve().parents[1])
