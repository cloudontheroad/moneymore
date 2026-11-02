"""Independent read-only replay ledger audit; writes only a new audit artifact."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

from moneymore.research.replay_attribution import reconstruct_cash, reconstruct_positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-name", default="independent-audit.json", choices=["independent-audit.json", "independent-t1-audit.json"])
    args = parser.parse_args()
    output = args.directory/args.output_name
    if output.exists():
        raise FileExistsError("immutable audit")
    source = args.directory/"results.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    model_path = root/"state/qlib-challenger/candidates/20261010/research.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    earliest = pd.Timestamp(report["start"], tz="Asia/Shanghai").tz_convert("UTC")
    if pd.to_datetime(model["created_at"], utc=True) >= earliest or pd.Timestamp(model["data_cutoff"]) >= pd.Timestamp(report["start"]):
        raise ValueError("MODEL_NOT_AVAILABLE_BEFORE_REPLAY")
    for path, digest in report["input_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("INPUT_HASH_CHANGED")
    limit_files = [Path(p) for p in report["input_hashes"] if Path(p).name.startswith("stk_limit-")]
    limits = pd.concat([pd.read_parquet(p) for p in limit_files]).set_index(["trade_date", "ts_code"])
    database = args.directory/"paper.sqlite3"
    with sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        fills = pd.read_sql_query("SELECT * FROM fills", connection)
        ledger = pd.read_sql_query("SELECT * FROM corporate_action_ledger", connection)
        final = pd.read_sql_query("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0", connection).set_index("symbol").quantity.to_dict()
    equity = pd.read_parquet(args.directory/"equity.parquet").set_index("trade_date")
    dates = equity.index.astype(str).tolist()
    cash = reconstruct_cash(fills, ledger, dates, 1_000_000.)
    positions = reconstruct_positions(fills, ledger, dates)
    max_error = max(abs(cash[d]-equity.loc[d,"cash"]) for d in dates)
    if max_error > .01 or positions[dates[-1]] != final:
        raise ValueError("INDEPENDENT_LEDGER_MISMATCH")
    previous = {}
    for date in dates:
        matured = ledger.loc[ledger.trade_date.astype(str).eq(date) & ledger.action_type.eq("STOCK_DIVIDEND")].groupby("symbol").share_quantity.sum().to_dict()
        sells = fills.loc[fills.trade_date.astype(str).eq(date) & fills.side.eq("SELL")].groupby("symbol").quantity.sum()
        if any(q > previous.get(s, 0)+matured.get(s, 0) for s,q in sells.items()):
            raise ValueError("T_PLUS_ONE_VIOLATION")
        previous = positions[date]
    for fill in fills.itertuples():
        bound = limits.loc[(str(fill.trade_date), fill.symbol)]
        if not bound.down_limit-1e-8 <= fill.price <= bound.up_limit+1e-8:
            raise ValueError("FILL_PRICE_OUTSIDE_OFFICIAL_LIMITS")
    opens = [x for x in report["summary"]["opening_plan_audits"] if x["plan"]["status"] == "PASS_REFERENCE_QUOTES_ONLY"]
    for event in opens:
        date = event["trade_date"]; plan = event["plan"]
        expected = {s: int(q) for s,q in plan["target_quantities"].items() if q>0}
        if positions[date] != expected or abs(cash[date]-plan["estimated_cash"]) > .01:
            # CA quantity/cash after an open needs explicit event-level
            # adjustment; never silently compare unmatched daily endpoints.
            raise ValueError("PLAN_TO_LEDGER_REQUIRES_CA_ADJUSTMENT_OR_IS_MISMATCHED")
    audit = {"status": "PASS_ENGINEERING_LEDGER_ONLY", "orders_enabled": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "fills": len(fills), "opens": len(opens),
        "cash_max_error": max_error, "final_positions_match": True, "official_limit_prices_match": True,
        "plan_to_independent_ledger_match": True, "corporate_ledger_rows": len(ledger)}
    audit["t_plus_one_sell_availability_match"] = True
    audit["model_record_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    audit["model_created_at"] = model["created_at"]
    audit["model_data_cutoff"] = model["data_cutoff"]
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
