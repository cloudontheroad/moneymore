"""Independent SQLite readback, tightened-bound verification and honest denominators."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import yaml


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--study",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError("immutable study audit")
    source=args.study/"results.json";report=json.loads(source.read_text(encoding="utf-8"))
    protocol=yaml.safe_load((args.study/"protocol.yaml").read_text(encoding="utf-8"));verified=[]
    for case in report["cases"]:
        database=args.study/f"{case['trade_date']}-{case['variant']}"/"paper.sqlite3"
        with sqlite3.connect(database.resolve().as_uri()+"?mode=ro",uri=True) as con:
            fills=pd.read_sql_query("SELECT * FROM fills",con)
            positions=pd.read_sql_query("SELECT symbol,quantity FROM positions WHERE account_id='isolated' AND quantity>0",con).set_index("symbol").quantity
            cash=float(con.execute("SELECT cash FROM accounts WHERE account_id='isolated'").fetchone()[0])
        rebuilt=float(protocol["capital_each_case"])+sum((1 if r.side=="SELL" else -1)*r.price*r.quantity-r.fee for r in fills.itertuples())
        if abs(rebuilt-cash)>.01 or len(fills)!=case["fills"]:raise ValueError("INDEPENDENT_CASH_OR_FILL_MISMATCH")
        plan=case["plan"];success=plan["status"]=="PASS_REFERENCE_QUOTES_ONLY"
        if not success:
            if len(fills) or len(positions):raise ValueError("BLOCKED_CASE_EXECUTED")
        else:
            expected={s:int(q) for s,q in plan["target_quantities"].items() if q>0}
            if positions.to_dict()!=expected or abs(cash-plan["estimated_cash"])>.01:raise ValueError("PLAN_TO_LEDGER_MISMATCH")
            proof=plan["execution_style_proof"]
            z=pd.DataFrame.from_dict(proof["standardized"],orient="index")
            lo=pd.Series(plan["style_active_lower_limits"]).reindex(z.columns)
            hi=pd.Series(plan["style_active_upper_limits"]).reindex(z.columns)
            if not lo.equals(pd.Series(proof["lower_limits"]).reindex(z.columns)) or not hi.equals(pd.Series(proof["upper_limits"]).reindex(z.columns)):
                raise ValueError("EXECUTION_PROOF_DIFFERS_FROM_INTEGER_CONTRACT")
            prices=pd.Series({o["symbol"]:o["reference_price"] for o in plan["orders"]})
            values=positions*prices.reindex(positions.index)
            active=(values/values.sum()) @ z.reindex(values.index)
            if not np.isfinite(active).all() or (active<lo-1e-5).any() or (active>hi+1e-5).any():raise ValueError("ACTUAL_LEDGER_ASYMMETRIC_BREACH")
            for symbol,q in positions.items():
                if q%100 or (symbol.startswith(("688","689")) and q<200):raise ValueError("LOT_RULE_FAILED")
        verified.append({"date":case["trade_date"],"variant":case["variant"],"executed":success,"fills":len(fills),"cash_error":abs(rebuilt-cash),
            "close_breach":case.get("close_breach") if success else None})
    summary={}
    for variant in protocol["paired_variants"]:
        rows=[r for r in verified if r["variant"]==variant];executed=[r for r in rows if r["executed"]]
        summary[variant]={"cases":len(rows),"executed":len(executed),"blocked":len(rows)-len(executed),
            "measured_close_breaches":sum(r["close_breach"] for r in executed) if executed else None}
    result={"status":"INDEPENDENT_SQLITE_AND_ASYMMETRIC_CONTRACT_AUDIT_PASSED","orders_enabled":False,"summary":summary,"verified_cases":verified,
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"warning":"No executions means unknown close-risk metric, never zero risk. Close outcomes are source measurements; this audit independently rebuilds opening holdings/cash/exposures."}
    args.output.mkdir(parents=True);(args.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8");print(json.dumps(summary))


if __name__=="__main__":main()
