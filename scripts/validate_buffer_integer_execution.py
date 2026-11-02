"""Five paired momentum diagnostics, real PaperBroker, no live accounts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from moneymore.challenger_diagnostics import clean_json
from moneymore.config import BacktestConfig
from moneymore.execution_replay import run_execution_replay
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.portfolio_error_envelope import robust_bounds
from moneymore.research.integer_execution import plan_integer_execution


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);parser.add_argument("--fixture",type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError("immutable integer experiment")
    root=Path(__file__).resolve().parents[1]
    protocol_path=args.fixture/"protocol.yaml" if args.fixture else root/"configs/style_buffer_integer_v1.yaml"
    protocol=yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    source=args.fixture or root/"state/style-balance/portfolio-buffer-heldout-20261102-a"
    envelope_path=root/"state/style-balance/portfolio-buffer-probes-20261102-c/observed-error-envelope.json"
    heldout=json.loads((source/"results.json").read_text(encoding="utf-8"))
    if hashlib.sha256(envelope_path.read_bytes()).hexdigest()!=heldout["source_envelope_sha256"]:raise ValueError("ENVELOPE_CHANGED")
    envelope=json.loads(envelope_path.read_text(encoding="utf-8"))
    evidence=args.fixture/"gates" if args.fixture else root/"state/style-balance/buffer-integer-quotes-20261102-a"
    manifest=json.loads((evidence/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="READY":raise ValueError("OFFICIAL_GATES_NOT_READY")
    for item in manifest["requests"]:
        if hashlib.sha256((evidence/item["file"]).read_bytes()).hexdigest()!=item["sha256"]:raise ValueError("GATE_HASH_CHANGED")
    pre=args.fixture or root/"state/style-balance/preperiod-drift-202609-b"
    parent=args.fixture or root/"state/style-balance/portfolio-buffer-probes-20261102-c"
    quote=pd.read_parquet(parent/"source-price-history.parquet")
    config=BacktestConfig(initial_cash=protocol["capital_each_case"],commission_rate=protocol["commission_rate"],minimum_commission=protocol["minimum_commission"],
        stamp_duty_rate=.0005,transfer_fee_rate=.00001,slippage_bps=protocol["slippage_bps_each_side"],lot_size=100,max_position_weight=.15,max_gross_exposure=1.,max_drawdown=.2)
    paths=sorted(source.glob(f"weights-*-{protocol['probe']}.parquet"))
    transitions=json.loads((parent/"results.json").read_text(encoding="utf-8"))["transition_counts"]
    following={r["from_date"]:r["to_date"] for r in transitions}
    if sorted(following[p.name.split("-")[1]] for p in paths)!=protocol["dates"]:raise ValueError("FIXED_CASE_SET_CHANGED")
    args.output.mkdir(parents=True);rows=[]
    (args.output/"protocol.yaml").write_bytes(protocol_path.read_bytes())
    empty=pd.DataFrame(columns=["date","symbol","target_weight"])
    for path in paths:
        signal=path.name.split("-")[1];date=following[signal]
        raw=pd.read_parquet(pre/f"observations-{signal}.parquet")
        controls=uncertain_controls(raw,CORE)[0];wb=np.exp(raw.log_size);wb/=wb.sum()
        limits=pd.Series({f:.01 if f.startswith("unknown_") else .03 if f.startswith(("industry_","board_")) else .1 for f in controls.columns})
        low,high=robust_bounds(limits,envelope)
        new_raw=pd.read_parquet(pre/f"observations-{date}.parquet");new_controls=uncertain_controls(new_raw,CORE)[0]
        new_wb=np.exp(new_raw.log_size);new_wb/=new_wb.sum();new_centered=new_controls-(new_wb @ new_controls)
        weights=pd.read_parquet(path)
        market=quote.loc[quote.trade_date.isin([signal,date])&quote.ts_code.isin(raw.index),["trade_date","ts_code","open","close"]].copy()
        gate=pd.read_parquet(evidence/f"stk_limit-{date}.parquet").set_index("ts_code")
        susp=pd.read_parquet(evidence/f"suspend_d-{date}.parquet")
        halted=set(susp.loc[susp.suspend_type.eq("S"),"ts_code"])
        market["can_buy"]=[bool(r.ts_code not in halted and r.open*(1+config.slippage_bps/10000)<gate.up_limit.get(r.ts_code,np.nan)-1e-8) for r in market.itertuples()]
        market["can_sell"]=[bool(r.ts_code not in halted and r.open*(1-config.slippage_bps/10000)>gate.down_limit.get(r.ts_code,np.nan)+1e-8) for r in market.itertuples()]
        market=market.rename(columns={"trade_date":"date","ts_code":"symbol"});market["date"]=pd.to_datetime(market.date)
        for variant in protocol["paired_variants"]:
            target=weights[variant];plan_holder={};case_folder=args.output/f"{date}-{variant}";case_folder.mkdir()
            def closing(day,snapshot):return empty.copy()
            def opening(day,snapshot,prices):
                if day!=date:return {"status":"NO_PRIOR_SIGNAL","orders":[]}
                kwargs={} if variant=="unbuffered" else {"style_lower_limits":low,"style_upper_limits":high}
                plan=plan_integer_execution(target,prices.open,{}, {},snapshot["cash"],snapshot["equity"],config,controls,limits,
                    can_buy=prices.can_buy.to_dict(),can_sell=prices.can_sell.to_dict(),benchmark_weights=wb,style_controls_prestandardized=True,
                    time_limit=protocol["integer_feasibility_seconds"],research_staged_warm_start=True,research_target_time_limit=protocol["integer_target_seconds"],
                    solver_threads=2,mip_relative_gap=protocol["integer_mip_relative_gap"],**kwargs)
                plan["signal_date"]=signal
                if plan["status"]=="PASS_REFERENCE_QUOTES_ONLY":
                    plan["execution_style_proof"]={"standardized":(controls-(wb @ controls)).to_dict("index"),"limits":limits.to_dict(),
                        "lower_limits":plan["style_active_lower_limits"],"upper_limits":plan["style_active_upper_limits"]}
                plan_holder.update(plan)
                return plan
            result=run_execution_replay(bars=market,targets=empty,config=config,database=case_folder/"paper.sqlite3",strategy_id="buffer_integer_diagnostic",account_id="isolated",
                closing_target_provider=closing,opening_plan_provider=opening)
            result.equity.to_parquet(case_folder/"equity.parquet",index=False)
            row={"signal_date":signal,"trade_date":date,"variant":variant,"plan_status":plan_holder["status"],"reason":plan_holder.get("reason"),
                "fills":len(result.fills),"fees":result.summary["transaction_cost"],"reconciled":result.reconciliation["matched"],"plan":plan_holder}
            row["actual_execution_check"]=result.summary["opening_plan_audits"][-1].get("execution_check")
            if plan_holder["status"]=="PASS_REFERENCE_QUOTES_ONLY":
                if not result.reconciliation["matched"]:raise ValueError("PAPER_RECONCILIATION_FAILED")
                q=pd.Series({s:v for s,v in plan_holder["target_quantities"].items() if v>0},dtype=float)
                if set(q.index)-set(new_controls.index):raise ValueError("ACTUAL_NEXT_RISK_MISSING")
                marks=market.loc[market.date.eq(pd.Timestamp(date))].set_index("symbol")
                value=q*marks.close.reindex(q.index);exposure=(value/value.sum()) @ new_centered.reindex(index=q.index,columns=controls.columns)
                excess=abs(exposure)-limits
                row.update(close_breach=bool((excess>1e-6).any()),close_breach_factors=int((excess>1e-6).sum()),maximum_close_excess=max(0.,float(excess.max())),
                    target_tracking_l1=plan_holder["tracking_l1"],cash=plan_holder["estimated_cash"],gross=plan_holder["estimated_gross"],holdings=len(q),
                    slippage_currency=float(sum(abs(f["price"]-marks.loc[f["symbol"],"open"])*f["quantity"] for f in result.fills)))
                for f in result.fills:
                    if not gate.loc[f["symbol"],"down_limit"]-1e-8<=f["price"]<=gate.loc[f["symbol"],"up_limit"]+1e-8:raise ValueError("FILL_OUTSIDE_LIMIT")
            rows.append(row)
            (args.output/"progress.json").write_text(json.dumps(clean_json(rows),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
            print(json.dumps({k:row[k] for k in ["trade_date","variant","plan_status","reason","fills"]}),flush=True)
    summary={}
    for variant in protocol["paired_variants"]:
        x=[r for r in rows if r["variant"]==variant];ok=[r for r in x if r["plan_status"]=="PASS_REFERENCE_QUOTES_ONLY"]
        summary[variant]={"cases":len(x),"executed":len(ok),"blocked":len(x)-len(ok),"close_breaches":sum(r["close_breach"] for r in ok) if ok else None,
            "mean_fee":float(np.mean([r["fees"] for r in ok])) if ok else None,"mean_target_tracking_l1":float(np.mean([r["target_tracking_l1"] for r in ok])) if ok else None}
    paired=[d for d in protocol["dates"] if all(any(r["trade_date"]==d and r["variant"]==v and r["plan_status"]=="PASS_REFERENCE_QUOTES_ONLY" for r in rows) for v in protocol["paired_variants"])]
    report={"status":"PAIRED_INTEGER_PAPER_DIAGNOSTIC_NOT_DEPLOYED","orders_enabled":False,"summary":summary,"paired_success_dates":paired,
        "protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),"source_envelope_sha256":hashlib.sha256(envelope_path.read_bytes()).hexdigest(),
        "cases":rows,"limitations":protocol["excluded_claims"],"study_scope":protocol["scope"],"buffer_adopted":False}
    (args.output/"results.json").write_text(json.dumps(clean_json(report),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"summary":summary,"paired_success_dates":paired}))


if __name__=="__main__":main()
