"""Fixed September heldout allocations with/without the frozen error envelope."""
import argparse
import hashlib
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from qlib.model.riskmodel import StructuredCovEstimator

from moneymore.challenger_diagnostics import trading_dates
from moneymore.data.store import ParquetStore
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.portfolio_error_envelope import robust_bounds
from moneymore.research.style_replay_inputs import optimize_daily


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable heldout comparison")
    root=Path(__file__).resolve().parents[1]
    source=root/"state/style-balance/portfolio-buffer-probes-20261102-c"
    calibration=root/"state/style-balance/preperiod-drift-202609-b"
    source_report=json.loads((source/"results.json").read_text(encoding="utf-8"))
    for path,digest in source_report["input_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError("CALIBRATION_INPUT_CHANGED")
    envelope_path=source/"observed-error-envelope.json"
    envelope=json.loads(envelope_path.read_text(encoding="utf-8"))
    bars=pd.read_parquet(source/"source-price-history.parquet")
    if hashlib.sha256((source/"source-price-history.parquet").read_bytes()).hexdigest()!=source_report["price_history_sha256"]:
        raise ValueError("PRICE_HISTORY_CHANGED")
    prices=bars.pivot(index="trade_date",columns="ts_code",values="total_price")
    quote=bars.set_index(["trade_date","ts_code"])
    calendar=trading_dates(ParquetStore(root/"data"),"20260930")
    args.output.mkdir(parents=True)
    rows=[];factor_rows=[]
    for case in source_report["transition_counts"]:
        date,next_date=case["from_date"],case["to_date"]
        if next_date<"20260923":
            continue
        raw=pd.read_parquet(calibration/f"observations-{date}.parquet")
        new_raw=pd.read_parquet(calibration/f"observations-{next_date}.parquet")
        controls=uncertain_controls(raw,CORE)[0];new=uncertain_controls(new_raw,CORE)[0]
        wb=np.exp(raw.log_size);wb/=wb.sum();new_wb=np.exp(new_raw.log_size);new_wb/=new_wb.sum()
        limits=pd.Series({f:.01 if f.startswith("unknown_") else .03 if f.startswith(("industry_","board_")) else .1 for f in controls.columns})
        lower,upper=robust_bounds(limits,envelope)
        days=[d for d in calendar if d<=date][-121:]
        returns=prices.reindex(index=days,columns=raw.index).pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("REQUIRED_RETURN_GAP")
        np.random.seed(17)
        F,cov,specific=StructuredCovEstimator(num_factors=10,factor_model="pca",scale_return=False).predict(returns,is_price=False,return_decomposed_components=True)
        choices={"cap_reference":None,"momentum60":controls.momentum60,"book_to_price":controls.book_to_price,
            "quality_roe":controls.quality_roe,"negative_residual_volatility120":-controls.residual_volatility120}
        original=pd.read_parquet(source/f"diagnostic-weights-{date}.parquet")
        centered=controls-(wb @ controls);new_centered=new-(new_wb @ new)
        for probe,score in choices.items():
            base=original[probe].reindex(raw.index)
            try:
                if score is None:
                    w=cp.Variable(len(raw));active=w-wb.to_numpy()
                    p=cp.Problem(cp.Minimize(cp.sum_squares(active)),[w>=0,w<=.15,cp.sum(w)==1,
                        controls.to_numpy().T @ active>=lower.reindex(controls.columns).to_numpy(),controls.to_numpy().T @ active<=upper.reindex(controls.columns).to_numpy()])
                    p.solve(solver=cp.CLARABEL,max_iter=300,tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
                    if p.status!="optimal":
                        raise ValueError("REFERENCE_BUFFER_"+p.status)
                    buffered=pd.Series(w.value,index=raw.index).clip(lower=0);buffered/=buffered.sum()
                else:
                    buffered=optimize_daily(score,raw,controls,F,cov,specific,wb,limits,pd.Series(dtype=float),lower_limits=lower,upper_limits=upper)
                support=buffered[buffered>1e-8]
                if buffered.sum()-support.sum()>1e-6:
                    raise ValueError("NONTRIVIAL_NUMERIC_DUST")
                support/=support.sum();buffered=support.reindex(raw.index,fill_value=0.)
                exposure=buffered @ centered
                if ((exposure<lower-1e-6).any() or (exposure>upper+1e-6).any() or buffered.max()>.15+1e-6):
                    raise ValueError("BUFFERED_POSTCHECK_FAILED")
            except ValueError as error:
                rows.append({"from_date":date,"to_date":next_date,"probe":probe,"variant":"buffered","status":"BLOCKED_BUFFERED_ALLOCATION","reason":str(error)})
                buffered=None
            target_distance=float(abs(buffered-base).sum()) if buffered is not None else None
            for variant,target in [("unbuffered",base)]+([("buffered",buffered)] if buffered is not None else []):
                support=target[target>0]
                q=quote.loc[next_date].reindex(support.index)
                if (set(support.index)-set(new.index) or not np.isfinite(q[["open","close"]].to_numpy()).all()
                        or set(centered.columns)-set(new_centered.columns)):
                    rows.append({"from_date":date,"to_date":next_date,"probe":probe,"variant":variant,"status":"UNMEASURABLE_NO_FILL"});continue
                values=support*q.close/q.open;end_weight=values/values.sum()
                outcome=end_weight @ new_centered.reindex(index=support.index,columns=centered.columns)
                excess=abs(outcome)-limits
                for factor in controls.columns:
                    factor_rows.append({"to_date":next_date,"probe":probe,"variant":variant,"factor":factor,"outcome":float(outcome[factor]),
                        "original_limit":float(limits[factor]),"breach":bool(excess[factor]>1e-6),"excess":max(0.,float(excess[factor]))})
                rows.append({"from_date":date,"to_date":next_date,"probe":probe,"variant":variant,"status":"MEASURED",
                    "breach_factors":int((excess>1e-6).sum()),"maximum_excess":max(0.,float(excess.max())),
                    "target_distance_l1":target_distance,"effective_holdings":float(1/(target @ target))})
            if buffered is not None:
                pd.DataFrame({"unbuffered":base,"buffered":buffered}).to_parquet(args.output/f"weights-{date}-{probe}.parquet")
        print(json.dumps({"to_date":next_date,"completed_probes":len(choices)}),flush=True)
    table=pd.DataFrame(rows);factors=pd.DataFrame(factor_rows)
    table.to_parquet(args.output/"portfolio-outcomes.parquet",index=False);factors.to_parquet(args.output/"factor-outcomes.parquet",index=False)
    summary={}
    for variant in ["unbuffered","buffered"]:
        x=table.loc[table.get("variant").eq(variant)&table.status.eq("MEASURED")]
        summary[variant]={"measured_portfolio_days":len(x),"breached_portfolio_days":int(x.breach_factors.gt(0).sum()) if len(x) else 0,
            "breached_factor_rows":int(x.breach_factors.sum()) if len(x) else 0,"maximum_excess":float(x.maximum_excess.max()) if len(x) else None}
    measured_distances=table.get("target_distance_l1",pd.Series(dtype=float)).dropna()
    result={"status":"HELDOUT_ALLOCATION_BUFFER_COMPARISON_NOT_ADOPTED","orders_enabled":False,"summary":summary,
        "blocked_cases":table.loc[~table.status.eq("MEASURED")].fillna("").to_dict("records"),
        "mean_target_distance_l1":float(measured_distances.mean()) if len(measured_distances) else None,
        "source_envelope_sha256":hashlib.sha256(envelope_path.read_bytes()).hexdigest(),"buffer_adopted":False,
        "limitations":["five heldout dates and synthetic fractional diagnostic portfolios, not actual paper/live deployment",
            "original envelope full-vector coverage failures remain; this does not change that acceptance measure",
            "no costs/lots/turnover or partial fills accepted here","no October returns used or new envelope parameters chosen"]}
    (args.output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"summary":summary,"blocked":len(result["blocked_cases"]),"mean_target_distance_l1":result["mean_target_distance_l1"]}))


if __name__=="__main__":
    main()
