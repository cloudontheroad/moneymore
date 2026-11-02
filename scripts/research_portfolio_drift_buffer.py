"""Predeclared September diagnostic portfolio errors, not trading strategies."""
import argparse
import hashlib
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import yaml
from qlib.model.riskmodel import StructuredCovEstimator

from moneymore.data.store import ParquetStore
from moneymore.challenger_diagnostics import trading_dates
from moneymore.research.style_missing_controls import uncertain_controls
from moneymore.research.frozen_risk_scale import CORE
from moneymore.research.style_replay_inputs import optimize_daily
from moneymore.research.portfolio_error_envelope import fit_envelope,robust_bounds


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable probe research")
    root=Path(__file__).resolve().parents[1]
    protocol_path=root/"configs/style_buffer_probe_v1.yaml"
    protocol=yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    source=root/"state/style-balance/preperiod-drift-202609-b"
    paths=sorted(source.glob("observations-*.parquet"))
    frames={p.stem.split("-")[-1]:pd.read_parquet(p) for p in paths}
    dates=sorted(frames); symbols=sorted({s for raw in frames.values() for s in raw.index})
    store=ParquetStore(root/"data");calendar=trading_dates(store,dates[-1])
    filters=[("trade_date",">=",calendar[calendar.index(dates[0])-120]),("trade_date","<=",dates[-1]),("ts_code","in",symbols)]
    bars=store.read("daily",columns=["ts_code","trade_date","open","close"],filters=filters)
    bars=bars.merge(store.read("adj_factor",columns=["ts_code","trade_date","adj_factor"],filters=filters),on=["ts_code","trade_date"],how="left",validate="one_to_one")
    bars["total_price"]=bars.close*bars.adj_factor
    prices=bars.pivot(index="trade_date",columns="ts_code",values="total_price")
    quote=bars.set_index(["trade_date","ts_code"])
    args.output.mkdir(parents=True)
    (args.output/"protocol.yaml").write_bytes(protocol_path.read_bytes())
    bars.to_parquet(args.output/"source-price-history.parquet",index=False)
    errors=[];unmeasurable=[];case_counts=[]
    controls_by_date={d:uncertain_controls(raw,CORE)[0] for d,raw in frames.items()}
    for i,date in enumerate(dates[:-1]):
        next_date=dates[i+1];raw=frames[date];controls=controls_by_date[date]
        wb=np.exp(raw.log_size);wb/=wb.sum()
        limits=pd.Series({f:.01 if f.startswith("unknown_") else .03 if f.startswith(("industry_","board_")) else .1 for f in controls.columns})
        days=[d for d in calendar if d<=date][-121:]
        returns=prices.reindex(index=days,columns=raw.index).pct_change(fill_method=None).iloc[1:]
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError("PREPERIOD_REQUIRED_RETURN_GAP")
        np.random.seed(17)
        F,cov,specific=StructuredCovEstimator(num_factors=10,factor_model="pca",scale_return=False).predict(returns,is_price=False,return_decomposed_components=True)
        new=controls_by_date[next_date]
        new_wb=np.exp(frames[next_date].log_size);new_wb/=new_wb.sum()
        centered=controls-(wb @ controls);new_centered=new-(new_wb @ new)
        choices={"cap_reference":None,"momentum60":controls.momentum60,"book_to_price":controls.book_to_price,
            "quality_roe":controls.quality_roe,"negative_residual_volatility120":-controls.residual_volatility120}
        if list(choices)!=protocol["probes"]:
            raise ValueError("PROBE_PROTOCOL_CHANGED")
        weights={};complete=0
        for probe,score in choices.items():
            target=wb.copy() if score is None else optimize_daily(score,raw,controls,F,cov,specific,wb,limits,pd.Series(dtype=float))
            support=target[target>1e-8]
            if target.sum()-support.sum()>1e-6:
                raise ValueError("NONTRIVIAL_PROBE_MASS_CANNOT_BE_DROPPED")
            support/=support.sum();weights[probe]=support.reindex(raw.index,fill_value=0.)
            missing=set(support.index)-set(new.index)
            q=quote.loc[next_date].reindex(support.index)
            if missing or not np.isfinite(q[["open","close"]].to_numpy()).all() or (q[["open","close"]]<=0).any().any():
                unmeasurable.append({"from_date":date,"to_date":next_date,"probe":probe,"status":"MISSING_NEXT_PRICE_OR_CONTROL",
                    "missing_controls":sorted(missing)})
                continue
            if set(centered.columns)-set(new_centered.columns):
                unmeasurable.append({"from_date":date,"to_date":next_date,"probe":probe,"status":"MISSING_NEXT_FACTOR_COORDINATE"})
                continue
            # Allocation decided using date t only. Next open converts to
            # fractional reference shares; no close-t execution is claimed.
            forecast=support @ centered.reindex(support.index)
            end_value=support*q.close/q.open;end_weight=end_value/end_value.sum()
            outcome=end_weight @ new_centered.reindex(index=support.index,columns=centered.columns)
            for factor in centered.columns:
                errors.append({"from_date":date,"to_date":next_date,"probe":probe,"factor":factor,
                    "forecast":float(forecast[factor]),"outcome":float(outcome[factor]),"error":float(outcome[factor]-forecast[factor])})
            complete+=1
        pd.DataFrame(weights).to_parquet(args.output/f"diagnostic-weights-{date}.parquet")
        case_counts.append({"from_date":date,"to_date":next_date,"measurable_probes":complete,"expected_probes":len(choices)})
        print(json.dumps(case_counts[-1]),flush=True)
    table=pd.DataFrame(errors);table.to_parquet(args.output/"portfolio-exposure-errors.parquet",index=False)
    complete_training_days=[c["to_date"] for c in case_counts if c["to_date"]<=protocol["calibration_end"] and c["measurable_probes"]==c["expected_probes"]]
    training=table.loc[table.to_date.isin(complete_training_days)]
    envelope=fit_envelope(training,protocol["calibration_end"])
    (args.output/"observed-error-envelope.json").write_text(json.dumps(envelope,indent=2,allow_nan=False),encoding="utf-8")
    validation=table.loc[table.to_date.ge(protocol["validation_start"])]
    failures=validation.apply(lambda row:row.error>envelope[row.factor]["upper"]+1e-6 or -row.error>envelope[row.factor]["lower"]+1e-6,axis=1)
    breakdown={}
    for name,mask in {
        "continuous":~validation.factor.str.startswith(("industry_","board_","unknown_")),
        "industry_or_board":validation.factor.str.startswith(("industry_","board_")),
        "unknown":validation.factor.str.startswith("unknown_")}.items():
        breakdown[name]={"rows":int(mask.sum()),"outside_rows":int(failures.loc[mask].sum())}
    weight_group=validation.loc[validation.factor.str.startswith(("industry_","board_","unknown_"))]
    max_weight_overshoot=max([0.]+[max(row.error-envelope[row.factor]["upper"],-row.error-envelope[row.factor]["lower"]) for row in weight_group.itertuples()])
    validation_days=[]
    for case in case_counts:
        if case["to_date"]<protocol["validation_start"]:
            continue
        day=case["to_date"];bad=int(failures.loc[validation.to_date.eq(day)].sum())
        validation_days.append({"date":day,"complete":case["measurable_probes"]==case["expected_probes"],"outside_rows":bad,
            "passed":case["measurable_probes"]==case["expected_probes"] and bad==0})
    # Feasibility only on a predeclared first evaluation snapshot; no returns
    # enter this check, and an empty range is never clipped to force success.
    evaluation=root/"state/style-balance/expanded-style-10bps-20261102-b"
    C=pd.read_parquet(evaluation/"controls-20261012.parquet")
    wb_eval=pd.read_parquet(evaluation/"weights-20261012.parquet").benchmark
    old_limits=pd.Series({f:.01 if f.startswith("unknown_") else .03 if f.startswith(("industry_","board_")) else .1 for f in C.columns})
    lower,upper=robust_bounds(old_limits,envelope)
    crossed=lower.index[lower>upper+1e-10].tolist()
    feasibility="INFEASIBLE_FACTOR_INTERVALS" if crossed else None
    if not crossed:
        w=cp.Variable(len(C));active=w-wb_eval.reindex(C.index).to_numpy()
        problem=cp.Problem(cp.Minimize(0),[w>=0,w<=.15,cp.sum(w)==1,C.to_numpy().T @ active>=lower.reindex(C.columns).to_numpy(),C.to_numpy().T @ active<=upper.reindex(C.columns).to_numpy()])
        problem.solve(solver=cp.CLARABEL,max_iter=300)
        feasibility=problem.status
        if feasibility=="optimal":
            actual=C.to_numpy().T @ (w.value-wb_eval.reindex(C.index).to_numpy())
            if (not np.isfinite(w.value).all() or w.value.min() < -1e-6 or w.value.max()>.15+1e-6
                    or abs(w.value.sum()-1)>1e-6 or (actual<lower.reindex(C.columns).to_numpy()-1e-6).any()
                    or (actual>upper.reindex(C.columns).to_numpy()+1e-6).any()):
                raise ValueError("BUFFER_FEASIBILITY_INDEPENDENT_POSTCHECK_FAILED")
    result={"status":"COMPLETED_DIAGNOSTIC_BUFFER_TEST_NOT_ADOPTED","orders_enabled":False,"protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "probes":protocol["probes"],"transition_counts":case_counts,"unmeasurable_cases":unmeasurable,"train_days":int(training.to_date.nunique()),
        "validation_days":validation_days,"envelope_factor_count":len(envelope),"buffer_feasibility":feasibility,"crossed_intervals":crossed,
        "training_incomplete_days":[c for c in case_counts if c["to_date"]<=protocol["calibration_end"] and c["measurable_probes"]<c["expected_probes"]],
        "holdout_simultaneous_days_passed":sum(d["passed"] for d in validation_days),
        "validation_error_breakdown":breakdown,"maximum_weight_group_envelope_overshoot":max_weight_overshoot,
        "interval_coverage_is_not_actual_portfolio_limit_compliance":True,
        "rejection_reasons":[reason for condition,reason in [
            (bool(unmeasurable),"UNMEASURABLE_NEXT_PERIOD_PRICE_OR_RISK_CASES"),
            (not all(d["passed"] for d in validation_days),"OBSERVED_ENVELOPE_NOT_RELIABLE_ON_HELDOUT_DAYS"),
            (feasibility!="optimal","ROBUST_CONSTRAINTS_NOT_FEASIBLE")] if condition],
        "input_hashes":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,source/"results.json",evaluation/"controls-20261012.parquet",evaluation/"weights-20261012.parquet"]},
        "price_history_sha256":hashlib.sha256((args.output/"source-price-history.parquet").read_bytes()).hexdigest(),
        "buffer_adopted":False,"limitations":["synthetic as-of diagnostic portfolios, not learned alpha or actual feasible trades",
            "fractional next-open allocation excludes costs/lots/execution error","observed max envelope is not a coverage guarantee",
            "incomplete next-day risk observations are explicit, not filled or treated as safe","no October returns used for choice",
            "small September regime and revised fundamentals limit generalization"]}
    (args.output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["train_days","validation_days","buffer_feasibility","crossed_intervals"]}))


if __name__=="__main__":
    main()
